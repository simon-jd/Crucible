"""Token-weighted GRPO steps and the fixed Torch 2.8 single-card FSDP2 factory.

Call runtime.configure_runtime() in the GPU process before importing this module.
The caller owns process/phase isolation and the shared scripts/with_gpu.sh lock.
"""

import math
from dataclasses import dataclass

import torch
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor

from kernels.ops import token_logprobs
from rl.learner.advantages import group_advantages
from rl.learner.grpo import grpo_loss
from rl.learner.staleness import filter_stale
from rl.rollout.sampler import Prompt, Sample, SamplingConfig


@dataclass(frozen=True)
class RolloutGroup:
    """One ordered completion group from one generate call; rewards share its order.

    batch_id identifies a generate call, group_id its prompt occurrence. Prompt IDs
    alone are insufficient: they may repeat between batches. Trainer's group_size
    is the expected G used to generate every group, not inferred from this tuple.
    """

    batch_id: str
    group_id: str
    prompt: Prompt
    samples: tuple[Sample, ...]
    rewards: tuple[float, ...]
    sampling: SamplingConfig


@dataclass(frozen=True)
class LearnerConfig:
    group_size: int
    microbatch_size: int = 1
    max_length: int = 256
    max_staleness: int = 1
    clip_eps: float = 0.2
    kl_coef: float = 0.001
    scale_by_std: bool = True

    def __post_init__(self):
        for name in ("group_size", "microbatch_size", "max_length"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.max_staleness) is not int or self.max_staleness < 0:
            raise ValueError("max_staleness must be a nonnegative integer")
        if not math.isfinite(self.clip_eps) or not 0 <= self.clip_eps < 1:
            raise ValueError("clip_eps must be in [0, 1)")
        if not math.isfinite(self.kl_coef) or self.kl_coef < 0:
            raise ValueError("kl_coef must be nonnegative and finite")


@dataclass(frozen=True)
class EncodedSample:
    input_ids: tuple[int, ...]
    prompt_length: int
    old_logprobs: tuple[float, ...]
    advantage: float


def encode_group(group, tokenizer, max_length, *, scale_by_std=True):
    # Match the rollout tokenizer's default encode behavior; never encode sample.text.
    prompt_ids = tuple(tokenizer.encode(group.prompt.text))
    if not prompt_ids:
        raise ValueError("prompt must encode to at least one token")
    advantages = group_advantages(
        torch.tensor(group.rewards, dtype=torch.float32), scale_by_std=scale_by_std
    ).tolist()
    records = []
    for sample, advantage in zip(group.samples, advantages, strict=True):
        ids = prompt_ids + sample.token_ids
        if len(ids) > max_length:
            raise ValueError("sequence length exceeds max_length; no silent truncation")
        records.append(EncodedSample(ids, len(prompt_ids), sample.logprobs, advantage))
    return records


def _validate_group(group, expected_size):
    if any(not isinstance(value, str) or not value for value in (group.batch_id, group.group_id)):
        raise ValueError("batch_id and group_id must be nonempty strings")
    if len(group.samples) != expected_size or len(group.rewards) != expected_size:
        raise ValueError("complete groups of the configured group_size are required")
    if len({sample.policy_version for sample in group.samples}) != 1:
        raise ValueError("group must have one policy version")
    if (group.sampling.temperature, group.sampling.top_p, group.sampling.top_k) != (1.0, 1.0, -1):
        raise ValueError("only neutral sampling has the required behavior/model logprob contract")
    if any(not math.isfinite(reward) for reward in group.rewards):
        raise ValueError("rewards must be finite")
    for sample in group.samples:
        if sample.prompt_id != group.prompt.prompt_id:
            raise ValueError("group prompt/sample mismatch")
        if not sample.token_ids or len(sample.token_ids) != len(sample.logprobs):
            raise ValueError("nonempty generated IDs and matching logprobs required")
        if any(type(token) is not int or token < 0 for token in sample.token_ids):
            raise ValueError("token IDs must be nonnegative integers")
        if any(not math.isfinite(lp) or lp > 1e-6 for lp in sample.logprobs):
            raise ValueError("old raw logprobs must be finite and nonpositive")


def _batch(records, device):
    batch_size = len(records)
    length = max(len(record.input_ids) for record in records)
    tokens = max(len(record.old_logprobs) for record in records)
    ids = torch.zeros((batch_size, length), dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    positions = torch.zeros((batch_size, tokens), dtype=torch.long, device=device)
    targets = torch.zeros_like(positions)
    old = torch.zeros((batch_size, tokens), dtype=torch.float32, device=device)
    mask = torch.zeros_like(old, dtype=torch.bool)
    for i, record in enumerate(records):
        size, count = len(record.input_ids), len(record.old_logprobs)
        ids[i, :size] = torch.tensor(record.input_ids, device=device)
        attention[i, :size] = 1
        positions[i, :count] = torch.arange(record.prompt_length - 1, size - 1, device=device)
        targets[i, :count] = ids[i, record.prompt_length : size]
        old[i, :count] = torch.tensor(record.old_logprobs, device=device)
        mask[i, :count] = True
    advantages = torch.tensor([record.advantage for record in records], device=device)
    return ids, attention, positions, targets, old, mask, advantages


def _score(model, ids, attention, positions, targets):
    logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits
    rows = torch.arange(ids.shape[0], device=ids.device)[:, None]
    # The op retains FP32 raw-logprob arithmetic without a full FP32 logits copy.
    return token_logprobs(logits[rows, positions], targets)


def _score_reference(model, ids, attention, positions, targets):
    """Original scoring path retained for matched performance/correctness evidence."""
    logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits
    rows = torch.arange(ids.shape[0], device=ids.device)[:, None]
    # Cast only completion positions to FP32, matching V1 raw_logprobs semantics.
    selected = logits[rows, positions].float()
    return selected.log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1)


def _check_finite(value, label):
    """Inspect supported SGD/AdamW state in place, never gather/copy DTensors."""
    if isinstance(value, torch.Tensor):
        local = value.to_local() if isinstance(value, DTensor) else value
        if not torch.isfinite(local).all():
            raise FloatingPointError(f"nonfinite {label}")
    elif isinstance(value, dict):
        for key, child in value.items():
            _check_finite(child, f"{label}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _check_finite(child, f"{label}[{index}]")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise FloatingPointError(f"nonfinite {label}")
    elif value is not None and not isinstance(value, (int, bool)):
        raise TypeError(f"unsupported state value for {label}: {type(value).__name__}")


class Learner:
    """One optimizer step per call; policy and optimizer remain in memory.

    Guarded policy/optimizer/policy_version are available only in a ready phase.
    Any uncertain execution/cleanup failure permanently invalidates the phase;
    last_committed_version is diagnostic and never identifies failed current weights.
    No rollback of failed state. Explicit owned RAM restore requires a fresh
    learner and never resets a failed phase. Only world_size=1 is supported.
    """

    def __init__(self, policy, reference, tokenizer, optimizer, config, *, policy_version=0):
        if type(policy_version) is not int or policy_version < 0:
            raise ValueError("policy_version must be a nonnegative integer")
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
            raise ValueError("multi-rank learner is not supported or validated")
        policy_parameters = {id(p) for p in policy.parameters()}
        if policy_parameters & {id(p) for p in reference.parameters()}:
            raise ValueError("reference must not share policy parameters")
        optimizer_parameters = {id(p) for group in optimizer.param_groups for p in group["params"]}
        if optimizer_parameters != policy_parameters:
            raise ValueError("optimizer must own exactly the policy parameters")
        if not isinstance(optimizer, (torch.optim.SGD, torch.optim.AdamW)):
            raise TypeError("only SGD/AdamW optimizer state structures are supported")
        self._policy = policy.eval()  # deterministic scoring, gradients remain enabled
        self.reference = reference.eval().requires_grad_(False)
        self.tokenizer, self._optimizer, self.config = tokenizer, optimizer, config
        self._last_committed_version = policy_version
        self._phase_status, self._failure_reason = "initializing", None
        self._handoff_fresh = True
        self.fsdp = isinstance(policy, FSDPModule)
        self._validate_state()
        self._phase_status = "ready"

    @property
    def phase_status(self):
        return self._phase_status

    @property
    def failure_reason(self):
        return self._failure_reason

    @property
    def last_committed_version(self):
        """Diagnostic only: failed/executing weights may differ from this version."""
        return self._last_committed_version

    def _require_ready(self):
        if self._phase_status != "ready":
            raise RuntimeError(f"learner phase is {self._phase_status}; state is not publishable")

    @property
    def policy(self):
        self._require_ready()
        return self._policy

    @property
    def optimizer(self):
        self._require_ready()
        return self._optimizer

    @property
    def policy_version(self):
        self._require_ready()
        return self._last_committed_version

    def _validate_state(self):
        for label, model in (("policy", self._policy), ("reference", self.reference)):
            for name, value in model.named_parameters():
                _check_finite(value, f"{label} parameter {name}")
            for name, value in model.named_buffers():
                _check_finite(value, f"{label} buffer {name}")
        for state in self._optimizer.state.values():
            _check_finite(state, "optimizer state")
        for group in self._optimizer.param_groups:
            _check_finite(
                {key: value for key, value in group.items() if key != "params"}, "optimizer options"
            )

    def _cleanup(self):
        # Attempt every cleanup independently; one failure must not hide another.
        actions = [lambda: self._optimizer.zero_grad(set_to_none=True)]
        if self.fsdp:
            actions.extend(
                [
                    lambda: self._policy.set_requires_gradient_sync(True),
                    lambda: self._policy.set_is_last_backward(True),
                ]
            )
        errors = []
        for action in actions:
            try:
                action()
            except BaseException as error:  # noqa: BLE001 - preserve and re-raise after cleanup
                errors.append(error)
        return errors

    def restore_handoff(self, fd, receipt, *, expected_version, binding):
        return self._restore_handoff(
            fd,
            receipt,
            expected_version=expected_version,
            binding=binding,
            stream_budget=None,
            stream=False,
        )

    def restore_handoff_stream(self, fd, receipt, *, expected_version, binding, budget):
        return self._restore_handoff(
            fd,
            receipt,
            expected_version=expected_version,
            binding=binding,
            stream_budget=budget,
            stream=True,
        )

    def _restore_handoff(self, fd, receipt, *, expected_version, binding, stream_budget, stream):
        """Restore a verified owned RAM snapshot into this fresh caller-owned learner.

        Public access/step stay blocked throughout validation, mutation and cleanup.
        Even a pre-mutation rejection invalidates this restore phase permanently.
        """
        self._require_ready()
        self._phase_status = "restoring"
        errors = []
        try:
            if not self._handoff_fresh or self._optimizer.state:
                raise ValueError("restore requires a fresh learner with empty optimizer state")
            self._handoff_fresh = False
            if type(expected_version) is not int or expected_version < 0:
                raise ValueError("invalid expected resume version")
            if self._last_committed_version != expected_version:
                raise ValueError("fresh learner version must match expected resume version")
            from rl.learner.state_handoff import _restore_into, _restore_rng

            restore, extra = _restore_into, {}
            if stream:
                from rl.learner.handoff_stream import SerialBudget, restore

                if type(stream_budget) is not SerialBudget:
                    raise TypeError("stream restore requires explicit SerialBudget")
                extra = {"budget": stream_budget}
            rng = restore(
                self._policy,
                self.reference,
                self._optimizer,
                self.config,
                fd,
                receipt,
                version=expected_version,
                binding=binding,
                **extra,
            )
            self._validate_state()
        except BaseException as error:  # noqa: BLE001 - preserve original restore cause
            errors.append(error)
        errors.extend(self._cleanup())
        if not errors:
            try:
                _restore_rng(rng)
            except BaseException as error:  # noqa: BLE001 - preserve RNG failure
                errors.append(error)
        if errors:
            self._phase_status = "failed"
            self._failure_reason = "; ".join(f"{type(error).__name__}: {error}" for error in errors)
            if len(errors) == 1:
                raise errors[0]
            raise BaseExceptionGroup("learner restore/cleanup failed", errors)
        self._last_committed_version = expected_version
        self._phase_status = "ready"

    def step(self, groups: list[RolloutGroup]):
        self._require_ready()
        # Input validation has no model/optimizer side effects: rejection is reusable.
        records, seen, dropped, accepted = [], set(), 0, 0
        for group in groups:
            _validate_group(group, self.config.group_size)
            identity = (group.batch_id, group.group_id)
            if identity in seen:
                raise ValueError("duplicate group identity in optimizer step")
            seen.add(identity)
            samples, count = filter_stale(
                group.samples, self.policy_version, self.config.max_staleness
            )
            dropped += count
            if samples:
                accepted += len(samples)
                records.extend(
                    encode_group(
                        group,
                        self.tokenizer,
                        self.config.max_length,
                        scale_by_std=self.config.scale_by_std,
                    )
                )
        total_tokens = sum(len(record.old_logprobs) for record in records)
        metrics = {
            "dropped_stale": dropped,
            "accepted_samples": accepted,
            "effective_tokens": total_tokens,
            "optimizer_steps": 0,
            "policy_version": self.policy_version,
            "loss": 0.0,
        }
        self._phase_status = "executing"
        self._handoff_fresh = False
        errors, loss_total = [], 0.0
        try:
            self._validate_state()
            if records:
                loss_total = self._execute_records(records, total_tokens)
        except BaseException as error:  # noqa: BLE001 - preserve and re-raise after cleanup
            errors.append(error)
        if records:
            errors.extend(self._cleanup())
        if not errors:
            try:
                # Includes real post-update parameters/moments, after cleanup.
                self._validate_state()
            except BaseException as error:  # noqa: BLE001 - preserve and re-raise after cleanup
                errors.append(error)
        if errors:
            self._phase_status = "failed"
            self._failure_reason = "; ".join(f"{type(error).__name__}: {error}" for error in errors)
            if len(errors) == 1:
                raise errors[0]
            raise BaseExceptionGroup("learner phase failed during execution/cleanup", errors)
        if records:
            self._last_committed_version += 1
            metrics.update(
                loss=loss_total, optimizer_steps=1, policy_version=self._last_committed_version
            )
        self._phase_status = "ready"
        return metrics

    def _execute_records(self, records, total_tokens):
        device = next(self._policy.parameters()).device
        self._optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        for start in range(0, len(records), self.config.microbatch_size):
            part = records[start : start + self.config.microbatch_size]
            last = start + self.config.microbatch_size >= len(records)
            if self.fsdp:
                self._policy.set_requires_gradient_sync(last)
                self._policy.set_is_last_backward(last)
            ids, attention, positions, targets, old, mask, adv = _batch(part, device)
            with torch.no_grad():
                ref = _score(self.reference, ids, attention, positions, targets)
            new = _score(self._policy, ids, attention, positions, targets)
            loss = grpo_loss(
                new, old, ref, adv, mask, clip_eps=self.config.clip_eps, kl_coef=self.config.kl_coef
            )
            weight = sum(len(record.old_logprobs) for record in part) / total_tokens
            (loss * weight).backward()
            loss_total += float(loss.detach()) * weight
        for parameter in self._policy.parameters():
            if parameter.grad is not None:
                _check_finite(parameter.grad, "gradient; optimizer step cancelled")
        self._optimizer.step()
        return loss_total


def load_fsdp2_learner(model_path, config, *, policy_version=0, learning_rate=1e-5):
    """Load offline on the caller-owned NCCL world1 process group.

    FP32 master parameters/Adam moments, BF16 forward parameters, FP32 reduction;
    frozen reference is BF16. Caller destroys the process group on phase exit.
    """
    if torch.__version__.split("+")[0] != "2.8.0":
        raise RuntimeError("this factory requires the validated Torch 2.8.0 runtime")
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() != 1:
        raise RuntimeError("initialize a caller-owned single-rank NCCL process group first")
    if torch.distributed.get_backend() != "nccl":
        raise RuntimeError("FSDP2 model factory requires NCCL/CUDA")
    from torch.distributed.device_mesh import init_device_mesh
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda", torch.cuda.current_device())
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    policy = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.float32, attn_implementation="sdpa"
    ).to(device)
    reference = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    mesh = init_device_mesh("cuda", (1,))
    precision = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in policy.model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=precision)
    fully_shard(policy, mesh=mesh, mp_policy=precision)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate, foreach=False)
    return Learner(policy, reference, tokenizer, optimizer, config, policy_version=policy_version)
