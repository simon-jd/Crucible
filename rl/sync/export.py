"""Explicit FP32 learner -> BF16 inference export, owned by a single ready phase."""

from pathlib import Path


def base_schema(model_path, dtype=None):
    """Derive full unique state from all base safetensors (tied alias in config)."""
    import torch
    from safetensors import safe_open

    schema = {}
    types = {
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "F16": torch.float16,
        "I64": torch.int64,
        "I32": torch.int32,
        "BOOL": torch.bool,
    }
    for file in sorted(Path(model_path).glob("*.safetensors")):
        with safe_open(file, framework="pt", device="cpu") as reader:
            for name in reader.keys():  # noqa: SIM118 - safetensors is not iterable
                if name in schema:
                    raise ValueError("duplicate base parameter")
                tensor = reader.get_slice(name)
                original = types[tensor.get_dtype()]
                target = dtype if dtype is not None and original.is_floating_point else original
                schema[name] = (tuple(tensor.get_shape()), target)
    if not schema:
        raise ValueError("base safetensors required")
    return schema


def export_ready_learner_bf16(learner, sync):
    """Fresh guarded access, explicit cast, and final guard before store commit.

    The caller owns the entire phase; concurrent step calls are forbidden. No
    cached learner alias or last_committed_version is accepted. Persistent state
    must match the complete base schema, allowing only the Qwen tied-head alias.
    Each tensor is materialized/copied individually. No optimizer state is saved.
    """
    import torch
    from torch.distributed.tensor import DTensor

    version = learner.policy_version  # raises immediately outside ready
    policy = learner.policy
    state = policy.state_dict()
    expected = set(sync.store.schema)
    aliases = set(state) - expected
    if aliases:
        if aliases != {"lm_head.weight"} or not getattr(
            policy.config, "tie_word_embeddings", False
        ):
            raise ValueError("unrecognized state aliases")
        if policy.get_output_embeddings().weight is not policy.get_input_embeddings().weight:
            raise ValueError("head alias is not tied to input embeddings")
    if expected - set(state):
        raise ValueError("learner state omits base parameters")

    def tensors():
        for name in sorted(expected):
            # Recheck public guards; never publish after an intervening failed step.
            if learner.policy_version != version or learner.policy is not policy:
                raise RuntimeError("learner changed during publication")
            value = state[name]
            if isinstance(value, DTensor):
                value = value.full_tensor()
            value = value.detach()
            target = torch.bfloat16 if value.is_floating_point() else value.dtype
            if sync.store.schema[name][1] != target:
                raise ValueError("store must explicitly declare BF16 inference schema")
            yield name, value.to(device="cpu", dtype=target).contiguous()
        if learner.policy_version != version or learner.policy is not policy:
            raise RuntimeError("learner changed during publication")

    return sync.push(tensors(), version)
