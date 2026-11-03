"""Opt-in source-process exit -> actual V1 full-vocabulary load/rollback acceptance."""

import json
import multiprocessing as mp
import os
from pathlib import Path

import pytest

from rl.sync.acceptance import PROTOCOL_SHA256, load_protocol, validate_reuse, write_binding

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("CRUCIBLE_SYNC_GPU") != "1",
        reason="use scripts/test_sync_gpu.sh under fixed GPU guard",
    ),
]
ROOT = Path(__file__).resolve().parents[3]
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MODEL = str(
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct"
    / "snapshots"
    / REVISION
)
RUN = ROOT / ".cache/task9/acceptance"
PROMPTS = ["The capital of France is", "One plus one is"]


def _source_process():
    from rl.learner.runtime import configure_runtime

    configure_runtime()
    import resource
    import tempfile

    import torch
    import torch.distributed as dist
    from transformers import AutoModelForCausalLM

    from rl.learner.train import LearnerConfig, RolloutGroup, load_fsdp2_learner
    from rl.rollout.sampler import Prompt, Sample, SamplingConfig
    from rl.sync.export import base_schema, export_ready_learner_bf16
    from rl.sync.nccl_broadcast import WeightSync
    from rl.sync.version import VersionStore

    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    rendezvous = tempfile.mktemp(prefix="sync-nccl-")
    dist.init_process_group(
        "nccl",
        init_method="file://" + rendezvous,
        rank=0,
        world_size=1,
        device_id=torch.device("cuda:0"),
    )
    try:
        learner = load_fsdp2_learner(
            MODEL, LearnerConfig(group_size=2, max_length=128), policy_version=7, learning_rate=1e-4
        )
        schema = base_schema(MODEL, torch.bfloat16)
        store = VersionStore(RUN / "versions", schema, MODEL)
        sync = WeightSync(store, timeout=300)
        before = (
            learner.policy.model.layers[0].self_attn.q_proj.weight.to_local().detach().cpu().clone()
        )
        initial = export_ready_learner_bf16(learner, sync)
        prompt = Prompt("train", PROMPTS[0])
        ids = learner.tokenizer(prompt.text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            initial_logits = learner.policy(**ids).logits[0, -1].float()
            logps = initial_logits.log_softmax(-1)
            selected = initial_logits.topk(2).indices.tolist()
        samples = tuple(
            Sample(prompt.prompt_id, "", (token,), (float(logps[token]),), 7) for token in selected
        )
        group = RolloutGroup(
            "controlled",
            "0",
            prompt,
            samples,
            (1.0, 0.0),
            SamplingConfig(max_tokens=1, max_model_len=128),
        )
        step = learner.step([group])
        assert step["policy_version"] == 8
        change = float(
            (
                learner.policy.model.layers[0].self_attn.q_proj.weight.to_local().detach().cpu()
                - before
            )
            .abs()
            .max()
        )
        assert change > 0
        updated = export_ready_learner_bf16(learner, sync)
        reference = {}
        mixed_difference = {}
        for version in (7, 8):
            checkpoint = store.checkpoint(version)
            # Reference uses the exact published BF16 bytes, full model vocabulary.
            model = (
                AutoModelForCausalLM.from_pretrained(
                    checkpoint.model_path,
                    torch_dtype=torch.bfloat16,
                    local_files_only=True,
                    attn_implementation="sdpa",
                )
                .to("cuda")
                .eval()
            )
            vectors = []
            for text in PROMPTS:
                inputs = learner.tokenizer(text, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    values = model(**inputs).logits[0, -1].float().cpu()
                    if version == 8:
                        mixed = learner.policy(**inputs).logits[0, -1].float().cpu()
                        mixed_difference[text] = float((mixed - values).abs().max())
                vectors.append(values)
            reference[str(version)] = torch.stack(vectors)
            del model
            torch.cuda.empty_cache()
        changed_logits = float((reference["8"] - reference["7"]).abs().max())
        assert changed_logits > 0.01
        torch.save(reference, RUN / "reference.pt")

        # Publication refuses an actually failed learner phase, including fresh version access.
        def fail_after_update(*_args):
            raise RuntimeError("sync acceptance injected post-update failure")

        hook = learner.optimizer.register_step_post_hook(fail_after_update)
        try:
            try:
                learner.step([group])
            except RuntimeError as exc:
                assert "post-update failure" in str(exc)
            else:
                raise AssertionError("expected failed phase")
        finally:
            hook.remove()
        try:
            export_ready_learner_bf16(learner, sync)
        except RuntimeError as exc:
            assert "phase" in str(exc)
        else:
            raise AssertionError("failed learner published")
        assert len(list(store.root.iterdir())) == 2
        evidence = {
            "source_pid": os.getpid(),
            "base_revision": REVISION,
            "reference_profile": "exact BF16 published weights; HF SDPA batch1",
            "source_step": step,
            "parameter_count": 494032768,
            "unique_tensors": len(schema),
            "dtype": "explicit FP32 master -> BF16 inference",
            "tensor_bytes_per_snapshot": updated.bytes,
            "largest_tensor_bytes": max(
                __import__("math").prod(shape) * 2 for shape, _ in schema.values()
            ),
            "push_seconds": [initial.seconds, updated.seconds],
            "parameter_max_change": change,
            "full_logits_max_change": changed_logits,
            "mixed_compute_vs_published_bf16_max_abs": mixed_difference,
            "failed_learner_publication_rejected": True,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nccl": torch.cuda.nccl.version(),
            "device": torch.cuda.get_device_name(),
        }
        (RUN / "source.json").write_text(json.dumps(evidence, indent=2))
        print(json.dumps({"source": evidence}), flush=True)
    finally:
        dist.destroy_process_group()
        Path(rendezvous).unlink(missing_ok=True)


def _create_source_snapshots():
    from rl.learner.runtime import configure_runtime

    configure_runtime()
    import subprocess

    memory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
        text=True,
        timeout=10,
    )
    total, free = (int(value.strip()) for value in memory.strip().split(","))
    assert free >= total * 0.82, f"insufficient GPU headroom under guard: {memory.strip()}"
    print(json.dumps({"gpu_preflight_total_mib": total, "free_mib": free}), flush=True)
    RUN.mkdir(parents=True, exist_ok=True)
    assert not (RUN / "versions").exists(), "clean only own prior acceptance snapshots first"
    child = mp.get_context("spawn").Process(target=_source_process)
    child.start()
    try:
        child.join(900)
        assert not child.is_alive() and child.exitcode == 0
    finally:
        if child.is_alive():
            child.terminate()
            child.join(10)
        if child.is_alive():
            child.kill()
            child.join(10)
        assert not child.is_alive()
    # Source child exited: no learner CUDA context is live when vLLM starts.
    print(
        json.dumps(
            {"phase": "source_joined_before_vllm", "pid": child.pid, "exitcode": child.exitcode}
        ),
        flush=True,
    )
    return child.pid, child.exitcode


def _diagnostic_reference_process(precision, prompts=None, output_name=None):
    from rl.learner.runtime import configure_runtime

    configure_runtime()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from rl.sync.export import base_schema
    from rl.sync.version import VersionStore

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    store = VersionStore(RUN / "versions", base_schema(MODEL, torch.bfloat16), MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    dtype = torch.bfloat16 if precision == "bf16_sdpa" else torch.float32
    prompts = PROMPTS if prompts is None else prompts
    result = {}
    for version in (7, 8):
        checkpoint = store.checkpoint(version)
        model = (
            AutoModelForCausalLM.from_pretrained(
                checkpoint.model_path,
                local_files_only=True,
                torch_dtype=dtype,
                attn_implementation="sdpa",
            )
            .to("cuda")
            .eval()
        )
        vectors, inputs_seen = [], []
        for text in prompts:
            inputs = tokenizer(text, return_tensors="pt").to("cuda")
            with torch.no_grad():
                vectors.append(model(**inputs).logits[0, -1].float().cpu())
            inputs_seen.append(inputs["input_ids"][0].cpu().tolist())
        result[str(version)] = torch.stack(vectors)
        result["input_ids"] = inputs_seen
        result["revisions_" + str(version)] = checkpoint.revision
        del model
        torch.cuda.empty_cache()
    result["matmul_precision"] = torch.get_float32_matmul_precision()
    result["allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
    torch.save(result, RUN / ((output_name or precision) + ".pt"))


PROTOCOL = ROOT / "docs/evidence/raw/task9-confirmation-protocol.json"


def _confirm_snapshots(binding, *, exercise_rollback=False):
    from rl.learner.runtime import configure_runtime

    configure_runtime()
    import hashlib
    import subprocess

    import torch

    from rl.sync.export import base_schema
    from rl.sync.version import VersionStore

    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == PROTOCOL_SHA256
    protocol = load_protocol(PROTOCOL)
    memory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
        text=True,
        timeout=10,
    )
    total, free = (int(value.strip()) for value in memory.strip().split(","))
    assert free >= total * 0.82, memory
    print(json.dumps({"confirmation_total_mib": total, "free_mib": free}), flush=True)
    store = VersionStore(RUN / "versions", base_schema(MODEL, torch.bfloat16), MODEL)
    snapshots = {str(v): store.checkpoint(v).revision for v in protocol["versions"]}
    assert snapshots == binding["snapshots"]
    assert validate_reuse(RUN, PROTOCOL, snapshots) == binding
    child = mp.get_context("spawn").Process(
        target=_diagnostic_reference_process,
        args=("bf16_sdpa", protocol["prompts"], "confirmation-reference"),
    )
    child.start()
    try:
        child.join(300)
        assert not child.is_alive() and child.exitcode == 0
    finally:
        if child.is_alive():
            child.terminate()
            child.join(10)
        if child.is_alive():
            child.kill()
            child.join(10)
        assert not child.is_alive()
    print(
        json.dumps({"confirmation_reference_joined": child.pid, "exitcode": child.exitcode}),
        flush=True,
    )
    import numpy as np

    from rl.rollout._worker import VLLMWorker
    from rl.rollout.engine import RolloutEngine
    from rl.rollout.sampler import Prompt, SamplingConfig

    reference = torch.load(RUN / "confirmation-reference.pt", weights_only=True)
    for version in ("7", "8"):
        assert reference["revisions_" + version] == snapshots[version]
    prompts = [Prompt(str(i), text) for i, text in enumerate(protocol["prompts"])]
    workers, load_order = [], []

    def factory(checkpoint, config, **kwargs):
        worker = VLLMWorker(checkpoint, config, **kwargs)
        workers.append(worker._process)
        load_order.append(checkpoint.policy_version)
        if exercise_rollback and checkpoint.policy_version == 8 and load_order.count(8) == 1:
            error = RuntimeError("injected candidate failure after actual load")
            error.worker, error.cleanup_confirmed = worker, False
            try:
                worker.close()
            except BaseException as cleanup_error:
                raise error from cleanup_error
            error.cleanup_confirmed = True
            raise error
        return worker

    actual, metadata = {}, {}
    with RolloutEngine(
        store.checkpoint(7),
        SamplingConfig(max_tokens=1, max_model_len=128),
        weight_store=store,
        _backend_factory=factory,
        _diagnostic_logits=True,
    ) as engine:
        for version in (7, 8):
            if version == 8:
                if exercise_rollback:
                    with pytest.raises(RuntimeError, match="injected candidate"):
                        engine.load_weights(8)
                    assert engine.policy_version == store.active_version == 7
                    rollback = []
                    for offset in range(0, len(prompts), 2):
                        rollback.extend(engine.full_logits(prompts[offset : offset + 2]))
                    torch.testing.assert_close(torch.tensor(rollback), actual["7"], atol=0, rtol=0)
                engine.load_weights(8)
            vectors, token_ids = [], []
            # Preserve the two-prompt batch shape used for numerical calibration.
            for offset in range(0, len(prompts), 2):
                vectors.extend(engine.full_logits(prompts[offset : offset + 2]))
                details = engine._backend.last_logits_metadata
                assert details["vocab_size"] == 151936
                token_ids.extend(details["prompt_token_ids"])
            actual[str(version)] = torch.tensor(vectors)
            assert token_ids == reference["input_ids"]
            assert actual[str(version)].shape == reference[str(version)].shape == (6, 151936)
            metadata[str(version)] = {
                "input_ids": token_ids,
                "positions": [len(ids) - 1 for ids in token_ids],
            }
            assert engine.policy_version == version
    assert all(not worker.is_alive() and worker.exitcode == 0 for worker in workers)
    limits = protocol["acceptance_per_prompt_per_version"]
    rows, failures, arrays = [], [], {}
    for version in ("7", "8"):
        arrays["vllm_v" + version] = actual[version].numpy()
        arrays["hf_bf16_sdpa_v" + version] = reference[version].numpy()
        for index, prompt in enumerate(protocol["prompts"]):
            observed, expected = actual[version][index], reference[version][index]
            error = (observed - expected).abs()
            finite = bool(torch.isfinite(observed).all() and torch.isfinite(expected).all())
            bad = int(
                (
                    error
                    > limits["absolute_tolerance"] + limits["relative_tolerance"] * expected.abs()
                ).sum()
            )
            row = {
                "version": int(version),
                "prompt_index": index,
                "prompt": prompt,
                "coordinates": error.numel(),
                "finite": finite,
                "max_abs": float(error.max()),
                "mean_abs": float(error.mean()),
                "rms": float(error.square().mean().sqrt()),
                "failed_coordinates": bad,
            }
            row["passed"] = (
                finite
                and bad == 0
                and row["rms"] <= limits["maximum_rms_error"]
                and row["mean_abs"] <= limits["maximum_mean_absolute_error"]
            )
            rows.append(row)
            if not row["passed"]:
                failures.append(row)
    dump = RUN / "confirmation-logits.npz"
    np.savez_compressed(dump, **arrays)
    result = {
        "protocol_sha256": PROTOCOL_SHA256,
        "snapshots": snapshots,
        "run_binding": binding,
        "load_order": load_order,
        "rollback_checked": exercise_rollback,
        "reference_batch_size": 1,
        "rows": rows,
        "metadata": metadata,
        "vocab_size": 151936,
        "batch_size": 2,
        "reference_precision": "BF16 SDPA exact published weights",
        "allow_tf32": reference["allow_tf32"],
        "matmul_precision": reference["matmul_precision"],
        "reference_process": {"pid": child.pid, "exitcode": child.exitcode},
        "workers": [{"pid": worker.pid, "exitcode": worker.exitcode} for worker in workers],
        "dump_sha256": hashlib.sha256(dump.read_bytes()).hexdigest(),
        "dump_bytes": dump.stat().st_size,
        "passed": not failures,
    }
    (RUN / "confirmation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    assert not failures, failures


def test_fresh_or_explicit_reuse_acceptance():
    """New orchestration: component evidence exists; this combined path is not rerun."""
    from rl.learner.runtime import configure_runtime

    configure_runtime()
    import torch

    from rl.sync.export import base_schema
    from rl.sync.version import VersionStore

    protocol = load_protocol(PROTOCOL)
    mode = os.environ.get("CRUCIBLE_SYNC_MODE", "fresh")
    if mode not in ("fresh", "reuse"):
        raise ValueError("expected fresh or explicit reuse")
    if mode == "fresh":
        if RUN.exists() and any(RUN.iterdir()):
            raise ValueError("fresh run requires an empty owned acceptance directory")
        pid, exitcode = _create_source_snapshots()
    store = VersionStore(RUN / "versions", base_schema(MODEL, torch.bfloat16), MODEL)
    snapshots = {str(v): store.checkpoint(v).revision for v in protocol["versions"]}
    if mode == "fresh":
        binding = write_binding(RUN, PROTOCOL, snapshots, source_pid=pid, source_exitcode=exitcode)
    else:
        binding = validate_reuse(RUN, PROTOCOL, snapshots)
    _confirm_snapshots(binding, exercise_rollback=mode == "fresh")
