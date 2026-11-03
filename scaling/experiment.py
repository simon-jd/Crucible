"""Offline scaling matrix and shared three-round result gate. Never executes jobs."""

import argparse
import hashlib
import json
from pathlib import Path

from scaling.analyze_nsys import mfu
from scripts.experiment_results import summarize

COMMIT = "29624e37fa6dd319fd671b009338651b5c665da2"
ROOT = Path(__file__).resolve().parents[1]


def scaling_metrics(
    tokens_per_s, params, seq_len, n_gpus, peak_flops, reference_tokens_per_s, reference_gpus=8
):
    """Strong-scaling efficiency vs same-precision 8-GPU global throughput.

    Caller must match global batch, data, model, timing scope and hardware. A
    changing local batch/accumulation is expected with fixed global batch 256.
    Never compare an FP8 scale point against a BF16 scaling reference.
    """
    reference_mfu = mfu(reference_tokens_per_s, params, seq_len, reference_gpus, peak_flops)
    current_mfu = mfu(tokens_per_s, params, seq_len, n_gpus, peak_flops)
    if reference_mfu <= 0:
        raise ValueError("positive same-precision reference throughput required")
    return {
        "tokens_per_s_per_gpu": tokens_per_s / n_gpus,
        "mfu": current_mfu,
        "scaling_efficiency": current_mfu / reference_mfu,
    }


def _run(comparison, round_id, variant):
    precision = comparison["precision"]
    if comparison["purpose"] == "precision":
        precision = "bf16" if variant == "baseline" else "fp8"
    ep = comparison.get("candidate_ep", 1) if variant == "candidate" else 1
    config = (
        "moe_small_ep.toml"
        if comparison["purpose"] == "expert_parallel"
        else "llama3_8b_fsdp_tp.toml"
    )
    run_id = f"{comparison['id']}-round{round_id}-{variant}"
    folder = f"artifacts/scaling/{run_id}"
    nodes = []
    for rank in range(comparison["gpus"] // 8):
        trace = f"{folder}/node{rank}"
        argv = [
            "nsys",
            "profile",
            "--trace=cuda,nvtx,osrt",
            "--sample=none",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            f"--output={trace}",
            "python",
            "-m",
            "torch.distributed.run",
            "--nnodes",
            str(comparison["gpus"] // 8),
            "--nproc-per-node",
            "8",
            "--node-rank",
            str(rank),
            "--rdzv-backend",
            "c10d",
            "--rdzv-endpoint",
            "REQUIRED_HOST:PORT",
            "--rdzv-id",
            run_id,
            "-m",
            "scaling.train",
            "--job.config_file",
            f"scaling/configs/{config}",
            "--job.dump_folder",
            folder,
        ]
        if precision == "fp8":
            argv += ["--model.converters", "float8"]
        if comparison["purpose"] == "expert_parallel":
            argv += ["--parallelism.expert_parallel_degree", str(ep)]
        env = {
            "NCCL_DEBUG": "INFO",
            "NCCL_DEBUG_SUBSYS": "INIT,COLL",
            "NCCL_DEBUG_FILE": f"{folder}/nccl.%h.%p.log",
            "CRUCIBLE_RUN_ID": run_id,
            "CRUCIBLE_OUTPUT": folder,
            "CRUCIBLE_PROFILE": "1",
            "PYTHONHASHSEED": "42",
        }
        if comparison["purpose"] == "tuning" and variant == "candidate":
            # NCCL >=2.24 collective-specific filter: keep FSDP AllGather/ReduceScatter
            # on their default algorithms; Tree is an AllReduce-only hypothesis.
            env["NCCL_ALGO"] = "allreduce:Tree"
        nodes.append(
            {
                "node_rank": rank,
                "env": env,
                "argv": argv,
                "setup": {"create_directory": folder, "working_directory": str(ROOT)},
                "export_argv": [
                    "nsys",
                    "export",
                    "--type=sqlite",
                    f"--output={trace}.sqlite",
                    f"{trace}.nsys-rep",
                ],
                "analyze_argv": ["python", "-m", "scaling.analyze_nsys", f"{trace}.sqlite"],
            }
        )
    return {
        "round": round_id,
        "variant": variant,
        "fp8": precision == "fp8",
        "run_id": run_id,
        "valid": None,
        "metrics": None,
        "nodes": nodes,
    }


def build_plan():
    comparisons = []
    for gpus in (8, 16, 32):
        comparisons.append(
            {
                "id": f"llama-{gpus}-precision",
                "purpose": "precision",
                "gpus": gpus,
                "precision": "bf16-vs-fp8",
            }
        )
        for precision in ("bf16", "fp8"):
            comparisons.append(
                {
                    "id": f"llama-{gpus}-{precision}-tuning",
                    "purpose": "tuning",
                    "gpus": gpus,
                    "precision": precision,
                }
            )
    for ep in (2, 4):
        comparisons.append(
            {
                "id": f"moe-8-ep{ep}",
                "purpose": "expert_parallel",
                "gpus": 8,
                "precision": "bf16",
                "candidate_ep": ep,
            }
        )
    for comparison in comparisons:
        comparison["conditions"] = {
            "gpus": comparison["gpus"],
            "global_batch_size": 256,
            "seq_len": 1024 if comparison["purpose"] == "expert_parallel" else 8192,
            "steps": 200,
            "warmup_steps": 50,
            "seed": 42,
            "hardware_inventory": None,
            "dataset_revision": None,
            "heldout_revision": None,
            "asset_revision_and_sha256": None,
            "timing_scope": "synchronized_train_step",
            "upstream_commit": COMMIT,
        }
        comparison["runs"] = [
            _run(comparison, n, v) for n in range(1, 4) for v in ("baseline", "candidate")
        ]
    return {
        "schema": 1,
        "status": "pending-authorization",
        "verification": "未在真实多机上验证",
        "upstream": {
            "project": "pytorch/torchtitan",
            "tag": "v0.2.0",
            "commit": COMMIT,
            "torch": "2.10.0.dev20251019+cu126",
            "torchao": "0.15.0.dev20251015+cu126",
            "environment_status": "candidate-unverified",
        },
        "config_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "scaling/configs").glob("*.toml"))
        },
        "required_before_execution": {
            "separate_budget_authorization": None,
            "image_digest": None,
            "driver_cuda_nccl_ofed_observations": None,
            "nccl_min_version_for_collective_algo_filter": "2.24",
            "model_and_data_revision": None,
            "precision_specific_peak_flops_per_gpu": None,
            "heldout_metric_and_max_delta": None,
            "rendezvous_endpoint": None,
            "node_local_tp_rank_mapping": None,
            "cpu_numa_affinity_and_clean_nccl_environment": None,
        },
        "stop_conditions": [
            "authorization or timebox exhausted (max 4 hours per job)",
            "health gate fails / XID / NIC carrier lost / nonfinite loss",
            "step-time ratio >1.2 for three fresh comparable samples: investigate",
            "less than 50 GB disk free or artifact/storage quota exceeded",
        ],
        "comparisons": comparisons,
    }


def summarize_results(records):
    required = {"tokens_per_s_per_gpu", "mfu", "scaling_efficiency"}
    if not isinstance(records, list) or not records:
        raise ValueError("completed result records required; plans are not results")
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("metrics"), dict):
            raise TypeError("completed measured metrics required")
        if not required <= record["metrics"].keys():
            raise ValueError("report tokens_per_s_per_gpu, mfu and scaling_efficiency together")
    return summarize(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "summarize"])
    parser.add_argument("records", nargs="?", type=Path)
    args = parser.parse_args()
    try:
        if args.action == "plan":
            result = build_plan()
        elif args.records is None:
            raise ValueError("summarize requires measured JSON records")
        else:
            result = summarize_results(json.loads(args.records.read_text()))
    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, f"invalid experiment: {error}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
