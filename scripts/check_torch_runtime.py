"""Tiny CPU/CUDA environment correctness check; not a model or performance experiment.

CUDA invocation must run in the foreground through scripts/with_gpu.sh.
"""

import argparse
import json
import platform
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    import torch
    import triton

    device = "cuda" if args.cuda else "cpu"
    result = {
        "kind": "environment-correctness-smoke",
        "performance_claim": False,
        "model": None,
        "device": device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "torch_build_cuda": torch.version.cuda,
    }
    if args.cuda:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise SystemExit("exactly one available local CUDA GPU is required")
        props = torch.cuda.get_device_properties(0)
        if "RTX 4090" not in props.name:
            raise SystemExit("current approved local profile requires RTX 4090")
        result.update(
            gpu=props.name,
            compute_capability=[props.major, props.minor],
            total_memory_bytes=props.total_memory,
            nccl=list(torch.cuda.nccl.version()),
            driver=subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
            ).strip(),
        )

    dtype = torch.bfloat16 if args.cuda else torch.float32
    values = torch.ones((16, 16), dtype=dtype, device=device, requires_grad=True)
    output = values @ torch.ones_like(values)
    output.float().sum().backward()
    if not torch.equal(output, torch.full_like(output, 16)):
        raise SystemExit("matrix product mismatch")
    if not torch.equal(values.grad, torch.full_like(values, 16)):
        raise SystemExit("gradient mismatch")
    if args.cuda:
        torch.cuda.synchronize()
    result.update(dtype=str(dtype), forward_ok=True, backward_ok=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
