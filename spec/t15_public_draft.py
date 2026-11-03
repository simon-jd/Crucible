"""SGLang-loadable view of the public AngelSlim/Qwen3-4B_eagle3 draft (T15 phase 7).

The checkpoint names its class `Eagle3LlamaForCausalLM` (vLLM's name); SGLang 0.5.18 registers
the same model as `LlamaForCausalLMEagle3`. The view is a directory with a copy of config.json
with only that name changed and symlinks to every other file, so the weights are untouched.
usage: python -m spec.t15_public_draft SNAPSHOT_DIR OUT_DIR [COMPARE_DIR]
With COMPARE_DIR (our own export) it also prints the weight keys that differ.
"""

import json
import sys
from pathlib import Path

SGLANG_ARCH = "LlamaForCausalLMEagle3"


def make_view(src, out):
    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "config.json").read_text())
    old = cfg["architectures"]
    cfg["architectures"] = [SGLANG_ARCH]
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    for path in sorted(src.iterdir()):
        link = out / path.name
        if path.name != "config.json" and not link.exists():
            link.symlink_to(path.resolve())
    return {"architecture": [old[0], SGLANG_ARCH], "files": sorted(p.name for p in out.iterdir())}


def weight_shapes(directory):
    from safetensors import safe_open

    shapes = {}
    for path in sorted(Path(directory).glob("*.safetensors")):
        with safe_open(str(path), "pt") as fh:
            for key in fh.keys():  # noqa: SIM118 - safe_open handles are not iterable
                shapes[key] = list(fh.get_slice(key).get_shape())
    return shapes


if __name__ == "__main__":
    report = make_view(sys.argv[1], sys.argv[2])
    if len(sys.argv) > 3:
        ours, theirs = weight_shapes(sys.argv[3]), weight_shapes(sys.argv[2])
        report["only_public"] = sorted(set(theirs) - set(ours))
        report["only_ours"] = sorted(set(ours) - set(theirs))
        report["shape_mismatch"] = sorted(k for k in set(ours) & set(theirs) if ours[k] != theirs[k])
    print(json.dumps(report, indent=1))
