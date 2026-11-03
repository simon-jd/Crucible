"""Evaluable view of verl's saved HF checkpoint for .venv-ml (transformers 4.55).

verl 0.9.1 saves config.json with transformers 5.9, which moves rope_theta into
rope_parameters; transformers 4.55 and vLLM 0.10.2 then read rope_theta as the default
10000 instead of Qwen2.5's 1e6. Its tokenizer files do not load in 4.55 at all. The view
links the checkpoint's *.safetensors (the trained weights) and the base model's other
files (config, tokenizer, generation config; none of them is trained). The architecture
fields of the two configs must match.
usage: python -m rl.verl.ckpt_view CKPT_DIR BASE_DIR OUT_DIR
"""

import json
import sys
from pathlib import Path

ARCH_KEYS = (
    "model_type", "hidden_size", "intermediate_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "vocab_size", "tie_word_embeddings",
)


def make_view(ckpt, base, out):
    ckpt, base, out = Path(ckpt).resolve(), Path(base).resolve(), Path(out)
    ours = json.loads((ckpt / "config.json").read_text())
    theirs = json.loads((base / "config.json").read_text())
    differ = [k for k in ARCH_KEYS if k in ours and k in theirs and ours[k] != theirs[k]]
    if differ:
        raise ValueError(f"checkpoint and base differ in {differ}")
    weights = sorted(ckpt.glob("*.safetensors"))
    if not weights:
        raise ValueError(f"no *.safetensors in {ckpt}")
    others = [p for p in sorted(base.iterdir()) if p.suffix != ".safetensors" and p.is_file()]
    out.mkdir(parents=True, exist_ok=True)
    for path in weights + others:
        link = out / path.name
        if link.is_symlink():
            link.unlink()
        link.symlink_to(path)
    return out


if __name__ == "__main__":
    print(make_view(*sys.argv[1:4]))
