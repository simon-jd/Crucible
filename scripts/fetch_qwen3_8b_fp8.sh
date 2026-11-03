#!/usr/bin/env bash
# Fetch Qwen/Qwen3-8B-FP8 (pre-quantized FP8 checkpoint, 9.45 GB) at a pinned revision
# into the shared HF cache, then verify every file against the revision's tree
# listing (sha256 for LFS files, git blob SHA1 otherwise). Task 13.
# Disk shares: checked by scripts.verify_hf_snapshot --preflight.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
rev=220b46e3b2180893580a4454f21f22d3ebb187d3
tree=.cache/t13/fp8ckpt/tree-$rev.json
dest=/home/postedism/.cache/huggingface/hub/models--Qwen--Qwen3-8B-FP8/snapshots/$rev
[[ -f $tree ]] || curl -fsSL "https://huggingface.co/api/models/Qwen/Qwen3-8B-FP8/tree/$rev" -o "$tree"
.venv/bin/python -m scripts.verify_hf_snapshot --tree "$tree" --dest "$dest" --preflight
mkdir -p "$dest"
for path in $(jq -r '.[] | select(.type == "file") | .path' "$tree"); do
  [[ -f "$dest/$path" ]] && continue
  curl -fsSL "https://huggingface.co/Qwen/Qwen3-8B-FP8/resolve/$rev/$path" -o "$dest/$path.part"
  mv "$dest/$path.part" "$dest/$path"
done
.venv/bin/python -m scripts.verify_hf_snapshot --tree "$tree" --dest "$dest"
