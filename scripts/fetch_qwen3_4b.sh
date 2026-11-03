#!/usr/bin/env bash
# Fetch Qwen/Qwen3-4B (BF16, about 8 GB) at a pinned revision into the shared HF cache and
# verify every file against the revision's tree listing (sha256 for LFS files, git blob
# SHA1 otherwise). Task 15 EAGLE-3 target. Shares: scripts.verify_hf_snapshot --preflight.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
rev=1cfa9a7208912126459214e8b04321603b3df60c
mkdir -p .cache/t15/model
tree=.cache/t15/model/tree-$rev.json
dest=/home/postedism/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/$rev
[[ -f $tree ]] || curl -fsSL "https://huggingface.co/api/models/Qwen/Qwen3-4B/tree/$rev" -o "$tree"
.venv/bin/python -m scripts.verify_hf_snapshot --tree "$tree" --dest "$dest" --preflight
mkdir -p "$dest"
for path in $(jq -r '.[] | select(.type == "file") | .path' "$tree"); do
  [[ -f "$dest/$path" ]] && continue
  curl -fsSL "https://huggingface.co/Qwen/Qwen3-4B/resolve/$rev/$path" -o "$dest/$path.part"
  mv "$dest/$path.part" "$dest/$path"
done
.venv/bin/python -m scripts.verify_hf_snapshot --tree "$tree" --dest "$dest"
