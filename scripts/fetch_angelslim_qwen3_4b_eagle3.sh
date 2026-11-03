#!/usr/bin/env bash
# Fetch the public EAGLE-3 draft AngelSlim/Qwen3-4B_eagle3 (Apache-2.0, Tencent; 0.44 GB)
# at a pinned revision into the shared HF cache and verify every file against the
# revision's tree listing, like scripts/fetch_qwen3_4b.sh. T15 phase 7 reference draft.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repo=AngelSlim/Qwen3-4B_eagle3
rev=fd331e59626c8e95c392381a16ee59d518727fbb
mkdir -p .cache/t15/public-draft
tree=.cache/t15/public-draft/tree-$rev.json
dest=/home/postedism/.cache/huggingface/hub/models--AngelSlim--Qwen3-4B_eagle3/snapshots/$rev
[[ -f $tree ]] || curl -fsSL "https://huggingface.co/api/models/$repo/tree/$rev" -o "$tree"
.venv/bin/python -m scripts.verify_hf_snapshot --tree "$tree" --dest "$dest" --preflight
mkdir -p "$dest"
for path in $(jq -r '.[] | select(.type == "file") | .path' "$tree"); do
  [[ -f "$dest/$path" ]] && continue
  curl -fsSL "https://huggingface.co/$repo/resolve/$rev/$path" -o "$dest/$path.part"
  mv "$dest/$path.part" "$dest/$path"
done
.venv/bin/python -m scripts.verify_hf_snapshot --tree "$tree" --dest "$dest"
