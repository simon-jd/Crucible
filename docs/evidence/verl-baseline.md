# T11 verl comparison: skipped on disk and version conflicts (2026-10-29)

**Update (2026-11-02, phase 7):** the comparison was run. With the 40 GB share, `.venv-verl`
(verl 0.9.1, vLLM 0.24.0, its own torch 2.11) was built separately, so neither version
conflict below applies. Results: [t11-verl/README.md](t11-verl/README.md). The rest of this file
is the 2026-10-29 record.

The plan's T11 baseline is the same GRPO budget run with verl (or prime-rl). It was not run.
The GPU is not the reason: this project's own 0.5B GRPO runs fit one RTX 4090 (peak CUDA
allocation 14.8 GB, [rl-grpo.md](rl-grpo.md)). The reasons are disk and dependency pins.

## Facts

- Releases (api.github.com, [list](t11-verl/verl-releases.txt)): latest v0.9.1 (2026-09-20).
- verl v0.9.1 ([setup.py](t11-verl/verl-v0.9.1-setup.py.txt)): `transformers>=5.5.3,!=5.6.0,<5.11`,
  `numpy>=2.0.0`, `ray[default]>=2.41.0`; rollout extra either `vllm>=0.18.0` or
  `sglang[srt,openai]==0.5.8` with `torch==2.9.1`.
  - `.venv-ml` has vLLM 0.10.2, torch 2.8.0, transformers 4.55.4: vLLM and transformers too old.
  - `.venv-specforge` has sglang 0.5.18, torch 2.13.0, transformers 5.12.1: all three outside the pins.
- verl v0.6.1 ([setup.py](t11-verl/verl-v0.6.1-setup.py.txt)): `vllm>=0.8.5,<=0.11.0` and
  `torch==2.8.0` match `.venv-ml`, but `numpy<2.0.0` conflicts with the numpy 2.2.6 installed
  there, so it cannot simply reuse `.venv-ml` either.
- So verl needs its own complete torch + CUDA + rollout-engine environment. The two such
  environments here measure 9,787,064,817 bytes (`.venv-ml`) and 9,178,684,958 bytes
  (`.venv-specforge`) ([du](t11-verl/disk-2026-10-29.txt)).
- The project uses 27,354,106,239 bytes of its 28 GB share after T15: 0.65 GB left.

## What it needs from the user

About 9–10 GB inside the share, for example by approving deletion of T15's own outputs
`.cache/t15/features` (4.21 GB) and `.cache/t15/train` (2.10 GB) plus one of the two
environments above, or by raising the share. Then: install verl v0.9.1 with its vLLM extra into
`.venv-verl` from pypi.org, and run the T11 base config (Qwen2.5-0.5B-Instruct, 6,400 samples)
under the same GPU lock.
