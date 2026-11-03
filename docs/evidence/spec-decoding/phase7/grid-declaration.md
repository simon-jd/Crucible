# T15 phase 7 spec-config grid, declared before any grid run (2026-11-01 05:00 PST)

Draft: the phase-7 full-size draft (`.cache/t15/draft-full-sglang`). Batch sizes 1 and 64 only.
Same harness as the main bench: SGLang, the same 128 held-out prompts, greedy, ≤256 new
tokens, closed loop, base and each config alternated with a server restart, two rounds.

| name | --speculative-num-steps | --speculative-eagle-topk | --speculative-num-draft-tokens |
|---|---|---|---|
| s3k1d4 (main bench default, phase 6) | 3 | 1 | 4 |
| s5k1d6 | 5 | 1 | 6 |
| s2k1d3 | 2 | 1 | 3 |
| s3k4d8 | 3 | 4 | 8 |

Command: `bash scripts/run_t15_gpu.sh bench docs/evidence/spec-decoding/phase7/grid 1,64 base
s5k1d6=.cache/t15/draft-full-sglang:5:1:6 s2k1d3=.cache/t15/draft-full-sglang:2:1:3
s3k4d8=.cache/t15/draft-full-sglang:3:4:8` (s3k1d4 comes from the main bench).
No other configs will be run or reported.
