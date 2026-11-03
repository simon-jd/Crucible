# T11 verl comparison: the base config in verl 0.9.1 on one RTX 4090 (2026-11-02)

The T11 base run [eos-sync-k0-s17](../rl-grpo-p4/eos-sync-k0-s17/) (`rl.local.grpo_run`, sync k=0,
EOS penalty, seed 17) was run again in verl 0.9.1. Both "after" models were scored by the same
eval script, `rl.local.eval_checkpoint`.

## Setup

- **Environment:** `.venv-verl`, built by `scripts/bootstrap_verl.sh` from pypi.org only:
  verl 0.9.1 (extras vllm and math), torch 2.11.0+cu130, vLLM 0.24.0, transformers 5.9.0,
  ray 2.59.0.
  - Size: 10,149,079,674 bytes after install and 10,230,032,330 bytes after the runs
    (Python byte-code caches).
- **Run:** `bash scripts/with_gpu.sh -- bash scripts/run_t11_verl.sh docs/evidence/t11-verl/verl-run`,
  one GPU job.
  - Settings: [overrides](verl-run/overrides.txt).
  - Matched to the base run: the same model, the same 800 training prompts in the same order
    (data sha256 bb491e55…), chat template and reward (`verify_math` with the EOS penalty,
    `rl/verl/reward.py`).
  - Also matched: 100 steps × 8 prompts × 8 samples, temperature 1, 384 response tokens,
    GRPO, AdamW lr 3e-6, clip 0.2, k3 KL 0.001, token-mean, one update per batch.
- **Eval:** the same 200 GSM8K and 100 MATH-500 held-out problems, 4 samples each, 512 tokens,
  seed 17, vLLM share 0.15.
  - The script reproduces the base run's "before" numbers exactly when given the base model
    (GSM8K 0.42625 with 50 truncated, MATH-500 0.2875 with 162 truncated;
    [eval-base-sanity](eval-base-sanity/summary.json)).
- **Machine:** 32 cores (`host-*.txt` print `nproc 4` because GNU nproc honours
  OMP_NUM_THREADS=4). Load average was 2.9 before and 3.7 after, and other sessions were
  running and sharing the GPU lock queue.

### Differences that remain

They are listed in the header of [scripts/run_t11_verl.sh](../../../scripts/run_t11_verl.sh).
In short:

1. **Seeds:** rollouts are seeded per engine, not per request, so the sampled completions
   differ.
2. **Advantages:** verl's GRPO std uses Bessel's correction, so advantages are 0.935× ours.
3. **Clamps:** verl clamps the log-ratio and the KL.
4. **Dual clip:** made a no-op.
5. **Grad clip:** made a no-op. verl also skips a step whose gradient is non-finite.
6. **Reference model:** fp32 weights computed in bf16.
7. **Attention:** gradient checkpointing with torch SDPA on padded batches. flash_attn has no
   PyPI wheel, so there is no remove-padding.
8. **Process layout:** Ray plus a separate vLLM server, at vLLM share 0.35 against ours 0.15.
9. **Library versions.**
10. **Reward input:** the reward is computed on decoded token ids.
11. **"After" eval:** it runs in a separate process on the saved fp32 weights loaded as bf16.
12. **Sample order:** different.

Everything fits on one 4090, so no config had to be scaled down.

## Results

[comparison.json](comparison.json); verl metrics are in [verl-run/summary.json](verl-run/summary.json),
the step log in [verl-run/train.log](verl-run/train.log) and 1 s nvidia-smi samples in
[verl-run/gpu_mem.csv](verl-run/gpu_mem.csv).

| | grpo_run (eos-sync-k0-s17) | verl 0.9.1 |
|---|---|---|
| GSM8K pass@1 after (± SE, truncated) | 0.4475 ± 0.0297, 9 | 0.44 ± 0.0297, 13 |
| MATH-500 pass@1 after (± SE, truncated) | 0.295 ± 0.0388, 96 | 0.34 ± 0.0423, 121 |
| Paired change vs base, GSM8K (pp) | +2.12 ± 2.32 | +1.38 ± 2.39 |
| Paired change vs base, MATH-500 (pp) | +0.75 ± 3.05 | +5.25 ± 2.57 |
| Training reward, mean of first / last 10 steps | 0.4281 / 0.5828 | 0.3922 / 0.6266 |
| Sum of step times (s) | 578.2 | 1418.6 |
| Samples/s (6,400 / step time) | 11.07 | 4.51 |
| Wall time (s) | 698.1, including 113.2 s of eval | 1572, without eval |
| Peak GPU memory | 14,794,147,328 B torch allocator (one process) | 16,423 MiB nvidia-smi (all processes); learner 9.221 GB allocated |

Base: GSM8K 0.42625 ± 0.0281, MATH-500 0.2875 ± 0.0361.

- **Accuracy:**
  - On GSM8K the two frameworks agree: verl minus grpo_run is −0.75 ± 2.41 pp.
  - On MATH-500 verl is 4.5 ± 2.47 pp higher than grpo_run (1.8 SE, under 2 SE). This is
    one seed per framework on 100 problems, and the SE covers only eval sampling, not
    training seeds, so it is not evidence that verl trains better.
- **Speed:** verl is 2.45× slower per sample on this setup. Its step totals over 100 steps
  were:
  - generation, including the reward: 412.4 s, against grpo_run's rollout 182.7 s plus
    reward 115.1 s;
  - reference log-probs: 475.5 s;
  - actor update: 500.9 s, against grpo_run's learn 278.0 s;
  - weight sync: 19.6 s, against 1.9 s.

  The reference forward in fp32 under FSDP, padded SDPA attention without remove-padding,
  and a second process with sleep/wake account for most of it.
- **Memory:** the two numbers measure different things. grpo_run's is the torch allocator
  peak of one process holding both vLLM and the learner. verl's is the whole-GPU peak from
  nvidia-smi, with vLLM's 0.35 share reserved in a second process. Neither is close to the
  24 GB card.

## Failed attempts, kept as evidence

1. **[verl-attempt1-ray-deadlock](verl-attempt1-ray-deadlock/)** (00:28–00:41):
   `ray_init.num_cpus=8` deadlocked.
   - The TaskRunner, the transfer_queue controller and 7 of its 8 storage units held all
     8 slots, so the 8th unit and the GPU workers never started
     ([Ray state](verl-attempt1-ray-deadlock/ray-debug_state.txt)).
   - The GPU sat idle and the job was stopped. The fix was num_cpus=16 (logical slots; the
     thread count is still OMP_NUM_THREADS=4).
2. **[verl-attempt2-kernels-shadow](verl-attempt2-kernels-shadow/)** (82 s): FSDP model init
   failed.
   - verl asks for flash_attention_2, which is not installed. transformers then looked for
     the HF `kernels` package and found the project's own `kernels/` directory, which was
     on PYTHONPATH (`InvalidVersion: 'N/A'`).
   - The fix was `override_config.attn_implementation=sdpa` with remove-padding off.
3. **[eval-verl-rope10000-invalid](eval-verl-rope10000-invalid/)** (GSM8K 0.16, MATH-500
   0.0975; not a result): the first eval loaded verl's saved checkpoint directly.
   - verl saves config.json with transformers 5.9, which moves `rope_theta` into
     `rope_parameters`. `.venv-ml`'s transformers 4.55 then reads rope_theta as 10000
     instead of 1,000,000. The answers were fluent, but the arithmetic was wrong
     (5000 + 200 = 5020).
   - The weights themselves were fine: the median relative change from base was 0.14%.
   - The fix is `rl/verl/ckpt_view.py`: a view with the trained `model.safetensors` and the
     base model's config and tokenizer files (the transformers-5 tokenizer files do not load
     in 4.55). The reported eval used that view.

GPU time for T11 in phase 7 was about 0.75 h: attempt 1 0.22 h (holding the lock, GPU idle),
attempt 2 0.02 h, the run 0.44 h, and three evals of about 1 min each.
The 1.99 GB checkpoint is scratch at `.cache/t11/verl-ckpt` (not evidence).
