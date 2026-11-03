#!/usr/bin/env bash
# T11 base config (docs/evidence/rl-grpo-p4/eos-sync-k0-s17) reproduced with verl 0.9.1.
# usage: bash scripts/with_gpu.sh -- bash scripts/run_t11_verl.sh OUT_DIR
# (holds no flock itself: the caller's with_gpu.sh does; never nest it here)
# Then the held-out eval, also under the flock, in .venv-ml, on a view with the trained
# weights and the base model's config and tokenizer (rl/verl/ckpt_view.py says why):
#   .venv/bin/python -m rl.verl.ckpt_view .cache/t11/verl-ckpt/global_step_100/actor/huggingface \
#     MODEL_SNAPSHOT .cache/t11/verl-eval-view
#   bash scripts/with_gpu.sh -- env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 .venv-ml/bin/python \
#     -m rl.local.eval_checkpoint --model .cache/t11/verl-eval-view --output OUT_DIR/eval
#
# Matched to grpo_run: model, data and its order (rl/verl/prepare_verl_data.py), prompt
# and chat template, 100 steps x 8 prompts x 8 samples, temp 1 / top_p 1 / top_k -1,
# 384 response tokens, vLLM max_model_len 1024, max_num_seqs 128, eager, prefix cache,
# raw logprobs, generation_config=vllm, reward = verify_math with the eos penalty
# (rl/verl/reward.py), GRPO advantage, one AdamW step per batch (lr 3e-6, wd 0.01,
# betas .9/.999, constant lr, no warmup), clip 0.2, token-mean over the whole batch,
# k3 KL loss coef 0.001 against the initial policy, old logprobs = rollout logprobs
# (bypass_mode), FSDP2 with fp32 masters and bf16 compute, microbatch 2, seed 17.
# Remaining differences from rl.local.grpo_run:
#  1. Rollout seeds: grpo_run seeds every request ((seed*1000003 + gen*1009 + i) % 2^31);
#     verl seeds only the engine (17 + replica rank), so the sampled completions differ.
#  2. GRPO std: verl uses torch.std (Bessel, /(n-1)); grpo_run std(correction=0), so
#     verl's advantages are sqrt(7/8) = 0.935x grpo_run's (both + 1e-6).
#  3. Clamps: verl clamps the log-ratio to +-20 and the k3 KL to +-10; grpo_run does not.
#  4. Dual clip: verl's vanilla loss always dual-clips; clip_ratio_c=10000 makes it a no-op.
#  5. Grad clip: verl requires clip_grad; 1e9 makes it a no-op. verl also skips a step
#     whose grad norm is non-finite (grpo_run would apply it).
#  6. Reference policy: verl's ref worker holds fp32 weights computed in bf16;
#     grpo_run's ref is a bf16 copy. Small logprob differences in the KL term.
#  7. Learner kernels: verl uses gradient checkpointing and torch SDPA on padded batches
#     (flash_attn has no PyPI wheel, so no remove-padding); torch.compile is off here.
#     Same math, different rounding.
#  8. Process layout: Ray + a separate vLLM server process (mp executor, sleep/wake
#     between phases) and weight sync through verl's checkpoint engine; grpo_run is
#     one process with in-memory load_weights. Wall time and peak memory are not
#     comparable one to one (two processes, different vLLM share: 0.35 vs 0.15).
#  9. Versions: .venv-verl (torch 2.11, vllm 0.24, transformers 5.9) vs .venv-ml.
# 10. Reward text: verl decodes the response ids (skip_special_tokens); grpo_run grades
#     vLLM's incremental c.text. Truncation is rebuilt from token ids (rl/verl/reward.py).
# 11. "After" eval: a separate process loads the saved fp32 HF weights as bf16;
#     grpo_run evaluates the in-memory weights in the same vLLM engine.
# 12. verl orders samples by its own batching (balance_batch off here); the update is a
#     token-mean over the whole batch, so the order does not change the gradient.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
root="$PWD"
out="${1:?output directory}"
mkdir -p "$out"
out="$(cd "$out" && pwd)"
model=/home/postedism/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
py=.venv-verl/bin/python
# The checkpoint (an fp32 HF model) is scratch, not evidence: it stays under .cache.
ckpt="$root/.cache/t11/verl-ckpt"

export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
export PYTHONPATH="$root"
# Ray's socket paths ($RAY_TMPDIR/session_<date>_<pid>/sockets/plasma_store) must stay under
# the 107-byte AF_UNIX limit, so its temp dir is short.
export TMPDIR="$root/.cache/t11/tmp" RAY_TMPDIR="$root/.cache/r"
export HF_DATASETS_CACHE="$root/.cache/t11/hf-datasets"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 RAY_USAGE_STATS_ENABLED=0
export TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
mkdir -p "$TMPDIR" "$RAY_TMPDIR" "$HF_DATASETS_CACHE"

# verl 0.9.1's main_ppo imports transfer_queue (a git dependency in its pyproject, not in
# setup.py's install_requires): fail before touching the GPU if it is missing.
"$py" -c 'import verl, transfer_queue, vllm, torch; print("verl", verl.__version__, "vllm", vllm.__version__, "torch", torch.__version__)'
"$py" -m rl.verl.prepare_verl_data --data .cache/t11/data.json --out .cache/t11/verl-data

{ echo "nproc $(nproc)"; uptime; } > "$out/host-before.txt"
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -lms 1000 > "$out/gpu_mem.csv" &
smi_pid=$!
trap 'kill "$smi_pid" 2>/dev/null || true' EXIT

overrides=(
  "hydra.run.dir=$out/hydra"
  "data.train_files=$root/.cache/t11/verl-data/train.parquet"
  "data.val_files=$root/.cache/t11/verl-data/val.parquet"
  data.train_batch_size=8 data.max_prompt_length=512 data.max_response_length=384
  data.shuffle=false data.seed=17 data.dataloader_num_workers=0 data.filter_overlong_prompts=false
  algorithm.adv_estimator=grpo algorithm.norm_adv_by_std_in_grpo=true algorithm.use_kl_in_reward=false
  algorithm.rollout_correction.bypass_mode=true
  "actor_rollout_ref.model.path=$model"
  actor_rollout_ref.model.enable_gradient_checkpointing=true actor_rollout_ref.model.use_remove_padding=false
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa
  actor_rollout_ref.actor.strategy=fsdp2 actor_rollout_ref.ref.strategy=fsdp2
  actor_rollout_ref.actor.optim.lr=3e-6 actor_rollout_ref.actor.optim.weight_decay=0.01
  actor_rollout_ref.actor.optim.lr_scheduler_type=constant
  actor_rollout_ref.actor.optim.clip_grad=1e9
  actor_rollout_ref.actor.ppo_mini_batch_size=8 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
  actor_rollout_ref.actor.use_dynamic_bsz=false actor_rollout_ref.actor.ppo_epochs=1 actor_rollout_ref.actor.shuffle=false
  actor_rollout_ref.actor.clip_ratio=0.2 actor_rollout_ref.actor.clip_ratio_low=0.2 actor_rollout_ref.actor.clip_ratio_high=0.2
  actor_rollout_ref.actor.clip_ratio_c=10000.0 actor_rollout_ref.actor.policy_loss.loss_mode=vanilla
  actor_rollout_ref.actor.loss_agg_mode=token-mean actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.use_torch_compile=false actor_rollout_ref.actor.fsdp_config.use_torch_compile=false
  actor_rollout_ref.actor.fsdp_config.seed=17
  "actor_rollout_ref.actor.checkpoint.save_contents=[hf_model]"
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n=8 actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 actor_rollout_ref.rollout.max_model_len=1024
  actor_rollout_ref.rollout.max_num_seqs=128 actor_rollout_ref.rollout.enforce_eager=true
  actor_rollout_ref.rollout.enable_prefix_caching=true
  actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 actor_rollout_ref.rollout.top_k=-1
  actor_rollout_ref.rollout.seed=17 actor_rollout_ref.rollout.full_determinism=false
  actor_rollout_ref.rollout.logprobs_mode=raw_logprobs actor_rollout_ref.rollout.calculate_log_probs=true
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 actor_rollout_ref.rollout.agent.num_workers=1
  +actor_rollout_ref.rollout.engine_kwargs.vllm.generation_config=vllm
  reward.num_workers=1
  reward.custom_reward_function.path=pkg://rl.verl.reward reward.custom_reward_function.name=compute_score
  reward.reward_manager.source=importlib reward.reward_manager.name=T11RewardManager
  reward.reward_manager.module.path=pkg://rl.verl.reward_manager
  trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.total_epochs=1 trainer.total_training_steps=100
  "trainer.logger=[console]" trainer.val_before_train=false trainer.test_freq=-1 trainer.log_val_generations=0
  trainer.save_freq=100 "trainer.default_local_dir=$ckpt" trainer.resume_mode=disable trainer.balance_batch=false
  trainer.project_name=t11 trainer.experiment_name=verl-eos-sync-k0-s17
  "+ray_kwargs.ray_init._temp_dir=$RAY_TMPDIR" ray_kwargs.ray_init.num_cpus=16
)
# num_cpus is Ray's logical slot count, not threads (OMP_NUM_THREADS=4). 8 deadlocked on
# 2026-11-02: TaskRunner, the transfer_queue controller and its 8 storage units take 10.
printf '%s\n' "${overrides[@]}" > "$out/overrides.txt"

start=$(date +%s)
set +e
"$py" -m verl.trainer.main_ppo "${overrides[@]}" 2>&1 | tee "$out/train.log"
status=${PIPESTATUS[0]}
set -e
end=$(date +%s)
kill "$smi_pid" 2>/dev/null || true
{ echo "nproc $(nproc)"; uptime; } > "$out/host-after.txt"
printf 'start %s\nend %s\nwall_s %s\nstatus %s\n' "$start" "$end" "$((end - start))" "$status" > "$out/timing.txt"
cat "$out/timing.txt"
test -f "$ckpt/global_step_100/actor/huggingface/config.json"
exit "$status"
