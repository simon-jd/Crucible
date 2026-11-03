"""verl reward manager for T11: NaiveRewardManager plus the truncation flag.

Loaded by verl (reward.reward_manager.source=importlib,
module.path=pkg://rl.verl.reward_manager, name=T11RewardManager); needs verl,
so only .venv-verl imports it. run_single mirrors verl 0.9.1's
NaiveRewardManager.run_single, with two changes:
  * extra_info is copied before it is written to (the naive manager mutates the
    dict it gets from non_tensor_batch, which can be shared by the n repeats);
  * extra_info["truncated"] is set from the response token ids (see
    rl.verl.reward.is_truncated), because finish_reason never reaches the reward.
"""

from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager

from rl.verl.reward import is_truncated


class T11RewardManager(NaiveRewardManager):
    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score, reward_router_address, reward_model_tokenizer)
        self.max_response_tokens = int(config.actor_rollout_ref.rollout.response_length)
        # grpo_run's vLLM uses generation_config="vllm": only the tokenizer's EOS
        # (<|im_end|> for Qwen2.5-Instruct) stops generation.
        self.eos_ids = {tokenizer.eos_token_id}

    async def run_single(self, data):
        data = data[-1:]
        item = data[0]
        response_ids = item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_length = int(item.batch["attention_mask"][-response_length:].sum())
        valid_ids = response_ids[:valid_length].tolist()

        extra_info = dict(item.non_tensor_batch.get("extra_info", None) or {})
        extra_info["truncated"] = is_truncated(valid_ids, self.max_response_tokens, self.eos_ids)
        extra_info["response_tokens"] = valid_length
        data_source = item.non_tensor_batch["data_source"]
        ground_truth = item.non_tensor_batch["reward_model"]["ground_truth"]

        text = await self.loop.run_in_executor(None, lambda: self.tokenizer.decode(valid_ids, skip_special_tokens=True))
        result = await self.loop.run_in_executor(
            None,
            lambda: self.compute_score(
                data_source=data_source, solution_str=text, ground_truth=ground_truth, extra_info=extra_info
            ),
        )
        return {"reward_score": result["score"], "reward_extra_info": dict(result)}
