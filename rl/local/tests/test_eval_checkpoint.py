from rl.local import eval_checkpoint as ec


def test_defaults_match_the_t11_base_eval():
    a = ec.parse_args(["--model", "m", "--output", "o"])
    assert (a.eval_n, a.eval_max_tokens, a.seed, a.vllm_mem) == (4, 512, 17, 0.15)
    assert a.tokenizer == ec.BASE_MODEL
    assert str(a.data) == ".cache/t11/data.json"


def test_uses_grpo_run_eval():
    assert ec.gr.run_eval.__module__ == "rl.local.grpo_run"
