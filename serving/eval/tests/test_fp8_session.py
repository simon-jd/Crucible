from serving.eval import fp8_session as fs


def test_plan_is_ab_ba_ab_with_three_rounds():
    variants = [v for _, _, v in fs.PLAN]
    assert variants == ["baseline", "candidate", "candidate", "baseline", "baseline", "candidate"]
    assert [r for _, r, _ in fs.PLAN] == [1, 1, 2, 2, 3, 3]
    assert len({i for i, _, _ in fs.PLAN}) == 6


def test_quality_prompts_ask_for_parseable_answers():
    heldout = {
        "gsm8k": [{"id": "g1", "question": "Q?", "answer": "7"}],
        "mmlu_pro": [{"id": "m1", "question": "Pick", "options": ["x", "y", "z"], "answer": "B"}],
    }
    reqs = fs.quality_requests(heldout)
    assert [(r[0], r[1], r[2]) for r in reqs] == [("gsm8k", "g1", "7"), ("mmlu_pro", "m1", "B")]
    # First attempt's '#### <number>' template was echoed literally; show a concrete example.
    assert "#### 42" in reqs[0][3] and "<number>" not in reqs[0][3]
    assert reqs[0][5] is None
    assert reqs[1][3] == "Pick\n\nA. x\nB. y\nC. z\n\nAnswer with only the letter of the correct option."
    # MMLU-Pro output is constrained to exactly one valid option letter.
    assert reqs[1][5] == ["A", "B", "C"]


def test_chat_body_carries_guided_choice_only_when_given():
    assert fs.chat_body("m", "p", 8, ["A", "B"])["guided_choice"] == ["A", "B"]
    body = fs.chat_body("m", "p", 768, None)
    assert "guided_choice" not in body and body["chat_template_kwargs"] == {"enable_thinking": False}


def test_candidate_is_online_fp8_or_a_prequantized_checkpoint():
    from types import SimpleNamespace

    args = SimpleNamespace(snapshot="/bf16", model_name="Qwen/Qwen3-8B", port=1, max_model_len=8192,
                           max_num_seqs=64, candidate_snapshot=None)  # fmt: skip
    bf16 = fs.server_cmd(args, "bf16")
    online = fs.server_cmd(args, "fp8")
    assert online == bf16 + ["--quantization", "fp8"] and "--enforce-eager" in bf16
    args.candidate_snapshot = "/fp8-ckpt"
    ckpt = fs.server_cmd(args, "fp8")
    # Same flags; only the weights differ (vLLM reads the checkpoint's quantization_config).
    assert ckpt == [("/fp8-ckpt" if x == "/bf16" else x) for x in bf16]
    assert fs.server_cmd(args, "bf16") == bf16
