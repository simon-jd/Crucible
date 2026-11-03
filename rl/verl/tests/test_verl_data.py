from rl.local import grpo_run as gr
from rl.verl import prepare_verl_data as pvd


def fake_data(n_train=20):
    train = [{"id": f"t{i}", "question": f"q{i}", "answer": str(i), "source": "openai/gsm8k:main/train"} for i in range(n_train)]
    evals = [{"id": f"e{i}", "question": f"eq{i}", "answer": str(i), "source": "x"} for i in range(12)]
    return {"train": train, "gsm8k_eval": evals, "math500_eval": [], "sha256": "x"}


def grpo_run_order(train, steps, per_step):
    """grpo_run's cursor: rows = [train[(cursor + i) % len(train)] ...]; cursor += per_step."""
    cursor, out = 0, []
    for _ in range(steps):
        out.extend(train[(cursor + i) % len(train)] for i in range(per_step))
        cursor += per_step
    return out


def test_train_order_matches_grpo_run_cursor_including_wraparound():
    data = fake_data(20)
    rows = pvd.train_rows(data, steps=5, per_step=8)  # 40 rows over 20 -> wraps twice
    assert [r["extra_info"]["id"] for r in rows] == [r["id"] for r in grpo_run_order(data["train"], 5, 8)]
    assert [r["extra_info"]["index"] for r in rows] == list(range(40))


def test_t11_size_is_one_epoch_of_100_steps():
    data = fake_data(1024)
    rows = pvd.train_rows(data)
    assert len(rows) == 800 and len(rows) // 8 == 100
    assert [r["extra_info"]["id"] for r in rows] == [f"t{i}" for i in range(800)]


def test_row_format_uses_grpo_run_prompt_and_reference_answer():
    data = fake_data(3)
    row = pvd.train_rows(data, steps=1, per_step=1)[0]
    assert row["prompt"] == gr.prompt_messages("q0")
    assert row["reward_model"] == {"style": "rule", "ground_truth": "0"}
    assert row["data_source"] == "openai/gsm8k:main/train" and row["ability"] == "math"
    assert row["extra_info"] == {"id": "t0", "index": 0, "split": "train"}


def test_val_rows_are_a_small_gsm8k_eval_slice():
    rows = pvd.val_rows(fake_data(), count=8)
    assert len(rows) == 8 and rows[0]["extra_info"]["split"] == "val"
    assert rows[0]["prompt"] == gr.prompt_messages("eq0")


def test_main_rejects_other_data(tmp_path):
    import json

    import pytest

    path = tmp_path / "data.json"
    path.write_text(json.dumps(fake_data()))
    with pytest.raises(SystemExit):
        pvd.main(["--data", str(path), "--out", str(tmp_path / "out")])
