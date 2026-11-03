"""Planning rules of the T15 feature-batch cycle (no GPU, no SpecForge)."""

import pytest

from spec.t15_cycle import latest_step, main, next_batch, should_start, stage_batch, train_overrides


def test_next_batch_follows_the_checkpoint_step():
    assert next_batch(None, batch_size=100, n_batches=5) == 0
    assert next_batch(300, batch_size=100, n_batches=5) == 3
    assert next_batch(500, batch_size=100, n_batches=5) is None


def test_next_batch_rejects_a_mid_batch_checkpoint():
    with pytest.raises(ValueError):
        next_batch(250, batch_size=100, n_batches=5)


def test_latest_step_reads_the_latest_link(tmp_path):
    assert latest_step(tmp_path, "run") is None
    (tmp_path / "run-step1200").mkdir()
    (tmp_path / "run-latest").symlink_to("run-step1200")
    assert latest_step(tmp_path, "run") == 1200


def test_should_start_keeps_the_job_under_its_budget():
    assert should_start(elapsed_s=0, per_batch_s=600, budget_s=3600)
    assert should_start(elapsed_s=2900, per_batch_s=600, budget_s=3600)
    assert not should_start(elapsed_s=3100, per_batch_s=600, budget_s=3600)


def test_train_overrides_resume_only_after_the_first_batch(tmp_path):
    assert train_overrides(0, 100, tmp_path, "run") == ["training.max_steps=100"]
    assert train_overrides(2, 100, tmp_path, "run") == [
        "training.max_steps=300",
        f"training.resume_from={tmp_path}/run-latest",
    ]


def test_stage_batch_uses_a_content_addressed_path(tmp_path):
    # prepare_hidden_states caches rows by the data path, so a rewritten batch file at the
    # same path silently reused the old 200 rows in the phase-7 smoke run.
    batch, base = tmp_path / "batch_000.jsonl", tmp_path / "sf-cache"
    batch.write_text('{"a": 1}\n')
    cache_dir, staged = stage_batch(base, batch)
    assert cache_dir.parent == base and cache_dir.name.startswith("batch-")
    assert staged.parent == cache_dir and staged.read_text() == '{"a": 1}\n'
    assert stage_batch(base, batch) == (cache_dir, staged)
    batch.write_text('{"a": 2}\n')
    other_dir, other = stage_batch(base, batch)
    assert other_dir != cache_dir and other.read_text() == '{"a": 2}\n'


def test_main_rejects_a_batch_size_the_capture_loader_would_truncate(tmp_path):
    # prepare_hidden_states drops the last partial capture batch: 200 rows -> 192 features.
    args = ["--config", "c", "--draft-config", "d", "--target", "t", "--specforge-src", "s",
            "--batches-dir", str(tmp_path), "--batch-size", "200", "--n-batches", "2",
            "--train-dir", str(tmp_path), "--run-id", "r", "--feature-bytes", "1",
            "--log", str(tmp_path / "l"), "--stdout-log", str(tmp_path / "o")]
    with pytest.raises(SystemExit, match="multiple of the capture batch size"):
        main(args)
