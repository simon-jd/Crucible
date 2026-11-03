from pathlib import Path

import pytest

from scaling.checkpoint_retention import CheckpointRetention


def save(root, step):
    guard = CheckpointRetention(root, "run")
    guard.begin(step)
    try:
        path = Path(root) / f"step-{step}"
        path.mkdir()
        (path / ".metadata").write_text("CPU-native-DCP-fixture")
        assert len(list(Path(root).glob("step-*"))) <= 2
        guard.confirm(step, f"checkpoint-{step}")
    finally:
        guard.end()


def test_pre_save_slot_preserves_latest_confirmed_and_offinterval_final_never_exceeds_two(tmp_path):
    root = tmp_path / "checkpoints"
    save(root, 2)
    save(root, 4)
    save(root, 5)
    assert sorted(p.name for p in root.glob("step-*")) == ["step-4", "step-5"]
    guard = CheckpointRetention(root, "run")
    guard.begin(6)
    guard.end()  # Simulated failed save: latest checkpoint 5 remains intact.
    assert [p.name for p in root.glob("step-*")] == ["step-5"]
    assert (root / "step-5/.metadata").is_file()


def test_unknown_incomplete_or_foreign_paths_are_never_deleted(tmp_path):
    root = tmp_path / "checkpoints"
    save(root, 2)
    save(root, 4)
    unknown = root / "step-99"
    unknown.mkdir()
    (unknown / "partial-data").write_text("must retain evidence")
    guard = CheckpointRetention(root, "run")
    with pytest.raises(ValueError, match="incomplete"):
        guard.begin(5)
    assert (root / "step-2/.metadata").exists()
    assert (root / "step-4/.metadata").exists()
    assert (unknown / "partial-data").read_text() == "must retain evidence"
    with pytest.raises(ValueError, match="ownership"):
        CheckpointRetention(root, "foreign").begin(5)


def test_symlink_or_duplicate_latest_checkpoint_is_refused_without_deletion(tmp_path):
    root = tmp_path / "checkpoints"
    save(root, 2)
    with pytest.raises(ValueError, match="latest confirmed"):
        CheckpointRetention(root, "run").begin(2)
    link = tmp_path / "link"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="nonsymlink"):
        CheckpointRetention(link, "run").begin(4)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "step-99").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="unknown"):
        CheckpointRetention(root, "run").begin(4)
    assert outside.exists()
    assert (root / "step-2/.metadata").exists()


def test_explicit_anchor_keeps_restored_older_checkpoint_and_retires_superseded_target(tmp_path):
    root = tmp_path / "checkpoints"
    save(root, 2)
    save(root, 4)
    anchor_before = (root / "step-2/.crucible-confirmed.json").read_text()
    guard = CheckpointRetention(root, "run")
    guard.begin(4, protected_checkpoint_id="checkpoint-2")
    try:
        assert (root / "step-2/.crucible-confirmed.json").read_text() == anchor_before
        assert not (root / "step-4").exists()
        assert len(list(root.glob("step-*"))) == 1
        (root / "step-4").mkdir()
        (root / "step-4/.metadata").write_text("new generation CPU DCP")
        assert len(list(root.glob("step-*"))) == 2
        guard.confirm(4, "checkpoint-4-replayed")
    finally:
        guard.end()
    assert (root / "step-2/.crucible-confirmed.json").read_text() == anchor_before
    assert "checkpoint-4-replayed" in (root / "step-4/.crucible-confirmed.json").read_text()


def test_active_anchor_cannot_be_overwritten_or_replaced_by_unknown_identity(tmp_path):
    root = tmp_path / "checkpoints"
    save(root, 2)
    save(root, 4)
    for step, anchor in [(2, "checkpoint-2"), (4, "unconfirmed-generation")]:
        with pytest.raises(ValueError, match="active recovery anchor"):
            CheckpointRetention(root, "run").begin(step, protected_checkpoint_id=anchor)
        assert sorted(p.name for p in root.glob("step-*")) == ["step-2", "step-4"]
