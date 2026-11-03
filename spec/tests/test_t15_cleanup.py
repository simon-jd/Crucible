"""The T15 feature-batch deleter only removes feature directories under .cache/t15/."""

import pytest

from spec.t15_cleanup import OWN_DIRS, delete_feature_batch, delete_own_dir


def _tree(tmp_path):
    root = tmp_path / ".cache" / "t15"
    feats = root / "features-full" / "rows_0-2000"
    feats.mkdir(parents=True)
    (feats / "data_0.ckpt").write_bytes(b"x" * 10)
    (feats / "data_1.ckpt").write_bytes(b"y" * 5)
    return root


def test_deletes_a_feature_directory_and_reports_bytes(tmp_path):
    root = _tree(tmp_path)
    freed = delete_feature_batch(root / "features-full", root=root)
    assert freed == {"files": 2, "bytes": 15}
    assert not (root / "features-full").exists()


def test_refuses_paths_outside_the_root(tmp_path):
    root = _tree(tmp_path)
    outside = tmp_path / "features-elsewhere"
    outside.mkdir()
    with pytest.raises(PermissionError):
        delete_feature_batch(outside, root=root)
    with pytest.raises(PermissionError):
        delete_feature_batch(root / "features-full" / ".." / ".." / "..", root=root)
    assert outside.exists()


def test_refuses_the_root_and_non_feature_directories(tmp_path):
    root = _tree(tmp_path)
    (root / "train").mkdir()
    with pytest.raises(PermissionError):
        delete_feature_batch(root, root=root)
    with pytest.raises(PermissionError):
        delete_feature_batch(root / "train", root=root)
    assert (root / "train").exists()


def test_refuses_a_symlink_that_escapes(tmp_path):
    root = _tree(tmp_path)
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("keep")
    (root / "features-link").symlink_to(victim)
    with pytest.raises(PermissionError):
        delete_feature_batch(root / "features-link", root=root)
    assert (victim / "keep.txt").exists()


def test_refuses_symlinks_inside_the_tree(tmp_path):
    root = _tree(tmp_path)
    victim = tmp_path / "victim.txt"
    victim.write_text("keep")
    (root / "features-full" / "rows_0-2000" / "link").symlink_to(victim)
    with pytest.raises(PermissionError):
        delete_feature_batch(root / "features-full", root=root)
    assert victim.exists() and (root / "features-full" / "rows_0-2000" / "data_0.ckpt").exists()


def _smoke(root):
    train = root / "smoke" / "train"
    step = train / "t15-smoke-step416"
    step.mkdir(parents=True)
    (step / "model.safetensors").write_bytes(b"z" * 7)
    (train / "t15-smoke-latest").symlink_to("t15-smoke-step416")
    return train


def test_deletes_the_smoke_checkpoints_with_their_inner_latest_link(tmp_path):
    root = _tree(tmp_path)
    train = _smoke(root)
    assert delete_own_dir("smoke/train", root=root) == {"files": 1, "links": 1, "bytes": 7}
    assert not train.exists() and (root / "smoke").exists()
    assert (root / "features-full").exists()


def test_smoke_deleter_refuses_a_link_that_leaves_the_run(tmp_path):
    root = _tree(tmp_path)
    train = _smoke(root)
    victim = tmp_path / "victim"
    victim.mkdir()
    (train / "escape").symlink_to(victim)
    with pytest.raises(PermissionError):
        delete_own_dir("smoke/train", root=root)
    assert victim.exists() and (train / "t15-smoke-step416" / "model.safetensors").exists()


def test_smoke_deleter_refuses_when_the_run_dir_is_a_symlink(tmp_path):
    root = _tree(tmp_path)
    victim = tmp_path / "victim"
    (victim / "x").mkdir(parents=True)
    (root / "smoke").mkdir()
    (root / "smoke" / "train").symlink_to(victim)
    with pytest.raises(PermissionError):
        delete_own_dir("smoke/train", root=root)
    assert (victim / "x").exists()


def test_own_dir_deleter_only_takes_the_listed_scratch_dirs(tmp_path):
    root = _tree(tmp_path)
    assert OWN_DIRS == ("smoke/train", "train-full", "sf-cache")
    (root / "full").mkdir()
    for name in ("full", "smoke", ".", "../t15/full", "features-full"):
        with pytest.raises(PermissionError):
            delete_own_dir(name, root=root)
    assert (root / "full").exists() and (root / "features-full").exists()
