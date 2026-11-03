import hashlib

from scripts.env_identity import identity


def test_identity_lists_sorted_normalized_distributions(tmp_path):
    site = tmp_path / "site-packages"
    for name in ("vllm-0.10.2.dist-info", "Torch-2.8.0.dist-info", "pkg_x-1.0.egg-info", "torch"):
        (site / name).mkdir(parents=True)
    lines, digest = identity(site)
    assert lines == ["pkg-x==1.0", "torch==2.8.0", "vllm==0.10.2"]
    assert digest == hashlib.sha256("".join(f"{x}\n" for x in lines).encode()).hexdigest()


def test_identity_changes_with_a_version(tmp_path):
    site = tmp_path / "s"
    (site / "a-1.dist-info").mkdir(parents=True)
    first = identity(site)[1]
    (site / "a-1.dist-info").rename(site / "a-2.dist-info")
    assert identity(site)[1] != first
