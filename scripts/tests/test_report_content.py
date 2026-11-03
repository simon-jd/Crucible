"""The two technical reports must carry the real single-GPU results, not CPU-phase drafts."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
W2 = ROOT / "reports/w2-async-grpo.md"
W3 = ROOT / "reports/w3-disaggregated-serving.md"
REPRO = ROOT / "reports/repro/README.md"


def headings(text):
    return [m[1].strip() for m in re.finditer(r"^#{2,3} (.+)$", text, re.MULTILINE)]


def test_each_report_has_method_results_limits_and_repro_sections():
    for path in (W2, W3):
        names = " ".join(headings(path.read_text()))
        for section in ("方法", "结果", "局限", "复现"):
            assert section in names, (path.name, section)


def test_w2_cites_the_real_grpo_runs_and_reward_hack():
    text = W2.read_text()
    for anchor in (
        "docs/evidence/rl-grpo.md#results",
        "docs/evidence/rl-grpo.md#reward-hacking-caught",
        "docs/evidence/rl-grpo.md#setup",
    ):
        assert f"[evidence: {anchor}]" in text
    for literal in ("0.426", "0.444", "703/800", "6,400"):
        assert literal in text


def test_w3_cites_the_real_baseline_and_fp8_runs():
    text = W3.read_text()
    for anchor in (
        "docs/evidence/serving-baseline.md#results",
        "docs/evidence/serving-baseline.md#setup",
        "docs/evidence/pd-disaggregation.md#results",
    ):
        assert f"[evidence: {anchor}]" in text
    for literal in ("1.089", "2.041", "0.104", "24.6", "15.5", "0.920", "0.910"):
        assert literal in text
    assert "未在真实多机上验证" in text


def test_repro_appendix_pins_commits_env_hash_and_seeds():
    text = REPRO.read_text()
    for commit in (
        "bc749482dfc1a4a92a01c79cdf20165784cb6349",
        "b49326bee0f7e0afd17f46e84995e1bf50d4dba4",
        "4f834c7babd964bfe6588276616549afd3635219",
    ):
        assert commit in text
    assert re.search(r"\b[0-9a-f]{64}\b", text), "environment hash missing"
    assert "seed" in text.lower()
    for command in ("run_grpo_gpu.sh", "run_serving_baseline_gpu.sh", "run_fp8_compare_gpu.sh"):
        assert command in text
