"""CPU-only regressions for the report literal/citation contract."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.check_report_numbers import check_reports

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/check_report_numbers.py"
CITE = "[evidence: docs/evidence/source.md#results]"


@pytest.fixture
def project(tmp_path):
    evidence = tmp_path / "docs/evidence"
    evidence.mkdir(parents=True)
    (evidence / "source.md").write_text(
        "# Source\n## Results\n"
        "42 -0.25 .375 1e-5 66.67% 1,234 FP32 BF16 Qwen2.5-0.5B-Instruct "
        "seed123 2.8.0 cu128 abc123\n### Nested\n77\n## Other\n999\n",
        encoding="utf-8",
    )
    return tmp_path


def report(project, text):
    path = project / "draft.md"
    path.write_text(text, encoding="utf-8")
    return path


def errors(project, text):
    return check_reports([report(project, text)], project_root=project).errors


def cli(project, *args):
    return subprocess.run(
        [sys.executable, "-B", str(SCRIPT), "--project-root", str(project), *args],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )


def test_valid_citation_and_nested_section(project):
    result = check_reports([report(project, f"值 42 和 77。 {CITE}\n")], project_root=project)
    assert result.errors == []
    assert result.claims == 2


def test_uncited_draft_fails_with_report_line(project):
    found = errors(project, "# Draft\n测量 42。\n")
    assert any("draft.md:2:" in item and "uncited" in item for item in found)


@pytest.mark.parametrize("anchor", ["absent", ""])
def test_wrong_or_missing_anchor(project, anchor):
    assert errors(project, f"42 [evidence: docs/evidence/source.md#{anchor}]\n")


def test_value_must_be_in_referenced_section(project):
    assert any("999" in item and "unsupported" in item for item in errors(project, f"999 {CITE}"))


def test_multiple_claims_tables_sign_decimal_scientific_and_identifiers(project):
    text = (
        f"| 42 | -0.25 | .375 | 1e-5 | 66.67% | 1,234 | {CITE}\n"
        f"FP32 BF16 Qwen2.5-0.5B-Instruct seed123 2.8.0+cu128 abc123 {CITE}\n"
        f'```json\n{{"seed": 42, "lr": 1e-5}} {CITE}\n```\n'
    )
    assert errors(project, text) == []
    assert errors(project, text.replace("1e-5", "1e-6", 1))


def test_same_line_required_and_every_claim_covered(project):
    assert errors(project, f"42\n{CITE}\n")
    assert errors(project, f"42 和 999 {CITE}\n")


def test_filters_keep_visible_link_text_and_code_values(project):
    assert errors(project, f"1. [42](https://example.invalid/999) {CITE}\n") == []
    assert errors(project, f"[999](https://example.invalid/42) {CITE}\n")
    assert errors(project, f"```\n999\n```\n42 {CITE}")
    assert errors(project, f"`[42](999)` {CITE}")


def test_evidence_markup_cannot_supply_values(project):
    (project / "docs/evidence/source.md").write_text(
        "## Results\n1. [plain](https://example.invalid/999)\n"
        "[evidence: docs/evidence/missing42.md#value123]\n```python888\n```\n",
        encoding="utf-8",
    )
    for value in ["1", "999", "missing42.md", "value123", "python888"]:
        assert errors(project, f"{value} {CITE}")


def test_dangling_or_malformed_citation_fails_even_without_claims(project):
    for citation in [
        "[evidence: docs/evidence/missing.md#results]",
        "[evidence: docs/evidence/source.md]",
        "[evidence:docs/evidence/source.md#results]",
    ]:
        assert errors(project, f"42 {CITE}\nplain {citation}")


@pytest.mark.parametrize("path", ["../outside.md", "docs/evidence/../outside.md", "/tmp/x.md"])
def test_path_escape_rejected(project, path):
    assert errors(project, f"42 [evidence: {path}#results]")


def test_symlink_escape_and_evidence_root_symlink_rejected(project, tmp_path):
    outside = project / "outside.md"
    outside.write_text("## Results\n42\n", encoding="utf-8")
    (project / "docs/evidence/escape.md").symlink_to(outside)
    assert errors(project, "42 [evidence: docs/evidence/escape.md#results]")
    alternative = project / "elsewhere"
    alternative.mkdir()
    (alternative / "source.md").write_text("## Results\n42\n", encoding="utf-8")
    (project / "docs/evidence/source.md").unlink()
    (project / "docs/evidence/escape.md").unlink()
    (project / "docs/evidence").rmdir()
    (project / "docs/evidence").symlink_to(alternative, target_is_directory=True)
    assert errors(project, f"42 {CITE}")


def test_duplicate_anchor_and_heading_in_fence(project):
    source = project / "docs/evidence/source.md"
    source.write_text("## Results\n42\n## Results\n42\n", encoding="utf-8")
    assert errors(project, f"42 {CITE}")
    source.write_text("## Results\n```\n## Other\n42\n```\n", encoding="utf-8")
    assert errors(project, f"42 {CITE}") == []


@pytest.mark.parametrize("glyph", ["４２", "٤٢", "²", "Ⅳ"])
def test_unicode_numeric_glyphs_fail_clearly(project, glyph):
    assert any("Unicode numeric" in item for item in errors(project, f"{glyph} 42 {CITE}"))


def test_missing_empty_and_each_zero_claim_input_fail(project):
    good = report(project, f"42 {CITE}")
    for text in ["", "plain factual prose"]:
        empty = project / "empty.md"
        empty.write_text(text, encoding="utf-8")
        assert check_reports([good, empty], project_root=project).errors
    assert check_reports([], project_root=project).errors
    assert check_reports([project / "absent.md"], project_root=project).errors


def test_cli_missing_input_and_uncited_failure_then_corrected_pass(project):
    assert cli(project).returncode != 0
    assert cli(project, "absent.md").returncode != 0
    path = report(project, "42\n")
    failed = cli(project, str(path))
    assert failed.returncode != 0 and "draft.md:1:" in failed.stderr
    report(project, f"42 {CITE}\n")
    passed = cli(project, str(path))
    assert passed.returncode == 0 and "PASS" in passed.stdout


def test_corrected_real_drafts_pass():
    paths = [
        ROOT / name
        for name in [
            "reports/w2-async-grpo.md",
            "reports/w3-disaggregated-serving.md",
            "reports/repro/README.md",
        ]
    ]
    result = check_reports(paths, project_root=ROOT)
    assert result.errors == []
    assert result.claims > 0


def test_grouped_unit_literal_is_indivisible(project):
    source = project / "docs/evidence/source.md"
    source.write_text("## Results\n7,409,397,760B\n999B\n", encoding="utf-8")
    assert errors(project, f"7,409,397,760B {CITE}") == []
    assert errors(project, f"7,409,397,999B {CITE}")
    source.write_text("## Results\n-1,234.50e-3MB\n", encoding="utf-8")
    assert errors(project, f"-1,234.50e-3MB {CITE}") == []
    assert errors(project, f"1,234.50e-3MB {CITE}")


def test_multibacktick_code_cannot_hide_linklike_numbers(project):
    assert errors(project, f"``literal` [42](999)`` {CITE}")


def test_signed_unit_literal_cannot_lose_sign(project):
    (project / "docs/evidence/source.md").write_text("## Results\n-42MB\n", encoding="utf-8")
    assert errors(project, f"-42MB {CITE}") == []
    assert errors(project, f"42MB {CITE}")


@pytest.mark.parametrize(
    "literal,changed",
    [
        ("-0.25", "0.25"),
        ("+0.25", "0.25"),
        ("-42MB", "42MB"),
        ("66.67%", "66.67"),
        ("7,409,397,760B", "7,409,397,999B"),
        ("-1,234.50e-3MB", "1,234.50e-3MB"),
    ],
)
@pytest.mark.parametrize(
    "source_period,report_period", [(False, True), (True, False), (True, True)]
)
def test_sentence_period_preserves_whole_exact_literal(
    project, literal, changed, source_period, report_period
):
    from scripts.check_report_numbers import literals

    source = project / "docs/evidence/source.md"
    source.write_text(
        f"## Results\n{literal}{'.' if source_period else ''}\n999B\n", encoding="utf-8"
    )
    assert literals(literal + ".") == {literal}
    assert errors(project, f"{literal}{'.' if report_period else ''} {CITE}") == []
    assert errors(project, f"{changed}{'.' if report_period else ''} {CITE}")


def test_sentence_period_does_not_shorten_identifiers_or_versions(project):
    from scripts.check_report_numbers import literals

    values = ["2.8.0", "Qwen2.5-0.5B-Instruct", "seed123", "abc123", "LR1e-5", "FP32"]
    assert literals(" ".join(value + "." for value in values)) == set(values)
    assert errors(project, f"2.8. {CITE}")
    assert errors(project, f"32 {CITE}")


@pytest.mark.parametrize("size", [3, 4, 6])
@pytest.mark.parametrize("body", ["999", "literal` [42](999)", "literal`` [42](999)"])
def test_inline_long_backtick_runs_keep_numeric_code_content(project, size, body):
    inline = "`" * size + body + "`" * size
    assert errors(project, inline + f"\n42 {CITE}")
    assert errors(project, inline + f" {CITE}")
    source = project / "docs/evidence/source.md"
    source.write_text(f"## Results\n{inline}\n", encoding="utf-8")
    assert errors(project, f"999 {CITE}") == []
    assert errors(project, inline + f" {CITE}") == []


@pytest.mark.parametrize("fence", ["```", "````", "~~~~"])
def test_real_fence_delimiters_and_info_are_excluded_but_body_is_checked(project, fence):
    block = f"{fence}python999\n42 {CITE}\n{fence}\n"
    assert errors(project, block) == []
    assert errors(project, block.replace(f"42 {CITE}", "999"))
    source = project / "docs/evidence/source.md"
    source.write_text(f"## Results\n{fence}python999\n42\n{fence}\n", encoding="utf-8")
    assert errors(project, f"42 {CITE}") == []
    assert errors(project, f"python999 {CITE}")
