"""The HTML that headless Chrome prints to text PDFs, and the PDF-vs-Markdown number check.

Runs in test-ml: markdown-it lives in .venv-ml only."""

import pytest

pytest.importorskip("markdown_it")

from scripts.build_report_html import build_html, extract_citations
from scripts.check_pdf_text import compare_literals, report_literals

SAMPLE = """# 标题 Title

第一段，含数字 0.426→0.444。[evidence: docs/evidence/rl-grpo.md#results]

| 列 | 值 |
|---|---|
| GSM8K | 0.920 [evidence: docs/evidence/pd-disaggregation.md#results] |

- 列表项 `code 1`
- **加粗** 和 [链接](https://example.com/v9)
- <script>alert(1)</script>
"""


def test_citations_become_numbered_notes_reusing_numbers():
    text, notes = extract_citations("a [evidence: x.md#a] b [evidence: y.md#b] c [evidence: x.md#a]")
    assert text == "a [1] b [2] c [1]"
    assert notes == ["x.md#a", "y.md#b"]


def test_html_has_table_notes_cjk_font_and_no_raw_html():
    html = build_html(SAMPLE, title="t")
    assert html.startswith("<!DOCTYPE html>") and '<meta charset="utf-8">' in html
    assert "<table>" in html and "<td>0.920 [2]</td>" in html
    assert "Noto Sans CJK SC" in html
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "docs/evidence/rl-grpo.md#results" in html and "证据引用" in html
    assert "[evidence:" not in html


def test_report_literals_ignore_citations_and_link_destinations():
    found = report_literals(SAMPLE)
    assert {"0.426", "0.444", "0.920", "1"} <= found
    assert "v9" not in found and not any("rl-grpo" in x for x in found)


def test_compare_literals_reports_missing_and_tolerates_wrapped_tokens():
    pdf_text = "0.426→0.444\n0.92\n列表项 code 1\nabc-\ndef 12,3\n45"
    missing = compare_literals({"0.426", "0.444", "0.920", "1", "12,345"}, pdf_text)
    assert missing == ["0.920"]
