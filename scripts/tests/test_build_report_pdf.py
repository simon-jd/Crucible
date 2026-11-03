"""Runs in test-ml: Pillow and markdown-it live in .venv-ml only."""

import pytest

pytest.importorskip("PIL")
pytest.importorskip("markdown_it")

from scripts.build_report_pdf import build, extract_citations

SAMPLE = """# 标题 Title

第一段，含数字 0.426→0.444。[evidence: docs/evidence/rl-grpo.md#results]

| 列 | 值 |
|---|---|
| GSM8K | 0.920 [evidence: docs/evidence/pd-disaggregation.md#results] |

- 列表项 `code 1`
- **加粗** 和 [链接](https://example.com)

```sh
make test
```
"""


def test_citations_become_numbered_notes_reusing_numbers():
    text, notes = extract_citations(
        "a [evidence: x.md#a] b [evidence: y.md#b] c [evidence: x.md#a]", []
    )
    assert text == "a [1] b [2] c [1]"
    assert notes == ["x.md#a", "y.md#b"]


def test_build_writes_a_pdf_with_cjk_text(tmp_path):
    out = tmp_path / "r.pdf"
    pages = build(SAMPLE, out)
    data = out.read_bytes()
    assert data.startswith(b"%PDF") and pages == 1


def test_long_document_paginates(tmp_path):
    out = tmp_path / "long.pdf"
    pages = build("# T\n\n" + "\n\n".join(f"段落 {i} " * 40 for i in range(80)), out)
    assert pages > 1
