"""Render a Markdown report to standalone HTML for headless Chrome's --print-to-pdf.

`[evidence: path#anchor]` citations become numbered notes listed at the end, as in the
earlier Pillow renderer; raw HTML in the Markdown is escaped. The page footer shows
"page / pages" through CSS @page margin boxes. scripts/build_report_pdfs.sh prints it.

    .venv-ml/bin/python scripts/build_report_html.py reports/w2-async-grpo.md out.html
"""

import html
import re
import sys
from pathlib import Path

from markdown_it import MarkdownIt

CITATION = re.compile(r"\[evidence: ([^\]\s]+)\]")
CSS = """
@page { size: A4; margin: 18mm 17mm 20mm 17mm;
  @bottom-center { content: counter(page) " / " counter(pages); font-size: 9pt; color: #666; } }
html { font-family: "Noto Sans CJK SC", "Noto Sans", sans-serif; font-size: 10.5pt; line-height: 1.55;
  color: #111; }
h1 { font-size: 19pt; margin: 0 0 10pt; border-bottom: 1.5pt solid #999; padding-bottom: 4pt; }
h2 { font-size: 14.5pt; margin: 16pt 0 6pt; border-bottom: 0.8pt solid #bbb; padding-bottom: 2pt;
  break-after: avoid; }
h3 { font-size: 12pt; margin: 12pt 0 4pt; break-after: avoid; }
p, li { overflow-wrap: break-word; }
code, pre { font-family: "DejaVu Sans Mono", "Noto Sans Mono CJK SC", monospace; font-size: 9pt; }
code { background: #f0f0f0; padding: 0 2pt; border-radius: 2pt; }
pre { background: #f0f0f0; padding: 6pt 8pt; white-space: pre-wrap; overflow-wrap: anywhere; }
pre code { padding: 0; background: none; }
table { border-collapse: collapse; margin: 6pt 0 10pt; font-size: 9pt; width: 100%; }
th, td { border: 0.6pt solid #999; padding: 3pt 5pt; vertical-align: top; overflow-wrap: break-word; }
th { background: #e8e8e8; }
tr { break-inside: avoid; }
a { color: #1a4f9c; text-decoration: none; }
ol.notes { font-size: 8.5pt; }
ol.notes code { font-size: 8pt; }
"""


def extract_citations(text):
    notes = []

    def number(match):
        if match[1] not in notes:
            notes.append(match[1])
        return f"[{notes.index(match[1]) + 1}]"

    return CITATION.sub(number, text), notes


def build_html(markdown, *, title):
    body_md, notes = extract_citations(markdown)
    md = MarkdownIt("commonmark", {"html": False}).enable("table")
    body = md.render(body_md)
    if notes:
        items = "".join(f"<li><code>{html.escape(n)}</code></li>" for n in notes)
        body += f'<h2>证据引用</h2>\n<ol class="notes">{items}</ol>\n'
    return (
        "<!DOCTYPE html>\n"
        f'<html lang="zh-CN"><head><meta charset="utf-8"><title>{html.escape(title)}</title>'
        f"<style>{CSS}</style></head><body>\n{body}</body></html>\n"
    )


def main(argv=None):
    source, output = (sys.argv[1:] if argv is None else argv)[:2]
    source = Path(source)
    Path(output).write_text(build_html(source.read_text(encoding="utf-8"), title=source.stem), encoding="utf-8")
    print(f"{output}: html from {source}")


if __name__ == "__main__":
    main()
