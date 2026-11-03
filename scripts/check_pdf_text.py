"""Check that a text PDF carries the same numeric literals as its Markdown report.

The report side uses check_report_numbers' visible-text rules (citations and link
destinations removed). The PDF side is `pdftotext` output; a literal also counts as
present when it appears in the PDF text with all whitespace removed, which covers a
token wrapped across lines. Exit 1 if the PDF has no text or misses any literal.

    .venv/bin/python scripts/check_pdf_text.py reports/w2-async-grpo.md reports/pdf/w2-async-grpo.pdf
"""

import json
import re
import subprocess
import sys
from pathlib import Path

from scripts.check_report_numbers import literals, surfaces


def report_literals(markdown):
    found = set()
    for line, _ in surfaces(markdown):
        found |= literals(line)
    return found


def compare_literals(expected, pdf_text):
    present = literals(pdf_text)
    joined = re.sub(r"\s+", "", pdf_text)
    return sorted(x for x in expected if x not in present and x not in joined)


def pdf_text(path):
    return subprocess.run(["pdftotext", "-enc", "UTF-8", str(path), "-"], capture_output=True, text=True, check=True).stdout


def main(argv=None):
    report, pdf = (sys.argv[1:] if argv is None else argv)[:2]
    expected = report_literals(Path(report).read_text(encoding="utf-8"))
    text = pdf_text(pdf)
    missing = compare_literals(expected, text)
    result = {
        "report": report,
        "pdf": pdf,
        "pdf_text_chars": len(text.strip()),
        "report_literals": len(expected),
        "missing_in_pdf": missing,
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0 if text.strip() and not missing else 1


if __name__ == "__main__":
    sys.exit(main())
