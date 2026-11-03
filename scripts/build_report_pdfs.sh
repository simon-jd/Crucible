#!/usr/bin/env bash
# Text PDFs of the W2/W3 reports: Markdown -> HTML (scripts/build_report_html.py) ->
# headless Chrome --print-to-pdf (user decision 2026-10-29). Each run uses a fresh
# --user-data-dir under .cache/t16/ and --no-first-run, so the desktop user's Chrome
# profile is never read or written. Then scripts/check_pdf_text.py compares numbers.
# usage: scripts/build_report_pdfs.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
stamp=$(date +%Y%m%dT%H%M%S%N)
profile="$PWD/.cache/t16/chrome-profile-$stamp"
html_dir="$PWD/.cache/t16/html"
mkdir -p "$html_dir" reports/pdf
mkdir "$profile"  # fresh: fails if it already exists
export TMPDIR="$PWD/.cache/t16/tmp"
mkdir -p "$TMPDIR"
for name in w2-async-grpo w3-disaggregated-serving; do
  .venv-ml/bin/python scripts/build_report_html.py "reports/$name.md" "$html_dir/$name.html"
  google-chrome --headless --no-first-run --user-data-dir="$profile" --no-default-browser-check \
    --disable-extensions --disable-sync --disable-gpu --no-pdf-header-footer \
    --print-to-pdf="$PWD/reports/pdf/$name.pdf" "file://$html_dir/$name.html"
  pdfinfo "reports/pdf/$name.pdf" | grep -E '^(Pages|Producer|Creator|Page size):'
  .venv/bin/python -m scripts.check_pdf_text "reports/$name.md" "reports/pdf/$name.pdf"
done
sha256sum reports/pdf/*.pdf
