"""Render a Markdown report to a local PDF with Pillow (raster pages, Noto Sans CJK).

Superseded on 2026-10-29 by scripts/build_report_pdfs.sh (headless Chrome, text PDFs).
Pages here are drawn as grayscale images: the text is not selectable. `[evidence: path#anchor]` citations
become numbered notes listed at the end. Supports the subset the reports use: ATX
headings, paragraphs, nested lists, pipe tables, fences, inline code, bold, links, rules.

    .venv-ml/bin/python scripts/build_report_pdf.py reports/w2-async-grpo.md out.pdf
"""

import re
import sys
from pathlib import Path

from markdown_it import MarkdownIt
from PIL import Image, ImageDraw, ImageFont

DPI = 150
PAGE_W, PAGE_H = 1240, 1754  # A4 at 150 dpi
MARGIN = 110
WIDTH = PAGE_W - 2 * MARGIN
BOTTOM = PAGE_H - MARGIN
CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-{}.ttc"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
CJK_SC_INDEX = 2  # JP, KR, SC, TC, HK in the Noto CJK collection
CITATION = re.compile(r"\[evidence: ([^\]\s]+)\]")
ATOM = re.compile(r"[\x21-\x7e]+[ \t]*|[ \t]+|.", re.DOTALL)


def extract_citations(text, notes):
    def number(match):
        if match[1] not in notes:
            notes.append(match[1])
        return f"[{notes.index(match[1]) + 1}]"

    return CITATION.sub(number, text), notes


class Fonts:
    def __init__(self):
        self.cache = {}

    def get(self, style, size):
        key = (style, size)
        if key not in self.cache:
            if style == "mono":
                self.cache[key] = ImageFont.truetype(MONO, size)
            else:
                path = CJK.format("Bold" if style == "bold" else "Regular")
                self.cache[key] = ImageFont.truetype(path, size, index=CJK_SC_INDEX)
        return self.cache[key]

    def for_char(self, style, size, char):
        # DejaVu Sans Mono has no CJK glyphs; draw those in the CJK face.
        if style == "mono" and ord(char) > 0x2FFF:
            return self.get("regular", size)
        return self.get(style, size)


class Canvas:
    def __init__(self):
        self.fonts = Fonts()
        self.pages = []
        self.new_page()

    def new_page(self):
        self.page = Image.new("L", (PAGE_W, PAGE_H), 255)
        self.draw = ImageDraw.Draw(self.page)
        self.pages.append(self.page)
        self.y = MARGIN

    def ensure(self, height):
        if self.y + height > BOTTOM and self.y > MARGIN:
            self.new_page()

    def width(self, text, style, size):
        return sum(self.fonts.for_char(style, size, c).getlength(c) for c in text)

    def wrap(self, segments, size, width):
        """Greedy line fill over (text, style) segments; returns lines of segments."""
        lines, line, used = [], [], 0.0
        for text, style in segments:
            for atom in ATOM.findall(text.replace("\n", " ")):
                w = self.width(atom, style, size)
                if used + w > width and line and atom.strip():
                    lines.append(line)
                    line, used = [], 0.0
                    atom = atom.lstrip()
                    w = self.width(atom, style, size)
                while w > width and len(atom) > 1:  # break overlong tokens by character
                    cut = len(atom)
                    while cut > 1 and self.width(atom[:cut], style, size) > width - used:
                        cut -= 1
                    line.append((atom[:cut], style))
                    lines.append(line)
                    line, used = [], 0.0
                    atom = atom[cut:]
                    w = self.width(atom, style, size)
                if atom:
                    line.append((atom, style))
                    used += w
        if line:
            lines.append(line)
        return lines or [[]]

    def text_line(self, line, x, y, size, fill=0):
        for text, style in line:
            for char in text:
                font = self.fonts.for_char(style, size, char)
                self.draw.text((x, y), char, font=font, fill=fill)
                x += font.getlength(char)

    def paragraph(self, segments, size=21, indent=0, prefix=None, gap=10):
        lh = int(size * 1.55)
        lines = self.wrap(segments, size, WIDTH - indent)
        for i, line in enumerate(lines):
            self.ensure(lh)
            if prefix and i == 0:
                self.text_line([(prefix, "regular")], MARGIN + indent - 30, self.y, size)
            self.text_line(line, MARGIN + indent, self.y, size)
            self.y += lh
        self.y += gap

    def heading(self, segments, level):
        size = {1: 36, 2: 28, 3: 24}.get(level, 22)
        self.y += {1: 0, 2: 22, 3: 14}.get(level, 10)
        self.ensure(int(size * 1.6) * 2)
        self.paragraph([(t, "bold") for t, _ in segments], size=size, gap=6)
        if level <= 2:
            self.draw.line((MARGIN, self.y - 4, PAGE_W - MARGIN, self.y - 4), fill=170, width=2)
            self.y += 8

    def code(self, text, indent=0):
        size, lh = 17, 26
        lines = []
        for raw in text.rstrip("\n").split("\n"):
            lines.extend(self.wrap([(raw, "mono")], size, WIDTH - indent - 24))
        for line in lines:
            self.ensure(lh)
            self.draw.rectangle(
                (MARGIN + indent, self.y - 2, PAGE_W - MARGIN, self.y + lh - 2), fill=238
            )
            self.text_line(line, MARGIN + indent + 12, self.y, size)
            self.y += lh
        self.y += 12

    def table(self, rows):
        size, pad = 17, 8
        lh = int(size * 1.5)
        cols = max(len(r) for r in rows)
        rows = [r + [[]] * (cols - len(r)) for r in rows]
        natural = [
            max(self.width("".join(t for t, _ in r[c]), "regular", size) + 2 * pad for r in rows)
            for c in range(cols)
        ]
        widths = list(natural)
        if sum(widths) > WIDTH:
            floor = min(WIDTH / cols, 110)
            flexible = [max(w - floor, 0) for w in natural]
            spare = WIDTH - floor * cols
            widths = [floor + spare * f / (sum(flexible) or 1) for f in flexible]
        for r, row in enumerate(rows):
            style = "bold" if r == 0 else None
            wrapped = [
                self.wrap([(t, style or s) for t, s in cell], size, widths[c] - 2 * pad)
                for c, cell in enumerate(row)
            ]
            height = max(len(w) for w in wrapped) * lh + 2 * pad
            if self.y + height > BOTTOM:
                self.new_page()
            x = MARGIN
            for c, lines in enumerate(wrapped):
                box = (x, self.y, x + widths[c], self.y + height)
                self.draw.rectangle(box, fill=232 if r == 0 else 255, outline=150)
                for i, line in enumerate(lines):
                    self.text_line(line, x + pad, self.y + pad + i * lh, size)
                x += widths[c]
            self.y += height
        self.y += 16


def inline_segments(token):
    segments, style = [], []
    for child in token.children or []:
        if child.type in ("strong_open", "em_open"):
            style.append("bold")
        elif child.type in ("strong_close", "em_close"):
            style.pop()
        elif child.type == "code_inline":
            segments.append((child.content, "mono"))
        elif child.type in ("softbreak", "hardbreak"):
            segments.append((" ", "regular"))
        elif child.type in ("text", "html_inline"):
            segments.append((child.content, style[-1] if style else "regular"))
    return segments


def build(markdown, output):
    notes = []
    markdown, notes = extract_citations(markdown, notes)
    tokens = MarkdownIt("commonmark").enable("table").parse(markdown)
    canvas = Canvas()
    lists, marker, rows, row = [], None, None, None
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        indent = 40 * len(lists)
        if tok.type == "heading_open":
            canvas.heading(inline_segments(tokens[i + 1]), int(tok.tag[1]))
            i += 2
        elif tok.type in ("bullet_list_open", "ordered_list_open"):
            lists.append([tok.type == "ordered_list_open", int(tok.attrGet("start") or 1)])
        elif tok.type in ("bullet_list_close", "ordered_list_close"):
            lists.pop()
            if not lists:
                canvas.y += 6
        elif tok.type == "list_item_open":
            ordered, n = lists[-1]
            marker = f"{n}." if ordered else "•"
            lists[-1][1] += 1
        elif tok.type == "inline" and rows is None:
            canvas.paragraph(
                inline_segments(tok), indent=indent, prefix=marker, gap=4 if lists else 10
            )
            marker = None
        elif tok.type in ("fence", "code_block"):
            canvas.code(tok.content, indent=indent)
        elif tok.type == "table_open":
            rows = []
        elif tok.type == "tr_open":
            row = []
        elif tok.type == "inline" and rows is not None:
            row.append(inline_segments(tok))
        elif tok.type == "tr_close":
            rows.append(row)
        elif tok.type == "table_close":
            canvas.table(rows)
            rows = None
        elif tok.type == "hr":
            canvas.ensure(20)
            canvas.draw.line((MARGIN, canvas.y + 8, PAGE_W - MARGIN, canvas.y + 8), fill=170)
            canvas.y += 24
        i += 1
    if notes:
        canvas.heading([("证据引用", "bold")], 2)
        for n, note in enumerate(notes, 1):
            canvas.paragraph([(f"[{n}] ", "regular"), (note, "mono")], size=17, gap=2)
    total = len(canvas.pages)
    for n, page in enumerate(canvas.pages, 1):
        draw = ImageDraw.Draw(page)
        label = f"{n} / {total}"
        font = canvas.fonts.get("regular", 16)
        draw.text(((PAGE_W - font.getlength(label)) / 2, PAGE_H - 70), label, font=font, fill=110)
    first, *rest = canvas.pages
    first.save(output, "PDF", save_all=True, append_images=rest, resolution=DPI, quality=85)
    return total


def main(argv=None):
    source, output = (sys.argv[1:] if argv is None else argv)[:2]
    pages = build(Path(source).read_text(encoding="utf-8"), output)
    print(f"{output}: {pages} pages")


if __name__ == "__main__":
    main()
