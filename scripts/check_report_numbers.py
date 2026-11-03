"""Check exact numeric literals against same-line, anchored Markdown evidence.

This small deterministic subset is documented in reports/repro/README.md. Passing
proves literal/citation consistency, never experiment authenticity or acceptance.
"""

import argparse
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

CITATION = re.compile(r"\[evidence: ([^\]\s]+)\]")
CITATION_START = re.compile(r"\[evidence\s*:")
HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)\s*#*\s*$")
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
LIST = re.compile(r"^( {0,3})(?:\d+[.)]|[-+*])[ \t]+")
# Consume the whole standalone signed numeric/identifier candidate, rather than
# rejecting it at punctuation and falling back to unsigned/grouped fragments.
# Dots continue the candidate only before another word segment, retaining versions
# while leaving sentence-ending dots outside the literal.
LITERAL = re.compile(
    r"(?<![A-Za-z0-9_.-])[+-]?(?:"
    r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
    r"(?:[eE][+-]?\d+)?[A-Za-z0-9_]*(?:[.-][A-Za-z0-9_]+)*%?"
    r"|[A-Za-z0-9_]+(?:[.-][A-Za-z0-9_]+)*",
    re.ASCII,
)


@dataclass
class Result:
    claims: int = 0
    errors: list[str] = field(default_factory=list)


def without_destinations(text):
    """Remove balanced inline destinations and autolinks, retaining visible text."""
    text = re.sub(r"<(?:https?://|mailto:)[^<>\s]*>", "", text)
    cursor = 0
    while (start := text.find("](", cursor)) >= 0:
        depth, end = 1, start + 2
        while end < len(text) and depth:
            if text[end] == "\\":
                end += 2
                continue
            depth += (text[end] == "(") - (text[end] == ")")
            end += 1
        if depth:
            break  # Unsupported/incomplete syntax is conservatively still checked.
        text = text[: start + 1] + text[end:]
        cursor = start + 1
    return re.sub(r"^( {0,3}\[[^\]]+\]:)[ \t]+\S+", r"\1", text)


def surfaces(text):
    """Yield (filtered physical line, actual ATX heading or None)."""
    fence = None
    for original in text.splitlines():
        line = CITATION.sub("", original)
        marker = FENCE.match(line)
        if fence:
            if (
                marker
                and marker[1][0] == fence[0]
                and len(marker[1]) >= len(fence)
                and not marker[2].strip()
            ):
                fence = None
                yield "", None
                continue
            yield line, None
        elif marker and (marker[1][0] != "`" or "`" not in marker[2]):
            # Backticks in a backtick info string disqualify a fence opener;
            # same-line triple/longer code spans must reach the inline parser.
            fence = marker[1]
            yield "", None
        else:
            heading = HEADING.match(line)
            line = LIST.sub(r"\1", line)
            yield without_markup_destinations(line), heading


def without_markup_destinations(line):
    """Keep exact-run backtick spans literal, including embedded shorter runs."""
    output, cursor = [], 0
    while opening := re.search(r"`+", line[cursor:]):
        start, end = cursor + opening.start(), cursor + opening.end()
        output.append(without_destinations(line[cursor:start]))
        size = end - start
        closing = next((run for run in re.finditer(r"`+", line[end:]) if len(run[0]) == size), None)
        if closing is None:
            # An unsupported/unclosed span is checked conservatively as literal text.
            output.append(line[start:])
            return "".join(output)
        cursor = end + closing.end()
        output.append(line[start:cursor])
    output.append(without_destinations(line[cursor:]))
    return "".join(output)


def literals(text):
    return {match[0] for match in LITERAL.finditer(text) if re.search(r"[0-9]", match[0])}


def unsupported_numeric(text):
    return sorted({c for c in text if ord(c) > 127 and unicodedata.category(c).startswith("N")})


def slug(title):
    title = re.sub(r"[`*_]", "", title).lower()
    title = re.sub(r"[^\w\-\s]", "", title)
    return re.sub(r"\s", "-", title)


def section(source, anchor):
    lines = list(surfaces(source))
    headings = [(i, len(h[1]), slug(h[2])) for i, (_, h) in enumerate(lines) if h]
    selected = [(i, level) for i, level, name in headings if name == anchor]
    if len(selected) != 1:
        reason = "ambiguous" if selected else "missing"
        raise ValueError(f"{reason} anchor #{anchor}")
    start, level = selected[0]
    end = next((i for i, depth, _ in headings if i > start and depth <= level), len(lines))
    return "\n".join(line for line, _ in lines[start:end])


def evidence_tokens(reference, root):
    if reference.count("#") != 1:
        raise ValueError("citation requires exactly one nonempty #anchor")
    filename, anchor = reference.split("#")
    if not anchor:
        raise ValueError("citation requires a nonempty #anchor")
    path = PurePosixPath(filename)
    if (
        path.is_absolute()
        or "\\" in filename
        or any(part in {".", ".."} for part in filename.split("/"))
        or path.parts[:2] != ("docs", "evidence")
    ):
        raise ValueError("citation path must stay within docs/evidence without traversal")
    directory = root / "docs/evidence"
    if directory.resolve() != directory:
        raise ValueError("evidence directory symlink is not allowed")
    resolved = (root / filename).resolve(strict=True)
    if not resolved.is_relative_to(directory) or not resolved.is_file():
        raise ValueError("citation must resolve to a contained regular evidence file")
    selected = section(resolved.read_text(encoding="utf-8"), anchor)
    if glyphs := unsupported_numeric(selected):
        raise ValueError(f"unsupported Unicode numeric glyphs in evidence: {glyphs!r}")
    return literals(selected)


def check_reports(paths, *, project_root):
    root = Path(project_root).resolve()
    result = Result()
    if not paths:
        result.errors.append("<inputs>:1: at least one report input is required")
    for path in paths:
        path = Path(path)
        if not path.is_absolute():
            path = root / path
        prefix = str(path)
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            result.errors.append(f"{prefix}:1: cannot read report: {exc}")
            continue
        if not content.strip():
            result.errors.append(f"{prefix}:1: empty report")
            continue
        count = 0
        for number, (original, (visible, _)) in enumerate(
            zip(content.splitlines(), surfaces(content), strict=True), 1
        ):
            location = f"{prefix}:{number}:"
            citations = list(CITATION.finditer(original))
            if CITATION_START.search(CITATION.sub("", original)):
                result.errors.append(f"{location} malformed evidence citation")
            supported = set()
            for citation in citations:
                try:
                    supported.update(evidence_tokens(citation[1], root))
                except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
                    result.errors.append(f"{location} invalid citation {citation[0]}: {exc}")
            if glyphs := unsupported_numeric(visible):
                result.errors.append(f"{location} unsupported Unicode numeric glyphs: {glyphs!r}")
            claims = literals(visible)
            count += len(claims)
            for claim in sorted(claims - supported):
                reason = "unsupported literal" if citations else "uncited literal"
                result.errors.append(f"{location} {reason} {claim!r}")
        result.claims += count
        if count == 0:
            result.errors.append(f"{prefix}:1: report has no checked numeric claims")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("reports", nargs="*", type=Path)
    args = parser.parse_args(argv)
    result = check_reports(args.reports, project_root=args.project_root)
    for error in result.errors:
        print(error, file=sys.stderr)
    if result.errors:
        return 1
    print(f"PASS: {len(args.reports)} reports; {result.claims} numeric literals checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
