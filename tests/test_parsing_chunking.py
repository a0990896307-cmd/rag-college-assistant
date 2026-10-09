"""Tests for parsing and structure-aware chunking.

The point of these tests is the section metadata, not the text. A chunker
that produces readable text but loses the heading path gives an answer
generator nothing to cite, which is the failure mode that matters.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf

from app.chunking import Chunker
from app.parsing import Block, ParsedDocument, parse_pdf

failures: list[str] = []
checks = 0


def check(name: str, actual, expected) -> None:
    global checks
    checks += 1
    if actual != expected:
        failures.append(f"{name}\n     expected: {expected!r}\n     actual:   {actual!r}")


def check_true(name: str, value) -> None:
    check(name, bool(value), True)


# --------------------------------------------------------------- fixtures

def make_pdf(lines: list[tuple[str, float]]) -> bytes:
    """Build a one-page PDF from (text, font_size) pairs."""
    doc = pymupdf.open()
    page = doc.new_page()
    y = 800
    for text, size in lines:
        page.insert_text((60, y), text, fontsize=size, fontname="helv")
        y -= size + 6
    data = doc.tobytes()
    doc.close()
    return data


BODY = 11.0
DOC_LINES = [
    ("ADMISSION REQUIREMENTS", BODY + 2.5),
    ("Applicants must hold a completed secondary certificate.", BODY),
    ("Proof of qualification is submitted with the application form.", BODY),
    ("International applicants additionally require certification of", BODY),
    ("their qualification in the country of origin.", BODY),
    ("3.2 Entrance Examination", BODY + 1.5),
    ("The entrance examination consists of two written papers.", BODY),
    ("Each paper is graded on a scale from one to six.", BODY),
    ("Results are published within six weeks of the examination date.", BODY),
    ("Application deadlines are set by the admissions committee.", BODY),
]


# ------------------------------------------------------------------ tests

parsed = parse_pdf(make_pdf(DOC_LINES), "admissions.pdf")

check("document name kept", parsed.name, "admissions.pdf")
check_true("blocks produced", parsed.blocks)
check_true("body size detected", parsed.body_size and 10.0 < parsed.body_size < 12.0)
check("two headings found", parsed.heading_count, 2)

# The numbered heading must nest under the ALL-CAPS parent.
numbered = [b for b in parsed.blocks if b.text.startswith("3.2")]
check_true("numbered heading detected", numbered)
if numbered:
    check(
        "nesting path correct",
        numbered[0].heading_path,
        ["ADMISSION REQUIREMENTS", "3.2 Entrance Examination"],
    )
    check(
        "section string joined",
        numbered[0].section,
        "ADMISSION REQUIREMENTS > 3.2 Entrance Examination",
    )

caps = [b for b in parsed.blocks if b.text == "ADMISSION REQUIREMENTS"]
check_true("all-caps heading at root", caps)
if caps:
    check("root heading has no parent", caps[0].heading_path, ["ADMISSION REQUIREMENTS"])

# Body text must not be misread as a heading.
body_after = [
    b for b in parsed.blocks
    if not b.is_heading and b.text.startswith("Applicants must hold")
]
check_true("body text is not a heading", body_after)
if body_after:
    check(
        "body carries parent path",
        body_after[0].heading_path,
        ["ADMISSION REQUIREMENTS"],
    )

# ------------------------------------------------------------- chunking

chunker = Chunker(max_chars=400, overlap_sentences=1, min_chars=80)
chunks = chunker.chunk(parsed)

check_true("chunks produced", chunks)
check("all chunks carry a section", all(c.section for c in chunks), True)
check("citation includes doc name", chunks[0].citation.startswith("admissions.pdf"), True)
check_true("citation includes section", "ADMISSION REQUIREMENTS" in chunks[0].citation)
check_true("chunk indices contiguous", [c.index for c in chunks] == list(range(len(chunks))))

# Nothing may be silently dropped.
joined = " ".join(c.text for c in chunks)
for fact in [
    "completed secondary certificate",
    "Proof of qualification",
    "certification of",
    "two written papers",
    "one to six",
    "six weeks",
    "admissions committee",
]:
    check_true(f"content preserved: {fact[:32]}", fact in joined)

# Heading text must not become an ordinary body chunk.
check(
    "headings not emitted as body chunks",
    any("ADMISSION REQUIREMENTS" in c.text and len(c.text) < 40 for c in chunks),
    False,
)

# Chunks must respect the length ceiling except for the documented
# single-sentence fallback.
check_true(
    "respects max_chars",
    all(len(c.text) <= 400 * 1.5 for c in chunks),
)

# ------------------------------------------------------------ edge cases

empty = parse_pdf(make_pdf([]), "empty.pdf")
check("empty pdf yields no blocks", empty.blocks, [])
check("empty chunking yields no chunks", chunker.chunk(empty), [])

# A document with no headings at all must still chunk, with an empty path.
flat = parse_pdf(
    make_pdf([("First sentence of a document with no structure.", BODY),
              ("Second sentence continues the same thought.", BODY)]),
    "flat.pdf",
)
flat_chunks = chunker.chunk(flat)
check_true("flat document still chunks", flat_chunks)
check("flat document has empty section", flat_chunks[0].section, "")

# Sibling headings must replace, not accumulate.
siblings = parse_pdf(
    make_pdf([("2.1 Alpha", BODY + 1.5),
              ("Body of alpha.", BODY),
              ("2.2 Beta", BODY + 1.5),
              ("Body of beta.", BODY)]),
    "siblings.pdf",
)
beta = [b for b in siblings.blocks if b.text.startswith("2.2")]
if beta:
    check("sibling does not nest under previous", len(beta[0].heading_path), 1)
    check("sibling is its own root", beta[0].heading_path, ["2.2 Beta"])

# Abbreviations must not create false sentence splits.
abbr = Chunker(max_chars=120, overlap_sentences=0, min_chars=1)
from app.chunking import _split_sentences
check(
    "abbreviation not split",
    _split_sentences("Dr. Smith filed the form. It was accepted."),
    ["Dr. Smith filed the form.", "It was accepted."],
)
check(
    "decimal not split",
    _split_sentences("The rate is 3.5 percent today. It holds."),
    ["The rate is 3.5 percent today.", "It holds."],
)
check(
    "e.g. not split",
    _split_sentences("Use e.g. this method. Then stop."),
    ["Use e.g. this method.", "Then stop."],
)

# Overlap must actually duplicate content across the boundary.
# The contract is sentence-level: with overlap_sentences=1 the final
# sentence of a chunk is repeated at the head of the next one.
ov = Chunker(max_chars=200, overlap_sentences=1, min_chars=1)
long_doc = parse_pdf(
    make_pdf([("1 Body", BODY + 1.5)]
             + [(f"Sentence number {i} carries some payload text.", BODY)
                for i in range(40)]),
    "long.pdf",
)
ov_chunks = ov.chunk(long_doc)
check_true("long doc split into several", len(ov_chunks) > 1)
if len(ov_chunks) > 1:
    from app.chunking import _split_sentences as _ss
    last = _ss(ov_chunks[0].text)[-1]
    first = _ss(ov_chunks[1].text)[0]
    check("overlap repeats the trailing sentence", first, last)

# Constructor must reject nonsense rather than silently misbehave.
for bad, msg in [
    ({"max_chars": 10}, "too small"),
    ({"overlap_sentences": -1}, "negative"),
]:
    try:
        Chunker(**bad)
        check(f"rejects {msg}", "no error", "ValueError")
    except ValueError:
        check(f"rejects {msg}", "ValueError", "ValueError")

# Unsupported type must raise, not return an empty document.
try:
    from app.parsing import parse
    parse("notes.txt")
    check("unsupported type raises", "no error", "ValueError")
except ValueError:
    check("unsupported type raises", "ValueError", "ValueError")


# ------------------------------------------------------------------ report

print()
if failures:
    print(f"FAILED {len(failures)}/{checks}")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print(f"PASSED all {checks} checks")