"""Document parsing with section detection.

Turns PDF and DOCX into a flat list of blocks, each carrying the section
it belongs to. Section tracking is the whole point: a chunk that knows it
came from "3.2 Admission Requirements > International" can cite that, and
retrieval can match on heading text as well as body text.

PDF heading detection is based on font size relative to the document's own
body text, not an absolute threshold. A thesis set in 11pt and a textbook
set in 9pt should both work, and a fixed cut-off would silently treat one
of them as having no headings at all.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Iterable

import pymupdf

# Numbered headings: "3.2 Something", "Chapter 4", "IV. Something",
# "APPENDIX B". Checked before font size because an explicit numbering
# marker is a stronger signal than typography.
_NUMBERED = re.compile(
    r"""^\s*
    (?:
        (?:chapter|section|appendix|part|unit|module)\s+\d+[A-Za-z]?
      | \d+(?:\.\d+){0,4}\.?\s+
      | [IVXLC]+\.\s+
    )
    \S""",
    re.IGNORECASE | re.VERBOSE,
)

# A short standalone line in caps or bold is very often a heading.
_ALL_CAPS = re.compile(r"^\s*[A-Z0-9][A-Z0-9\s'’\-:,&/]{2,70}\s*$")

_SENTENCE_END = re.compile(r"[.!?…:;]\s*$")


@dataclass
class Block:
    """One contiguous piece of text with its location in the source."""

    text: str
    page: int  # 1-based; DOCX is flat so everything is page 1
    heading_path: list[str] = field(default_factory=list)
    is_heading: bool = False

    @property
    def section(self) -> str:
        return " > ".join(self.heading_path) if self.heading_path else ""


@dataclass
class ParsedDocument:
    name: str
    blocks: list[Block]
    body_size: float | None = None

    @property
    def heading_count(self) -> int:
        return sum(1 for b in self.blocks if b.is_heading)


def _looks_like_heading(text: str, size: float | None, body_size: float | None) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 120:
        return False
    if _NUMBERED.match(stripped):
        return True
    if _ALL_CAPS.match(stripped) and not _SENTENCE_END.search(stripped):
        return True
    if size is not None and body_size is not None:
        # 8% larger than body text is a heading in most technical prose.
        # Smaller-than-body is a caption or a footnote, not a heading.
        if size >= body_size * 1.08:
            return True
    return False


def _clean(text: str) -> str:
    text = text.replace("­", "").replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def parse_pdf(data: bytes, name: str) -> ParsedDocument:
    """Parse a PDF, tracking font-relative heading structure."""
    doc = pymupdf.open(stream=data, filetype="pdf")

    # First pass: collect every span and its size, so we can find what
    # "normal body text" looks like in this specific document.
    spans: list[tuple[int, str, float]] = []
    per_page_lines: dict[int, list[tuple[str, float]]] = {}

    for page_index, page in enumerate(doc):
        page_no = page_index + 1
        lines: list[tuple[str, float]] = []
        raw = page.get_text("dict")
        for block in raw.get("blocks", []):
            if block.get("type") != 0:  # 0 = text, 1 = image
                continue
            for line in block.get("lines", []):
                pieces = [s.get("text", "") for s in line.get("spans", [])]
                text = _clean("".join(pieces))
                if not text:
                    continue
                # Use the tallest span on the line as the line's size: a
                # line with one bold word should not be classified by it.
                size = max((s.get("size", 0.0) for s in line.get("spans", [])), default=0.0)
                lines.append((text, size))
                spans.append((page_no, text, size))
        per_page_lines[page_no] = lines

    sizes = [s for _, _, s in spans if s > 0]
    if not sizes:
        return ParsedDocument(name=name, blocks=[])
    # Median, not mean: a document with many headings and captions has a
    # mean well above the body size, which would break the ratio test.
    body_size = statistics.median(sizes)

    # Second pass: rebuild blocks with heading paths.
    #
    # Depth is tracked as an explicit stack of (level, text) rather than a
    # plain path list. Truncating the path by length alone is wrong: two
    # sibling headings "2.1 Alpha" and "2.2 Beta" both sit at depth 1, so
    # path[:1] would keep the previous sibling instead of replacing it.
    blocks: list[Block] = []
    stack: list[tuple[int, str]] = []  # (depth, heading text)

    for page_no in sorted(per_page_lines):
        for text, size in per_page_lines[page_no]:
            is_heading = _looks_like_heading(text, size, body_size)
            if is_heading:
                level = _heading_level(text)
                if level is None:
                    # Unnumbered heading: nest one level below whatever is
                    # currently open, so it becomes a child of its context
                    # rather than a sibling of it.
                    level = (stack[-1][0] + 1) if stack else 0
                # Drop this level and anything deeper, then descend into it.
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, text))
                blocks.append(
                    Block(
                        text=text,
                        page=page_no,
                        heading_path=[h for _, h in stack],
                        is_heading=True,
                    )
                )
            else:
                blocks.append(
                    Block(
                        text=text,
                        page=page_no,
                        heading_path=[h for _, h in stack],
                        is_heading=False,
                    )
                )

    doc.close()
    return ParsedDocument(name=name, blocks=blocks, body_size=body_size)


def _heading_level(text: str) -> int | None:
    """Nesting depth implied by numbering, or None for unnumbered."""
    m = re.match(r"^\s*(\d+(?:\.\d+)*)\.?\s+\S", text)
    if m:
        return min(m.group(1).count("."), 6)
    m = re.match(r"^\s*(?:chapter|section|appendix|part|unit|module)\s+(\d+)", text, re.I)
    if m:
        return 0
    return None


def parse_docx(path: str, name: str) -> ParsedDocument:
    """Parse a .docx, using paragraph style names as the heading signal.

    python-docx exposes Word's own outline levels, which are far more
    reliable than heuristics. Headings are detected via the built-in
    "Heading N" styles; anything else is body text.
    """
    import docx  # imported lazily: only needed for .docx paths

    document = docx.Document(path)
    blocks: list[Block] = []
    path: list[str] = []

    for para in document.paragraphs:
        text = _clean(para.text)
        if not text:
            continue
        style = (para.style.name or "") if para.style is not None else ""
        m = re.match(r"^Heading\s+(\d+)$", style, re.I)
        if m:
            level = min(int(m.group(1)) - 1, 6)
            path = path[:level]
            path.append(text)
            blocks.append(
                Block(text=text, page=1, heading_path=list(path), is_heading=True)
            )
        else:
            blocks.append(
                Block(text=text, page=1, heading_path=list(path), is_heading=False)
            )

    return ParsedDocument(name=name, blocks=blocks)


def parse(path: str, data: bytes | None = None) -> ParsedDocument:
    """Dispatch on file extension."""
    lower = path.lower()
    name = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if lower.endswith(".pdf"):
        if data is None:
            data = open(path, "rb").read()
        return parse_pdf(data, name)
    if lower.endswith(".docx"):
        return parse_docx(path, name)
    raise ValueError(f"unsupported document type: {path}")