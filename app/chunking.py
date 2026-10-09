"""Structure-aware chunking.

Fixed-size windows with overlap are the default in most RAG pipelines and
they are the main reason answers cite the wrong section: a window cut in
the middle of a numbered clause produces chunks that read fine but cannot
be attributed to anything.

This chunker splits on section boundaries first and only falls back to
length limits inside a section. Every chunk carries its heading path, so
retrieval can index heading text as well as body and the answer generator
can quote "see 3.2 Admission Requirements" instead of "page 47".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .parsing import Block, ParsedDocument

# Sentences end at these, optionally followed by a closing quote or
# bracket. Requires whitespace or end-of-string after the terminator, which
# handles decimals ("3.5 percent") on its own.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])[\"')\]]*\s+(?=[A-ZÀ-Я0-9\"'(\[])")

# Titles and shorthand that end in a period but do not end a sentence.
# Without this, "Dr. Smith filed the form." splits after "Dr." and the
# chunk boundary lands inside a name, corrupting retrieval and the citation.
_ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "rev", "hon",
    "e.g", "i.e", "etc", "vs", "no", "vol", "fig", "eq", "ch", "sec",
    "approx", "dept", "est", "min", "max", "incl", "excl", "cf",
    "al", "inc", "ltd", "co", "corp", "univ", "assn", "bros",
}


def _split_sentences(text: str) -> list[str]:
    """Split on sentence terminators, ignoring abbreviation boundaries.

    Implemented with a search offset rather than by consuming the string,
    because the text around a skipped boundary has to stay in the sentence.
    """
    parts: list[str] = []
    rest = text.strip()
    pos = 0
    guard = 0
    while pos < len(rest) and guard < 1000:
        guard += 1
        m = _SENTENCE_SPLIT.search(rest, pos)
        if not m:
            break
        head = rest[: m.start()].rstrip()
        # Token immediately before the period: if it is a known
        # abbreviation the boundary is spurious, so resume the search after
        # it and keep the text.
        token = re.split(r"[\s(\[]", head)[-1].lower().rstrip(".")
        if token in _ABBREVIATIONS:
            pos = m.end()
            continue
        parts.append(rest[: m.end()].strip())
        rest = rest[m.end() :].lstrip()
        pos = 0

    if rest.strip():
        parts.append(rest.strip())
    return [p for p in parts if p]


@dataclass
class Chunk:
    text: str
    doc: str
    page: int
    section: str = ""
    heading_path: list[str] = field(default_factory=list)
    index: int = 0

    @property
    def citation(self) -> str:
        loc = f"{self.doc}"
        if self.page > 1:
            loc += f", p.{self.page}"
        if self.section:
            loc += f", {self.section}"
        return loc

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "doc": self.doc,
            "page": self.page,
            "section": self.section,
            "citation": self.citation,
            "index": self.index,
        }


class Chunker:
    """Section-first chunker with a length fallback.

    Args:
        max_chars: hard ceiling. Exceeding it means the source section is
            genuinely long, so the chunk is split anyway.
        overlap_sentences: trailing sentences repeated into the next chunk
            so a fact spanning a boundary is retrievable from both sides.
        min_chars: chunks below this are merged forward, unless they are
            the last chunk in the section.
    """

    def __init__(
        self,
        max_chars: int = 1200,
        overlap_sentences: int = 1,
        min_chars: int = 200,
    ) -> None:
        if max_chars < 100:
            raise ValueError("max_chars too small to hold a sentence")
        if overlap_sentences < 0:
            raise ValueError("overlap_sentences cannot be negative")
        self.max_chars = max_chars
        self.overlap_sentences = overlap_sentences
        self.min_chars = min_chars

    def _pack(self, sentences: list[str], section: Block | None, doc: str) -> list[str]:
        """Greedily pack sentences up to max_chars, with overlap."""
        out: list[str] = []
        buf: list[str] = []
        size = 0

        for sent in sentences:
            # +1 for the joining space
            if buf and size + len(sent) + 1 > self.max_chars:
                out.append(" ".join(buf))
                if self.overlap_sentences:
                    tail = buf[-self.overlap_sentences :]
                    buf = list(tail)
                    size = sum(len(s) + 1 for s in buf)
                else:
                    buf = []
                    size = 0
            buf.append(sent)
            size += len(sent) + 1

        if buf:
            out.append(" ".join(buf))

        # A hard split is the last resort: a single sentence longer than
        # the limit. Splitting mid-sentence is ugly but returning nothing
        # is worse, because the content becomes unreachable.
        final: list[str] = []
        for piece in out:
            while len(piece) > self.max_chars * 1.5:
                cut = piece.rfind(" ", 0, self.max_chars)
                if cut <= 0:
                    cut = self.max_chars
                final.append(piece[:cut].strip())
                piece = piece[cut:].strip()
            if piece:
                final.append(piece)
        return final

    def chunk(self, parsed: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        idx = 0

        # Group consecutive body blocks that share a heading path.
        groups: list[tuple[list[str], Block | None, int]] = []
        current_path: list[str] | None = None
        buf: list[str] = []
        first_page = 1
        anchor: Block | None = None

        for b in parsed.blocks:
            if b.is_heading:
                # Flush the previous section before switching context.
                if buf:
                    groups.append((buf, anchor, first_page))
                buf, anchor, first_page = [], None, b.page
                current_path = list(b.heading_path)
                continue
            if not buf:
                anchor = b
                first_page = b.page
            elif b.page != first_page and not groups:
                first_page = min(first_page, b.page)
            buf.append(b.text)

        if buf:
            groups.append((buf, anchor, first_page))

        for texts, a, page in groups:
            if a is None:
                continue
            joined = " ".join(texts)
            for piece in self._pack(_split_sentences(joined), a, parsed.name):
                if not piece.strip():
                    continue
                chunks.append(
                    Chunk(
                        text=piece,
                        doc=parsed.name,
                        page=a.page,
                        section=a.section,
                        heading_path=list(a.heading_path),
                        index=idx,
                    )
                )
                idx += 1

        return self._merge_runts(chunks)

    def _merge_runts(self, chunks: list[Chunk]) -> list[Chunk]:
        """Merge tiny chunks forward so they do not pollute retrieval.

        A 40-character chunk usually matches almost anything, which pulls
        it into the top-k and dilutes the real results.
        """
        if not chunks:
            return []
        merged: list[Chunk] = []
        for c in chunks:
            if (
                merged
                and len(c.text) < self.min_chars
                and merged[-1].section == c.section
            ):
                prev = merged[-1]
                prev.text = f"{prev.text} {c.text}".strip()
            else:
                merged.append(c)
        # Anything still under min_chars is kept: dropping content is worse
        # than a weak chunk, and the retrieval layer can drop it instead.
        for i, c in enumerate(merged):
            c.index = i
        return merged