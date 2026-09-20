"""Read, normalize, parse and validate a lyrics file.

Pure: takes a path, reads that path, returns a value. No torch, no network, no
global state. This is the layer that lets a malformed file be rejected in under
a second instead of after a multi-minute model load, which is the whole of
SPEC R3.

Section headers may be bracketed (`[Verse]`) or Markdown (`**Verse 1**`,
`## Verse`), because that is what a chat or an editor actually produces. A
header is recognised only when the pattern matches the whole line, so
"I **really** like bananas" stays a lyric.

Body text is never touched. The only transformations permitted here are the
three declared ones -- BOM strip, newline normalization, and trimming trailing
whitespace from a line. Anything further would silently alter the user's
lyrics, which the prohibition forbids and
`test_body_text_survives_parsing_verbatim` catches.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# The section names HeartMuLa documents. Anything else still renders -- the user
# writes the song, not us -- but `unknown_sections` surfaces it so the CLI can
# say so rather than let a mystery reach the model.
KNOWN_SECTIONS = ("Intro", "Verse", "Prechorus", "Chorus", "Bridge", "Outro")

# A header may arrive bracketed, or as the Markdown a chat or editor produces.
# Each pattern must match the WHOLE line: "I **really** like bananas" is a lyric.
_HEADER_PATTERNS = (
    re.compile(r"\[(.+)\]"),  # [Verse]
    re.compile(r"\*\*(.+?)\*\*"),  # **Verse 1**
    re.compile(r"__(.+?)__"),  # __Bridge__
    re.compile(r"#{1,6}\s+(.+)"),  # ## Verse
)


class LyricsError(ValueError):
    """A lyrics file that the model could not be asked to sing."""


@dataclass(frozen=True)
class Section:
    name: str
    lines: tuple[str, ...]


@dataclass(frozen=True)
class Lyrics:
    sections: tuple[Section, ...]

    def to_text(self) -> str:
        """Render back to the bracketed-section format the model expects."""
        out: list[str] = []
        for section in self.sections:
            out.append(f"[{section.name}]")
            out.extend(section.lines)
            out.append("")
        return "\n".join(out).rstrip("\n") + "\n"


def load_lyrics(path: Path) -> Lyrics:
    """Parse `path` into `Lyrics`, or raise `LyricsError` saying why not."""
    with open(path, "rb") as fh:
        raw = fh.read()
    return parse_lyrics(_decode(raw, path))


def _decode(raw: bytes, path: Path) -> str:
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LyricsError(
            f"{path}: not valid UTF-8 at byte {exc.start} ({exc.reason}). "
            "Save the file as UTF-8 and try again."
        ) from exc
    return text.replace("\r\n", "\n").replace("\r", "\n")


def parse_lyrics(text: str) -> Lyrics:
    """Split normalized `text` into sections, rejecting every malformed shape."""
    if not text.strip():
        raise LyricsError("lyrics file is empty - write at least one [Verse] with a line under it")

    sections: list[Section] = []
    name: str | None = None
    lines: list[str] = []

    for number, line in enumerate(text.split("\n"), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        header = _header(stripped)
        if header is not None:
            if name is not None:
                _require_body(name, lines)
                sections.append(Section(name, tuple(lines)))
            name, lines = header, []
            continue
        if name is None:
            raise LyricsError(
                f"line {number}: {stripped!r} appears before any section header. "
                "Start the file with a section header such as [Verse]."
            )
        lines.append(line.rstrip())

    # `name` cannot still be None here: the text is non-blank, so the loop saw
    # at least one non-blank line, which either set `name` or already raised.
    assert name is not None
    _require_body(name, lines)
    sections.append(Section(name, tuple(lines)))
    return Lyrics(tuple(sections))


def _header(stripped: str) -> str | None:
    """Return the section name if this whole line is a header, else None."""
    for pattern in _HEADER_PATTERNS:
        match = pattern.fullmatch(stripped)
        if match:
            # `or None` rejects `[]`, `[ ]` and `****` alike. Guarding on length
            # instead let a whitespace-only bracket through as a section with an
            # empty name, which `to_text()` re-emitted as `[]` -- a shape that no
            # longer parses, breaking the round-trip every other input satisfies.
            return match.group(1).strip() or None
    return None


def unknown_sections(lyrics: Lyrics) -> list[str]:
    """Section names outside the vocabulary HeartMuLa documents, in order.

    Not an error: the user's structure is theirs. The CLI reports these so a
    `[Dance Break]` reaching the model as raw text is a known choice rather than
    a silent one.
    """
    known = {name.lower() for name in KNOWN_SECTIONS}
    seen: list[str] = []
    for section in lyrics.sections:
        if section.name.lower() not in known and section.name not in seen:
            seen.append(section.name)
    return seen


def _require_body(name: str, lines: list[str]) -> None:
    if not lines:
        raise LyricsError(f"section [{name}] has no lines under it")
