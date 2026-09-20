"""Lyric validation: every malformed shape rejected, in under a second."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from rhymes.lyrics import Lyrics, LyricsError, load_lyrics, parse_lyrics


def write(tmp_path: Path, content: str | bytes, name: str = "r.txt") -> Path:
    p = tmp_path / name
    if isinstance(content, bytes):
        p.write_bytes(content)
    else:
        p.write_text(content, encoding="utf-8")
    return p


@pytest.mark.parametrize("content", ["", "   ", "\n\n\t\n"])
def test_empty_file_is_rejected(tmp_path: Path, content: str) -> None:
    with pytest.raises(LyricsError, match="empty"):
        load_lyrics(write(tmp_path, content))


def test_body_before_any_header_names_the_line(tmp_path: Path) -> None:
    content = "Twinkle twinkle little star\n[Verse]\nHow I wonder what you are\n"

    with pytest.raises(LyricsError) as exc:
        load_lyrics(write(tmp_path, content))

    assert "line 1" in str(exc.value)
    assert "Twinkle twinkle little star" in str(exc.value)


def test_file_with_no_header_anywhere_reports_its_first_line(tmp_path: Path) -> None:
    """SPEC R3 asks for the offending line, so the line-numbered message is the
    right one here too -- there is no separate "no header at all" case, because
    any non-blank line before a header already trips the line-numbered branch and
    a whitespace-only file is caught by the empty check first."""
    with pytest.raises(LyricsError) as exc:
        load_lyrics(write(tmp_path, "just some words\nand some more\n"))

    assert "line 1" in str(exc.value)
    assert "section header" in str(exc.value)


def test_undecodable_bytes_raise_lyrics_error_not_unicode_error(tmp_path: Path) -> None:
    with pytest.raises(LyricsError) as exc:
        load_lyrics(write(tmp_path, b"\xff\xfe[Verse]\nhello\n"))

    assert not isinstance(exc.value, UnicodeDecodeError)
    assert "UTF-8" in str(exc.value)


def test_bom_and_crlf_parse_identically_to_clean_input(tmp_path: Path) -> None:
    clean = "[Verse]\nTwinkle twinkle little star\n[Chorus]\nUp above the world so high\n"
    messy = b"\xef\xbb\xbf" + clean.replace("\n", "\r\n").encode("utf-8")

    assert load_lyrics(write(tmp_path, messy, "messy.txt")) == load_lyrics(
        write(tmp_path, clean, "clean.txt")
    )


def test_header_with_no_body_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(LyricsError, match=r"\[Chorus\]"):
        load_lyrics(write(tmp_path, "[Verse]\nsomething\n[Chorus]\n"))


@pytest.mark.parametrize(
    "header", ["Intro", "Verse", "Prechorus", "Chorus", "Bridge", "Outro", "VERSE", "chorus"]
)
def test_every_expected_header_spelling_parses(tmp_path: Path, header: str) -> None:
    lyrics = load_lyrics(write(tmp_path, f"[{header}]\na line\n"))

    assert len(lyrics.sections) == 1


def test_body_text_survives_parsing_verbatim(tmp_path: Path) -> None:
    """Prohibition: MUST NOT silently alter the user's lyric text.

    Mixed case, punctuation, internal double spaces, and a word a censor might
    object to. Nothing may be lowercased, replaced, reordered or dropped.
    """
    body = ["Oh  DEAR, what can the matter be?", "Johnny's so long at the fair -- damn it all!"]
    content = "[Verse]\n" + "\n".join(body) + "\n"

    lyrics = load_lyrics(write(tmp_path, content))

    assert [line for s in lyrics.sections for line in s.lines] == body


def test_rejecting_a_large_malformed_file_is_fast(tmp_path: Path) -> None:
    content = "\n".join(f"line {i} with no header anywhere" for i in range(200))
    path = write(tmp_path, content)

    start = time.perf_counter()
    with pytest.raises(LyricsError):
        load_lyrics(path)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.5


def test_parse_lyrics_round_trips_through_to_text() -> None:
    parsed = parse_lyrics("[Verse]\none\ntwo\n\n[Chorus]\nthree\n")

    assert isinstance(parsed, Lyrics)
    assert parse_lyrics(parsed.to_text()) == parsed


def test_whitespace_only_bracket_is_not_a_header(tmp_path: Path) -> None:
    """R-4: '[ ]' is length 3 so it passed the `len > 2` guard, then stripped to
    an empty section name that nothing rejected -- and `to_text()` re-emitted it
    as '[]', which no longer re-parses."""
    with pytest.raises(LyricsError) as exc:
        load_lyrics(write(tmp_path, "[ ]\nHello there\n"))

    assert "line 1" in str(exc.value)
