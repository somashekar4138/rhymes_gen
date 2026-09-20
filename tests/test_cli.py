"""The flag surface and the exit-code contract.

0 success, 1 runtime/validation failure, 2 usage error.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import RenderingStubs

from rhymes import cli
from rhymes.styles import PRESETS


@pytest.fixture
def capture_request(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Record what reaches engine.render, without running it."""
    import rhymes.engine as engine_mod

    seen: list[object] = []

    def fake_render(req: object) -> Path:
        seen.append(req)
        return Path("/dev/null")

    monkeypatch.setattr(engine_mod, "render", fake_render)
    return seen


@pytest.fixture
def no_render(monkeypatch: pytest.MonkeyPatch):
    """Fail the test if engine.render is reached at all."""
    import rhymes.engine as engine_mod

    def boom(req: object) -> Path:
        raise AssertionError("engine.render must not be reached")

    monkeypatch.setattr(engine_mod, "render", boom)


def test_styles_lists_every_preset(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(["styles"])

    out = capsys.readouterr().out
    assert rc == 0
    for name in PRESETS:
        assert name in out


def test_style_flag_reaches_the_engine_as_its_expansion(
    capture_request: list, valid_lyrics_file: Path
) -> None:
    cli.main(["render", str(valid_lyrics_file), "--style", "lullaby"])

    assert capture_request[0].tags == PRESETS["lullaby"]


def test_tags_flag_reaches_the_engine_verbatim(
    capture_request: list, valid_lyrics_file: Path
) -> None:
    cli.main(["render", str(valid_lyrics_file), "--tags", "piano,gentle"])

    assert capture_request[0].tags == "piano,gentle"


def test_style_and_tags_together_is_a_usage_error(valid_lyrics_file: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["render", str(valid_lyrics_file), "--style", "lullaby", "--tags", "piano"])

    assert exc.value.code == 2


def test_empty_tags_is_a_usage_error(valid_lyrics_file: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["render", str(valid_lyrics_file), "--tags", ""])

    assert exc.value.code == 2


def test_unknown_style_exits_1_not_2(
    valid_lyrics_file: Path, no_render, capsys: pytest.CaptureFixture[str]
) -> None:
    """Deliberately not argparse `choices=`, which would exit 2. SPEC R2 wants 1."""
    rc = cli.main(["render", str(valid_lyrics_file), "--style", "deathmetal"])

    assert rc == 1
    err = capsys.readouterr().err
    for name in PRESETS:
        assert name in err


def test_defaults_are_nursery_rhyme_sized(capture_request: list, valid_lyrics_file: Path) -> None:
    cli.main(["render", str(valid_lyrics_file)])

    req = capture_request[0]
    assert req.seconds == 60
    assert req.temperature == 0.9
    assert req.topk == 50
    assert req.cfg_scale == 1.5
    assert req.seconds * 1000 <= 90_000


@pytest.mark.parametrize("seconds", ["5", "300", "30"])
def test_seconds_inside_the_range_reaches_the_engine(
    capture_request: list, valid_lyrics_file: Path, seconds: str
) -> None:
    cli.main(["render", str(valid_lyrics_file), "--seconds", seconds])

    assert capture_request[0].seconds == int(seconds)


@pytest.mark.parametrize("seconds", ["4", "0", "999", "-1"])
def test_seconds_outside_the_range_is_a_usage_error(valid_lyrics_file: Path, seconds: str) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["render", str(valid_lyrics_file), "--seconds", seconds])

    assert exc.value.code == 2


@pytest.mark.parametrize("flag", ["--temperature", "--topk", "--cfg-scale"])
def test_non_positive_sampling_parameters_are_usage_errors(
    valid_lyrics_file: Path, flag: str
) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["render", str(valid_lyrics_file), flag, "0"])

    assert exc.value.code == 2


def test_missing_lyrics_path_exits_1_naming_the_path(
    tmp_path: Path, no_render, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "nope.txt"

    rc = cli.main(["render", str(missing)])

    assert rc == 1
    assert str(missing) in capsys.readouterr().err


# SPEC acceptance criterion 5 is a CLI-level promise. Proving `LyricsError` is
# raised is a different claim from proving the command exits 1 and says which line.


def test_headerless_file_exits_1_at_the_cli_naming_the_line(
    tmp_path: Path, no_render, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.txt"
    bad.write_text("no header here\n", encoding="utf-8")

    rc = cli.main(["render", str(bad)])

    assert rc == 1
    err = capsys.readouterr().err
    assert "line 1" in err
    assert "Traceback" not in err


def test_empty_file_exits_1_at_the_cli(
    tmp_path: Path, no_render, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n", encoding="utf-8")

    rc = cli.main(["render", str(empty)])

    assert rc == 1
    assert "empty" in capsys.readouterr().err


def test_undecodable_file_exits_1_at_the_cli(
    tmp_path: Path, no_render, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"\xff\xfe[Verse]\nhello\n")

    rc = cli.main(["render", str(bad)])

    assert rc == 1
    assert "UTF-8" in capsys.readouterr().err


def test_successful_render_prints_the_output_path(
    rendering_stubs: RenderingStubs, valid_lyrics_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli.main(["render", str(valid_lyrics_file)])

    assert rc == 0
    assert str(valid_lyrics_file.with_suffix(".mp3")) in capsys.readouterr().out


@pytest.mark.parametrize("flag", ["--temperature", "--cfg-scale"])
def test_nan_sampling_parameters_are_usage_errors(valid_lyrics_file: Path, flag: str) -> None:
    """R-2: float('nan') does not raise, and IEEE-754 makes `nan <= 0` false, so
    NaN slipped past the positivity guard as if it were positive."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["render", str(valid_lyrics_file), flag, "nan"])

    assert exc.value.code == 2


def test_unknown_section_names_warn_but_still_render(
    rendering_stubs: RenderingStubs, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lyrics = tmp_path / "md.txt"
    lyrics.write_text("**Verse 1**\na line\n\n**Dance Break**\nanother\n", encoding="utf-8")

    rc = cli.main(["render", str(lyrics)])

    err = capsys.readouterr().err
    assert rc == 0
    assert "Verse 1" in err and "Dance Break" in err
    assert "Bridge" in err  # names the vocabulary the model does know


def test_standard_section_names_produce_no_warning(
    rendering_stubs: RenderingStubs, valid_lyrics_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli.main(["render", str(valid_lyrics_file)])

    assert rc == 0
    assert capsys.readouterr().err == ""
