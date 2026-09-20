"""The tracer: one lyric file through all four layers to a file at the target path.

Every test here takes `rendering_stubs`. None patches `_generate` directly --
that is precisely the failure this fixture exists to prevent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rhymes import cli
from tests.conftest import RenderingStubs


def test_render_writes_mp3_beside_the_input(
    rendering_stubs: RenderingStubs, valid_lyrics_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli.main(["render", str(valid_lyrics_file)])

    assert rc == 0
    out = valid_lyrics_file.with_suffix(".mp3")
    assert out.exists()
    assert out.stat().st_size > 0


def test_out_flag_overrides_the_derived_path(
    rendering_stubs: RenderingStubs, valid_lyrics_file: Path, tmp_path: Path
) -> None:
    other = tmp_path / "elsewhere.mp3"

    rc = cli.main(["render", str(valid_lyrics_file), "-o", str(other)])

    assert rc == 0
    assert other.exists()
    assert not valid_lyrics_file.with_suffix(".mp3").exists()


def test_failed_generation_leaves_nothing_at_the_target(
    rendering_stubs: RenderingStubs,
    valid_lyrics_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def boom(ckpt_root, lyrics_path, tags_path, tmp_out, req):
        raise RuntimeError("generation exploded")

    rendering_stubs.set_generate(boom)
    out = tmp_path / "doomed.mp3"

    rc = cli.main(["render", str(valid_lyrics_file), "-o", str(out)])

    assert rc == 1
    assert not out.exists()
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert len(err.strip().splitlines()) == 1


def test_generate_receives_a_temp_path_not_the_target(
    rendering_stubs: RenderingStubs, valid_lyrics_file: Path, tmp_path: Path
) -> None:
    out = tmp_path / "final.mp3"

    rc = cli.main(["render", str(valid_lyrics_file), "-o", str(out)])

    assert rc == 0
    assert len(rendering_stubs.calls) == 1
    assert rendering_stubs.calls[0]["tmp_out"] != out
