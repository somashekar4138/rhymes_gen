"""The heartlib boundary: preflight, checkpoint bootstrap, the pipeline call.

Every case fakes torch (and later heartlib) by inserting a stub into
sys.modules for the duration of the test. heartlib and a real CUDA device are
both absent on the dev machine, and STANDARDS.md forbids reaching into a
heartlib internal to work around that -- the seam we mock is our own.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rhymes import cli, engine


def test_no_cuda_raises_and_names_colab(fake_torch) -> None:
    fake_torch(cuda_available=False)

    with pytest.raises(engine.EngineError) as exc:
        engine.preflight(None)

    message = str(exc.value)
    assert "CUDA" in message
    assert "Colab" in message


def test_cuda_available_resolves_to_cuda(fake_torch) -> None:
    fake_torch(cuda_available=True)

    assert engine.preflight(None) == "cuda"


def test_explicit_device_overrides_with_a_stated_warning(
    fake_torch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_torch(cuda_available=False)

    assert engine.preflight("cpu") == "cpu"
    assert "unsupported" in capsys.readouterr().err


def test_unparseable_device_surfaces_its_reason_not_a_traceback(fake_torch) -> None:
    fake_torch(cuda_available=False, device_raises=True)

    with pytest.raises(engine.EngineError) as exc:
        engine.preflight("nonsense")

    assert "device string" in str(exc.value)


def test_render_preflights_before_reaching_for_checkpoints(
    fake_torch, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Nothing is downloaded on a machine that cannot generate."""
    fake_torch(cuda_available=False)

    def must_not_run(cache_dir: Path | None = None) -> Path:
        raise AssertionError("ensure_checkpoints reached on a machine with no CUDA")

    monkeypatch.setattr(engine, "ensure_checkpoints", must_not_run, raising=True)

    req = engine.RenderRequest(
        lyrics_text="[Verse]\nhello\n",
        tags="piano",
        out_path=tmp_path / "out.mp3",
        seconds=30,
        temperature=0.9,
        topk=50,
        cfg_scale=1.5,
    )

    with pytest.raises(engine.EngineError):
        engine.render(req)


def test_cuda_less_render_exits_1_through_the_cli(
    fake_torch, valid_lyrics_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Asserting preflight raises is a different claim from asserting the user
    sees exit 1 and one line. SPEC R4 is about the second."""
    fake_torch(cuda_available=False)

    rc = cli.main(["render", str(valid_lyrics_file)])

    assert rc == 1
    err = capsys.readouterr().err
    assert "Colab" in err
    assert "Traceback" not in err
    assert len(err.strip().splitlines()) == 1
