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


# --- Checkpoint bootstrap and the pipeline call -----------------------------


EXPECTED_CHECKPOINTS = [
    ("HeartMuLa/HeartMuLaGen", ""),
    ("HeartMuLa/HeartMuLa-oss-3B-happy-new-year", "HeartMuLa-oss-3B"),
    ("HeartMuLa/HeartCodec-oss-20260123", "HeartCodec-oss"),
]


def test_checkpoints_table_names_the_load_bearing_directories() -> None:
    """from_pretrained looks for exactly these directory names."""
    assert list(engine.CHECKPOINTS) == EXPECTED_CHECKPOINTS


def test_ensure_checkpoints_delegates_all_three(
    download_recorder: list[dict], tmp_path: Path
) -> None:
    root = engine.ensure_checkpoints(tmp_path)

    assert root == tmp_path
    assert [c["repo_id"] for c in download_recorder] == [r for r, _ in EXPECTED_CHECKPOINTS]
    for call, (_, rel) in zip(download_recorder, EXPECTED_CHECKPOINTS, strict=True):
        assert call["local_dir"] == str(tmp_path / rel if rel else tmp_path)


def test_ensure_checkpoints_implements_no_downloader_of_its_own(
    download_recorder: list[dict], tmp_path: Path
) -> None:
    """Every repo arrives through the recorder, so nothing is fetched by
    another route. Resume and partial-file handling are the library's to own,
    which is exactly why they are delegated: a hand-rolled existence check is
    how a truncated checkpoint gets treated as complete."""
    engine.ensure_checkpoints(tmp_path)

    assert len(download_recorder) == len(EXPECTED_CHECKPOINTS)


def test_second_run_delegates_again_and_creates_no_second_cache(
    download_recorder: list[dict], tmp_path: Path
) -> None:
    first = engine.ensure_checkpoints(tmp_path)
    before = sorted(p.name for p in tmp_path.iterdir())
    second = engine.ensure_checkpoints(tmp_path)

    assert first == second == tmp_path
    assert sorted(p.name for p in tmp_path.iterdir()) == before
    assert len(download_recorder) == 2 * len(EXPECTED_CHECKPOINTS)


def test_cache_dir_honours_the_environment(
    monkeypatch: pytest.MonkeyPatch, download_recorder: list[dict], tmp_path: Path
) -> None:
    monkeypatch.setenv("RHYMES_CACHE_DIR", str(tmp_path / "elsewhere"))

    assert engine.ensure_checkpoints() == tmp_path / "elsewhere"


@pytest.fixture
def fake_heartlib(monkeypatch: pytest.MonkeyPatch):
    """A stand-in for heartlib that records how it was built and called."""
    import sys
    import types

    record: dict = {}

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            record["model_path"] = model_path
            record.update(kwargs)
            return cls()

        def __call__(self, inputs, **kwargs):
            record["inputs"] = inputs
            record["lyrics_text"] = Path(inputs["lyrics"]).read_text(encoding="utf-8")
            record["tags_text"] = Path(inputs["tags"]).read_text(encoding="utf-8")
            record["call_kwargs"] = kwargs
            Path(kwargs["save_path"]).write_bytes(b"ID3fake")

    mod = types.ModuleType("heartlib")
    mod.HeartMuLaGenPipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "heartlib", mod)
    return record


def test_generate_builds_and_calls_the_pipeline_as_the_model_expects(
    fake_torch, fake_heartlib: dict, download_recorder: list[dict], tmp_path: Path
) -> None:
    fake_torch(cuda_available=True)
    out = tmp_path / "song.mp3"
    req = engine.RenderRequest(
        lyrics_text="[Verse]\nTwinkle twinkle\n",
        tags="piano,gentle",
        out_path=out,
        seconds=30,
        temperature=0.8,
        topk=42,
        cfg_scale=2.0,
    )

    result = engine.render(req)

    assert result == out
    assert out.read_bytes() == b"ID3fake"
    assert fake_heartlib["version"] == "3B"
    assert set(fake_heartlib["device"]) == {"mula", "codec"}
    assert fake_heartlib["dtype"]["mula"] == "bfloat16"
    assert fake_heartlib["dtype"]["codec"] == "float32"
    assert fake_heartlib["lazy_load"] is True
    # The upstream pipeline takes file *paths*, not strings.
    assert set(fake_heartlib["inputs"]) == {"lyrics", "tags"}
    assert fake_heartlib["lyrics_text"] == req.lyrics_text
    assert fake_heartlib["tags_text"] == req.tags
    kwargs = fake_heartlib["call_kwargs"]
    assert kwargs["max_audio_length_ms"] == 30_000
    assert kwargs["topk"] == 42
    assert kwargs["temperature"] == 0.8
    assert kwargs["cfg_scale"] == 2.0


def test_temporary_lyrics_file_is_gone_after_a_successful_render(
    fake_torch, fake_heartlib: dict, download_recorder: list[dict], tmp_path: Path
) -> None:
    fake_torch(cuda_available=True)
    req = engine.RenderRequest(
        lyrics_text="[Verse]\nhello\n",
        tags="piano",
        out_path=tmp_path / "song.mp3",
        seconds=10,
        temperature=0.9,
        topk=50,
        cfg_scale=1.5,
    )

    engine.render(req)

    assert not Path(fake_heartlib["inputs"]["lyrics"]).exists()
