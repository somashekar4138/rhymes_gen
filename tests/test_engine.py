"""The heartlib boundary: preflight, checkpoint bootstrap, the pipeline call.

Every case fakes torch (and later heartlib) by inserting a stub into
sys.modules for the duration of the test. heartlib and a real CUDA device are
both absent on the dev machine, and STANDARDS.md forbids reaching into a
heartlib internal to work around that -- the seam we mock is our own.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import RenderingStubs

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
    mod.HeartMuLaGenPipeline = FakePipeline  # type: ignore[attr-defined]
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


def test_torch_not_installed_at_all_is_reported_not_raised(
    monkeypatch: pytest.MonkeyPatch, valid_lyrics_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Regression: found by running the real CLI on the dev Mac.

    Every other test stubs torch INTO sys.modules, so none of them exercised
    the case where torch is simply not installed -- which is every machine
    without the `gpu` extra. There, `import torch` raises ModuleNotFoundError,
    which is not an EngineError, so it escaped the CLI seam as a traceback.
    SPEC R4 is specifically about the user never seeing one.
    """
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)

    rc = cli.main(["render", str(valid_lyrics_file)])

    assert rc == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "Colab" in err


def test_heartlib_not_installed_names_the_gpu_extra(
    fake_torch, download_recorder: list[dict], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Same defect class as the torch case: the render wrapper already stops the
    traceback, but 'generation failed: No module named heartlib' does not tell
    anyone what to do about it."""
    import sys

    fake_torch(cuda_available=True)
    monkeypatch.setitem(sys.modules, "heartlib", None)

    req = engine.RenderRequest(
        lyrics_text="[Verse]\nhello\n",
        tags="piano",
        out_path=tmp_path / "out.mp3",
        seconds=10,
        temperature=0.9,
        topk=50,
        cfg_scale=1.5,
    )

    with pytest.raises(engine.EngineError) as exc:
        engine.render(req)

    assert "heartlib" in str(exc.value)
    assert "gpu" in str(exc.value)


# --- Stage 9 fixes: failures that reached the user as tracebacks ------------


def test_output_path_that_is_a_directory_exits_1_not_a_traceback(
    rendering_stubs: RenderingStubs,
    valid_lyrics_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """R-1: os.replace sat outside render's try, so IsADirectoryError escaped
    the CLI seam as a traceback."""
    target = tmp_path / "a_directory"
    target.mkdir()

    rc = cli.main(["render", str(valid_lyrics_file), "-o", str(target)])

    err = capsys.readouterr().err
    assert rc == 1
    assert "Traceback" not in err
    assert len(err.strip().splitlines()) == 1


def test_checkpoint_download_failure_exits_1_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
    fake_torch,
    valid_lyrics_file: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """R-1: a 22.4 GB download on a flaky Colab session raises OSError, which
    is not an EngineError."""
    fake_torch(cuda_available=True)

    def out_of_space(cache_dir: Path | None = None) -> Path:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(engine, "ensure_checkpoints", out_of_space, raising=True)

    rc = cli.main(["render", str(valid_lyrics_file)])

    err = capsys.readouterr().err
    assert rc == 1
    assert "Traceback" not in err


def test_a_multiline_failure_is_collapsed_to_one_stderr_line(
    rendering_stubs: RenderingStubs,
    valid_lyrics_file: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """R-1b: CUDA OOM and HfHubHTTPError bodies are routinely multi-line."""

    def boom(ckpt_root, lyrics_path, tags_path, tmp_out, req, device):
        raise RuntimeError("CUDA out of memory.\nTried to allocate 2.00 GiB\nSee docs.")

    rendering_stubs.set_generate(boom)

    rc = cli.main(["render", str(valid_lyrics_file)])

    err = capsys.readouterr().err
    assert rc == 1
    assert len(err.strip().splitlines()) == 1


def test_preflight_runs_once_per_render(
    monkeypatch: pytest.MonkeyPatch,
    fake_torch,
    download_recorder: list[dict],
    fake_heartlib: dict,
    tmp_path: Path,
) -> None:
    """R-3: render and _generate each resolved the device, so a --device cpu
    run printed the unsupported-escape-hatch warning twice."""
    fake_torch(cuda_available=True)
    calls: list[object] = []
    real = engine.preflight

    def counting(device: str | None = None) -> str:
        calls.append(device)
        return real(device)

    monkeypatch.setattr(engine, "preflight", counting, raising=True)

    engine.render(
        engine.RenderRequest(
            lyrics_text="[Verse]\nhello\n",
            tags="piano",
            out_path=tmp_path / "s.mp3",
            seconds=10,
            temperature=0.9,
            topk=50,
            cfg_scale=1.5,
        )
    )

    assert len(calls) == 1


@pytest.mark.parametrize(
    "device,expected_lazy",
    [("cuda", True), ("cuda:1", True), ("cpu", False), ("mps", False)],
)
def test_lazy_load_is_off_on_non_cuda_devices(
    fake_torch,
    fake_heartlib: dict,
    download_recorder: list[dict],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    expected_lazy: bool,
) -> None:
    """heartlib's `_unload()` calls torch.cuda.memory_allocated/empty_cache
    unconditionally, but only when lazy_load is True. On a non-CUDA device that
    path raises, so the --device escape hatch could never have worked while we
    passed lazy_load=True -- it advertised something that dies in heartlib.

    lazy_load is the VRAM pressure valve on a 16 GB T4, so it stays on there.
    """
    fake_torch(cuda_available=True)
    monkeypatch.setattr(engine, "preflight", lambda d=None: device, raising=True)

    engine.render(
        engine.RenderRequest(
            lyrics_text="[Verse]\nhello\n",
            tags="piano",
            out_path=tmp_path / "s.mp3",
            seconds=10,
            temperature=0.9,
            topk=50,
            cfg_scale=1.5,
            device=device,
        )
    )

    assert fake_heartlib["lazy_load"] is expected_lazy
