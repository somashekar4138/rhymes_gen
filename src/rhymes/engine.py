"""The only module that touches heartlib, torch or the network.

Every heavy import happens inside a function body, never at module scope.
Three things depend on that: validation must be able to reject a file before a
model loads, the test suite must run on a machine with no CUDA and no heartlib,
and `rhymes --help` must be instant.
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


class EngineError(RuntimeError):
    """Anything that went wrong between a validated request and an audio file."""


def preflight(device: str | None = None) -> str:
    """Resolve the device to generate on, or explain why we cannot.

    torch is imported inside this function body, not at module scope. The whole
    test suite's ability to run on a CUDA-less Mac depends on that, as does
    SPEC R3's fail-fast promise and a sub-second `rhymes styles`.

    Nothing escapes this function unwrapped: SPEC R4 is specifically about the
    user never seeing a torch traceback. That includes torch not being
    installed at all, which is the default on any machine that skipped the
    `gpu` extra.
    """
    try:
        import torch
    except ImportError as exc:
        raise EngineError(
            "torch is not installed, so there is nothing to generate with. "
            "Generation needs a GPU: open notebooks/rhymes_colab.ipynb in Google "
            "Colab, set the runtime to a T4 GPU, and run it there. "
            f"({exc})"
        ) from exc

    if device is not None:
        try:
            resolved = str(torch.device(device))
        except Exception as exc:
            raise EngineError(f"could not use --device {device!r}: {exc}") from exc
        if not resolved.startswith("cuda"):
            print(
                f"rhymes: --device {resolved} is an unsupported escape hatch; "
                "generation is only supported on CUDA.",
                file=sys.stderr,
            )
        return resolved

    if not torch.cuda.is_available():
        raise EngineError(
            "no CUDA device found. Generation needs a GPU: open "
            "notebooks/rhymes_colab.ipynb in Google Colab, set the runtime to a "
            "T4 GPU, and run it there."
        )
    return "cuda"


# (repo_id, directory relative to the cache root). The directory names are
# load-bearing: `from_pretrained` looks for exactly these. Module constants,
# never assembled from user input.
CHECKPOINTS: list[tuple[str, str]] = [
    ("HeartMuLa/HeartMuLaGen", ""),
    ("HeartMuLa/HeartMuLa-oss-3B-happy-new-year", "HeartMuLa-oss-3B"),
    ("HeartMuLa/HeartCodec-oss-20260123", "HeartCodec-oss"),
]

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "rhymes" / "ckpt"


def cache_root(cache_dir: Path | None = None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    env = os.environ.get("RHYMES_CACHE_DIR")
    return Path(env) if env else DEFAULT_CACHE_DIR


def ensure_checkpoints(cache_dir: Path | None = None) -> Path:
    """Place the three HeartMuLa checkpoints (22.4 GB on a cold run).

    Deliberately implements no downloader, no resume loop, no checksum pass and
    no "have I already got this" short-circuit. `snapshot_download` already
    caches, already writes partial files under a temporary name and already
    resumes; a hand-rolled existence check is precisely how a truncated
    checkpoint gets treated as complete. Delegation is the mitigation.
    """
    from huggingface_hub import snapshot_download

    root = cache_root(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    for repo_id, relative in CHECKPOINTS:
        target = root / relative if relative else root
        snapshot_download(repo_id=repo_id, local_dir=str(target))
    return root


@dataclass(frozen=True)
class RenderRequest:
    lyrics_text: str
    tags: str
    out_path: Path
    seconds: int
    temperature: float
    topk: int
    cfg_scale: float
    device: str | None = None


def render(req: RenderRequest) -> Path:
    """Generate audio for `req` and place it at `req.out_path`.

    The temporary directory lives beside the target so the final `os.replace`
    is a same-filesystem rename, which is what makes it atomic. A failure or an
    interrupted Colab session therefore leaves nothing at the target path
    rather than a half-written mp3.
    """
    out_path = req.out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Ordering is the requirement, not an optimization: a user on a laptop
    # should be told to go to Colab in a second, not after 22.4 GB of download.
    preflight(req.device)
    ckpt_root = ensure_checkpoints()
    with tempfile.TemporaryDirectory(dir=out_path.parent) as work:
        workdir = Path(work)
        lyrics_path = workdir / "lyrics.txt"
        tags_path = workdir / "tags.txt"
        # The upstream pipeline takes file *paths*, not strings.
        lyrics_path.write_text(req.lyrics_text, encoding="utf-8")
        tags_path.write_text(req.tags, encoding="utf-8")
        tmp_out = workdir / "out.mp3"

        try:
            _generate(ckpt_root, lyrics_path, tags_path, tmp_out, req)
        except EngineError:
            raise
        except Exception as exc:
            raise EngineError(f"generation failed: {exc}") from exc

        if not tmp_out.exists():
            raise EngineError("generation produced no audio")
        os.replace(tmp_out, out_path)

    return out_path


def _generate(
    ckpt_root: Path | None,
    lyrics_path: Path,
    tags_path: Path,
    tmp_out: Path,
    req: RenderRequest,
) -> None:
    """The single seam the test suite mocks.

    heartlib and torch are imported inside this body. `lazy_load` is on because
    a free T4 has 16 GB while the 3B weights in bf16 plus the fp32 codec are
    ~14.5 GB before activations -- it is the pressure valve, not an option.
    """
    try:
        import torch
        from heartlib import HeartMuLaGenPipeline
    except ImportError as exc:
        raise EngineError(
            "heartlib is not installed, so there is no model to sing with. "
            'Install it with: pip install "rhymes[gpu] @ '
            f'git+https://github.com/somashekar4138/rhymes_gen" ({exc})'
        ) from exc

    device = torch.device(preflight(req.device))
    pipe = HeartMuLaGenPipeline.from_pretrained(
        ckpt_root,
        device={"mula": device, "codec": device},
        dtype={"mula": torch.bfloat16, "codec": torch.float32},
        version="3B",
        lazy_load=True,
    )
    pipe(
        # Upstream takes file *paths*, not strings, which is why `render`
        # materializes both into its temporary directory.
        {"lyrics": str(lyrics_path), "tags": str(tags_path)},
        max_audio_length_ms=req.seconds * 1000,
        save_path=str(tmp_out),
        topk=req.topk,
        temperature=req.temperature,
        cfg_scale=req.cfg_scale,
    )
