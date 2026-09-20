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
    user never seeing a torch traceback.
    """
    import torch

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


def ensure_checkpoints(cache_dir: Path | None = None) -> Path:
    """Place the three HeartMuLa checkpoints. Filled in by plan 02 task 2."""
    raise EngineError("checkpoint bootstrap not yet wired")


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
    """The single seam the test suite mocks. Filled in by plan 02."""
    raise EngineError("not yet wired")
