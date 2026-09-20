"""Shared fixtures.

The single mock seam the rendering tests hang off. It patches three names on
`rhymes.engine`, not one: plan 02 inserts `preflight` and `ensure_checkpoints`
ahead of `_generate` inside `render`, and on a machine with no CUDA `preflight`
raises while `ensure_checkpoints` would reach for a 22.4 GB download. A fixture
that patched only `_generate` would go red the moment those land.

Deliberately NOT autouse: this file sits at the suite root, and stubbing
`preflight` everywhere would pull it out from under `test_engine.py`, which
needs the real one.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, field
from pathlib import Path

import pytest


@dataclass
class RenderingStubs:
    """Handle onto the patched seam, so a test can swap behaviour without
    reaching around the fixture."""

    calls: list[dict[str, object]] = field(default_factory=list)
    generate: object = None

    def set_generate(self, fn: object) -> None:
        self.generate = fn


@pytest.fixture
def rendering_stubs(monkeypatch: pytest.MonkeyPatch) -> RenderingStubs:
    from rhymes import engine

    handle = RenderingStubs()

    def default_generate(
        ckpt_root: Path, lyrics_path: Path, tags_path: Path, tmp_out: Path, req: object
    ) -> None:
        handle.calls.append(
            {
                "ckpt_root": ckpt_root,
                "lyrics_path": lyrics_path,
                "tags_path": tags_path,
                "tmp_out": tmp_out,
                "req": req,
            }
        )
        tmp_out.write_bytes(b"ID3")

    handle.set_generate(default_generate)

    def dispatch(*args: object, **kwargs: object) -> None:
        fn = handle.generate
        assert callable(fn)
        return fn(*args, **kwargs)

    monkeypatch.setattr(engine, "_generate", dispatch, raising=True)
    # raising=True now that both names exist. raising=False was right while
    # plan 01 had not created them, but it never tightens on its own: leave it
    # and a later rename silently stubs a dead attribute while `render` calls
    # the real function and reaches for a 22.4 GB download.
    monkeypatch.setattr(engine, "preflight", lambda device=None: "cuda", raising=True)
    monkeypatch.setattr(
        engine,
        "ensure_checkpoints",
        lambda cache_dir=None: Path("/nonexistent/ckpt"),
        raising=True,
    )
    return handle


@pytest.fixture
def valid_lyrics_file(tmp_path: Path) -> Path:
    p = tmp_path / "rhyme.txt"
    p.write_text("[Verse]\nTwinkle twinkle little star\n", encoding="utf-8")
    return p


@pytest.fixture
def fake_torch(monkeypatch: pytest.MonkeyPatch):
    """Insert a stub `torch` into sys.modules.

    heartlib and a real CUDA device are both absent on the dev machine, and
    STANDARDS.md forbids reaching into a heartlib internal to work around that.
    """

    def _make(cuda_available: bool = False, device_raises: bool = False):
        mod = types.ModuleType("torch")

        class _Device:
            def __init__(self, spec: str) -> None:
                if device_raises:
                    raise RuntimeError(f"Expected one of cpu, cuda ... device string: {spec}")
                self.spec = spec

            def __str__(self) -> str:
                return self.spec

        mod.device = _Device  # type: ignore[attr-defined]
        mod.cuda = types.SimpleNamespace(is_available=lambda: cuda_available)  # type: ignore[attr-defined]
        mod.bfloat16 = "bfloat16"  # type: ignore[attr-defined]
        mod.float32 = "float32"  # type: ignore[attr-defined]
        monkeypatch.setitem(__import__("sys").modules, "torch", mod)
        return mod

    return _make
