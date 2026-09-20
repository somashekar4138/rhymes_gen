"""argparse adapter. Owns exit codes and stderr formatting, nothing else.

Exit codes: 0 success, 1 runtime or validation failure (one stderr line, no
traceback), 2 usage error -- argparse's own convention, which is why the
range and mutual-exclusion checks are expressed as parser constructs rather
than hand-written branches. They get exit 2 for free that way.

`--style` is deliberately NOT argparse `choices=`: that would exit 2, and
SPEC R2 wants an unknown style to exit 1 with the valid names listed.

The import order in `_cmd_render` is the point of the whole design: validation
and style resolution both return before `rhymes.engine` is imported, so a
malformed lyrics file never pays for a model load.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rhymes import styles
from rhymes.lyrics import LyricsError, load_lyrics
from rhymes.styles import StyleError

DEFAULT_SECONDS = 60
MIN_SECONDS = 5
MAX_SECONDS = 300


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rhymes",
        description="Turn a nursery rhyme you wrote into a sung audio track.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("styles", help="list the available style presets")

    render = sub.add_parser("render", help="render a lyrics file to an mp3")
    render.add_argument("lyrics", type=Path, help="path to your lyrics .txt")
    render.add_argument(
        "-o", "--out", type=Path, default=None, help="output .mp3 path (default: beside the input)"
    )

    # Mutually exclusive by construction, so "both given" exits 2 for free.
    style_group = render.add_mutually_exclusive_group()
    style_group.add_argument("--style", default=None, help="named preset (see `rhymes styles`)")
    style_group.add_argument(
        "--tags", type=_nonempty_tags, default=None, help="raw comma-separated tag string"
    )

    render.add_argument(
        "--seconds",
        type=_seconds,
        default=DEFAULT_SECONDS,
        help=f"length in seconds ({MIN_SECONDS}-{MAX_SECONDS}, default {DEFAULT_SECONDS})",
    )
    render.add_argument("--temperature", type=_positive_float, default=0.9)
    render.add_argument("--topk", type=_positive_int, default=50)
    render.add_argument("--cfg-scale", dest="cfg_scale", type=_positive_float, default=1.5)
    render.add_argument(
        "--device", default=None, help="unsupported escape hatch, e.g. cpu or cuda:1"
    )
    return parser


def _nonempty_tags(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise argparse.ArgumentTypeError("--tags cannot be empty")
    return cleaned


def _seconds(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a whole number of seconds") from None
    if not MIN_SECONDS <= parsed <= MAX_SECONDS:
        raise argparse.ArgumentTypeError(
            f"--seconds must be between {MIN_SECONDS} and {MAX_SECONDS}, got {parsed}"
        )
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number") from None
    # `not parsed > 0`, not `parsed <= 0`: IEEE-754 makes every comparison with
    # NaN false, so `nan <= 0` is False and NaN would sail through as positive.
    if not parsed > 0:
        raise argparse.ArgumentTypeError(f"must be a number greater than 0, got {value!r}")
    return parsed


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a whole number") from None
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be greater than 0, got {parsed}")
    return parsed


MIN_PYTHON = (3, 10)
MAX_PYTHON = (3, 12)


def _check_interpreter(version: tuple[int, int]) -> None:
    """Refuse an unsupported interpreter, naming the actual cause.

    Takes the version as a parameter so it is testable without a second
    interpreter. A user who knows why can decide what to do; one who does not
    files an issue.
    """
    if MIN_PYTHON <= version <= MAX_PYTHON:
        return
    raise RuntimeError(
        f"Python {version[0]}.{version[1]} is not supported. rhymes needs "
        f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}-{MAX_PYTHON[0]}.{MAX_PYTHON[1]} because heartlib "
        "hard-pins numpy==2.0.2, which publishes no wheel for 3.13 or newer."
    )


def main(argv: list[str] | None = None) -> int:
    try:
        _check_interpreter(sys.version_info[:2])
    except RuntimeError as exc:
        return _fail(exc)

    args = build_parser().parse_args(argv)
    if args.command == "styles":
        return _cmd_styles()
    return _cmd_render(args)


def _cmd_styles() -> int:
    table = styles.describe()
    width = max(len(name) for name, _ in table)
    for name, tags in table:
        marker = "  (default)" if name == styles.DEFAULT_STYLE else ""
        print(f"{name.ljust(width)}  {tags}{marker}")
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    try:
        lyrics = load_lyrics(args.lyrics)
        tags = styles.resolve_tags(args.style, args.tags)
    except (LyricsError, StyleError) as exc:
        return _fail(exc)
    except OSError as exc:
        return _fail(f"cannot read {args.lyrics}: {exc.strerror or exc}")

    # Only now, once nothing cheap can still fail, does the heavy module load.
    from rhymes import engine

    request = engine.RenderRequest(
        lyrics_text=lyrics.to_text(),
        tags=tags,
        out_path=args.out or args.lyrics.with_suffix(".mp3"),
        seconds=args.seconds,
        temperature=args.temperature,
        topk=args.topk,
        cfg_scale=args.cfg_scale,
        device=args.device,
    )
    try:
        out = engine.render(request)
    except engine.EngineError as exc:
        return _fail(exc)

    print(out)
    return 0


def _fail(message: object) -> int:
    # One line, always. CUDA OOM and HfHubHTTPError response bodies are
    # routinely multi-line, and the contract is a single actionable line.
    lines = str(message).splitlines() or [""]
    print(f"rhymes: {lines[0]}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
