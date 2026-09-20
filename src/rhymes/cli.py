"""argparse adapter. Owns exit codes and stderr formatting, nothing else.

Exit codes: 0 success, 1 runtime or validation failure (one stderr line, no
traceback), 2 usage error (argparse's own convention).

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
    render.add_argument("-o", "--out", type=Path, default=None, help="output .mp3 path")

    # Mutually exclusive by construction, so "both given" exits 2 for free.
    style_group = render.add_mutually_exclusive_group()
    style_group.add_argument("--style", default=None, help="named preset (see `rhymes styles`)")
    style_group.add_argument("--tags", default=None, help="raw comma-separated tag string")

    render.add_argument("--seconds", type=_seconds, default=DEFAULT_SECONDS)
    render.add_argument("--temperature", type=_positive_float, default=0.9)
    render.add_argument("--topk", type=_positive_int, default=50)
    render.add_argument("--cfg-scale", dest="cfg_scale", type=_positive_float, default=1.5)
    render.add_argument("--device", default=None, help="unsupported escape hatch")
    return parser


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
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be greater than 0, got {parsed}")
    return parsed


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a whole number") from None
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be greater than 0, got {parsed}")
    return parsed


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "styles":
        return _cmd_styles()
    return _cmd_render(args)


def _cmd_styles() -> int:
    width = max(len(name) for name, _ in styles.describe())
    for name, tags in styles.describe():
        marker = " (default)" if name == styles.DEFAULT_STYLE else ""
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
    from rhymes.engine import EngineError, RenderRequest, render

    request = RenderRequest(
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
        out = render(request)
    except EngineError as exc:
        return _fail(exc)

    print(out)
    return 0


def _fail(message: object) -> int:
    print(f"rhymes: {message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
