"""Kid-friendly style presets and tag resolution.

Pure. The preset table is what lets someone pick a sound without learning
HeartMuLa's tag vocabulary, and `DEFAULT_STYLE` being a member of it is the
mechanical half of the child-safe-default prohibition.
"""

from __future__ import annotations

# Comma-separated, no spaces around the commas - the format the model expects.
# Insertion order is the display order; `rhymes styles` must be stable.
PRESETS: dict[str, str] = {
    "nursery": "nursery rhyme,children,acoustic guitar,cheerful,simple melody,gentle",
    "lullaby": "lullaby,children,music box,soft,slow,soothing,gentle",
    "singalong": "children,sing-along,upbeat,piano,playful,clapping,major key",
    "marching": "children,marching band,brass,steady beat,playful,bright",
}

DEFAULT_STYLE = "nursery"


class StyleError(ValueError):
    """A style the preset table does not know."""


def resolve_tags(style: str | None, tags: str | None) -> str:
    """Return the tag string to generate with.

    `tags` wins when given; otherwise the named preset; otherwise the default.
    The CLI makes the two mutually exclusive, so this never has to merge them.
    """
    if tags is not None:
        cleaned = tags.strip()
        if not cleaned:
            raise StyleError("--tags was empty")
        return cleaned
    if style is None:
        return PRESETS[DEFAULT_STYLE]
    return lookup(style)


def lookup(style: str) -> str:
    try:
        return PRESETS[style.strip().lower()]
    except KeyError:
        raise StyleError(f"unknown style {style!r}. Valid styles: {', '.join(PRESETS)}") from None


def describe() -> list[tuple[str, str]]:
    return list(PRESETS.items())
