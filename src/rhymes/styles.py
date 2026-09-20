"""Kid-friendly style presets and tag resolution.

Pure. The preset table is what lets someone pick a sound without learning
HeartMuLa's tag vocabulary. `DEFAULT_STYLE` being a member of this curated
table is the mechanical half of the child-safe-default prohibition; the
contents of the table itself carry a judgment review.
"""

from __future__ import annotations

# Comma-separated with no spaces around the commas -- the format the model
# expects. Insertion order is display order: `rhymes styles` must be stable.
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

    Raw `tags` wins when given; otherwise the named preset; otherwise the
    default. The CLI makes the two mutually exclusive, so this never merges.
    """
    if tags is not None:
        return tags
    if style is None:
        return PRESETS[DEFAULT_STYLE]
    return lookup(style)


def lookup(style: str) -> str:
    """Case- and whitespace-insensitive preset lookup."""
    try:
        return PRESETS[style.strip().lower()]
    except KeyError:
        raise StyleError(f"unknown style {style!r}. Valid styles: {', '.join(PRESETS)}") from None


def describe() -> list[tuple[str, str]]:
    """Preset names with their expansions, in stable declared order."""
    return list(PRESETS.items())
