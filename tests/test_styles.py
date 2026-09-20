"""Style presets and tag resolution."""

from __future__ import annotations

import pytest

from rhymes.styles import DEFAULT_STYLE, PRESETS, StyleError, describe, resolve_tags


def test_there_are_at_least_three_presets() -> None:
    assert len(PRESETS) >= 3


def test_preset_order_is_stable() -> None:
    assert list(PRESETS) == list(PRESETS)
    assert [name for name, _ in describe()] == list(PRESETS)


def test_default_style_is_a_member_of_the_curated_table() -> None:
    """Prohibition: MUST NOT ship a default style preset that is not
    child-appropriate. Membership of the kid-safe table is the mechanical half;
    the table's contents carry a judgment review."""
    assert DEFAULT_STYLE in PRESETS


@pytest.mark.parametrize("name", list(PRESETS))
def test_every_expansion_is_well_formed_for_the_model(name: str) -> None:
    tags = PRESETS[name]

    assert tags
    assert ", " not in tags
    assert " ," not in tags
    assert not tags.startswith(",")
    assert not tags.endswith(",")
    assert all(part.strip() for part in tags.split(","))


@pytest.mark.parametrize("spelling", ["lullaby", "LULLABY", "  Lullaby  "])
def test_lookup_is_case_and_whitespace_insensitive(spelling: str) -> None:
    assert resolve_tags(spelling, None) == PRESETS["lullaby"]


def test_unknown_style_lists_the_valid_names() -> None:
    with pytest.raises(StyleError) as exc:
        resolve_tags("deathmetal", None)

    for name in PRESETS:
        assert name in str(exc.value)


def test_no_arguments_gives_the_default_preset() -> None:
    assert resolve_tags(None, None) == PRESETS[DEFAULT_STYLE]


def test_raw_tags_are_passed_through() -> None:
    assert resolve_tags(None, "piano,gentle") == "piano,gentle"
