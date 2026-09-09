"""ABO free-text attribute values -> Omega_j (Phase 1.3).

ABO values are seller-written, not a controlled vocabulary: 2,147 distinct
colour strings and 608 material strings across the two furnishing verticals,
against a 12-value colour domain. Imported raw, every value is out-of-domain,
schema validation rejects the row, and it escalates before any repair is
attempted -- H10's failure mode one layer earlier.
"""

import pytest

from tiger.data.abo_vocab import normalize_color, normalize_material
from tiger.schema import load_schema

SCHEMA = load_schema("configs/schema.yaml")
CD = {SCHEMA.normalize("color", v) for v in SCHEMA.domain("color")}
MD = {SCHEMA.normalize("material", v) for v in SCHEMA.domain("material")}


@pytest.mark.parametrize("raw,expected", [
    ("black", "black"), ("grey", "gray"),
    ("light grey", "gray"), ("dark bronze", "brown"), ("matte black", "black"),
    ("blanco", "white"), ("negro", "black"), ("silber", "gray"), ("ホワイト", "white"),
    ("silver", "gray"), ("rose gold", "pink"), ("brushed nickel", "gray"),
    ("navy", "blue"), ("charcoal", "gray"), ("cream", "white"),
    ("multicolor", "multicolour"), ("multi", "multicolour"),
    ("platinum plated silver", "gray"),
])
def test_colour_resolves(raw, expected):
    assert normalize_color(raw, CD) == expected


@pytest.mark.parametrize("raw,expected", [
    ("metal", "metal"), ("metall", "metal"), ("aluminum", "metal"),
    ("faux leather", "leather"), ("engineered wood", "wood"),
    ("stone", "stone"), ("marble", "stone"),
    ("velvet", "fabric"), ("tela", "fabric"), ("polypropylene", "plastic"),
])
def test_material_resolves(raw, expected):
    assert normalize_material(raw, MD) == expected


@pytest.mark.parametrize("raw", [
    "no aplica", "not applicable", "不适用", "other", "fine-other-material",
    "n/a", "", None, "   ", "nicht zutreffend",
])
def test_non_values_return_none_not_a_guess(raw):
    """An unresolvable value must drop the attribute, never fabricate one.

    Mapping "no aplica" to a plausible colour would invent ground truth and
    silently inflate restoration accuracy.
    """
    assert normalize_color(raw, CD) is None
    assert normalize_material(raw, MD) is None


def test_unknown_tail_values_return_none():
    for raw in ("aquamarine-ish", "custom print 47", "zzzz"):
        assert normalize_color(raw, CD) is None


def test_every_resolved_value_is_in_domain():
    """Whatever comes back must validate, or the import writes invalid rows."""
    samples = ["light grey", "rose gold", "blanco", "matte black", "navy",
               "platinum plated silver", "charcoal", "multi"]
    for raw in samples:
        v = normalize_color(raw, CD)
        assert v is not None and SCHEMA.in_domain("color", v), raw
