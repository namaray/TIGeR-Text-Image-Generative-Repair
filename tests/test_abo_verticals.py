"""ABO vertical definitions and per-category capping (Phase 1.1 / 1.4)."""

from tiger.data.import_abo import CATEGORY_MAP, VERTICALS, VERTICAL_A_FURNISHING
from tiger import text_views
from tiger.schema import load_schema

SCHEMA = load_schema("configs/schema.yaml")


def test_every_mapped_category_is_a_schema_category():
    """An unmapped category makes validate_attrs emit `known_category` on every row."""
    unknown = sorted(set(CATEGORY_MAP.values()) - set(SCHEMA.categories))
    assert not unknown, f"absent from schema.yaml categories: {unknown}"


def test_every_category_is_t2v_eligible():
    """A6: a category missing from the allowlist is force-escalated before
    schema or gamma is consulted, which is how the whole ABO cross-domain run
    ended up with image repair switched off."""
    import yaml
    cfg = yaml.safe_load(open("configs/tiger.yaml"))
    allowed = set(cfg["arbiter"]["t2v_policy"]["allowed_categories"])
    missing = sorted(set(CATEGORY_MAP.values()) - allowed)
    assert not missing, f"absent from t2v_policy.allowed_categories: {missing}"


def test_every_category_has_a_real_singular_noun():
    """D8: falling through to rstrip('s') yields 'a photo of a red light_fixture'."""
    for cat in CATEGORY_MAP.values():
        noun = text_views.singular(cat)
        assert "_" not in noun, f"{cat} -> {noun!r} still carries an underscore"
        assert noun and noun != cat.rstrip("s") or cat == noun


def test_verticals_partition_the_category_map():
    a, b = VERTICALS["furnishing"], VERTICALS["accessories"]
    assert not (a & b), "verticals must not overlap"
    assert a | b == set(CATEGORY_MAP.values())


def test_phone_cases_are_excluded():
    """64,853 listings, 44% of the catalogue -- it would swamp every other type."""
    assert "CELLULAR_PHONE_CASE" not in CATEGORY_MAP
