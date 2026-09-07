"""Alias surface forms must not disagree with themselves (findings D5, D6, D7).

silver/gold/beige/navy/teal were listed as colour `values` AND aliased onto
other values. schema.domain() therefore returned semantic duplicates, and every
raw-vs-normalised comparison in the codebase quietly broke.
"""

import pytest

from tiger import solver, text_views
from tiger.schema import Schema, load_schema, _assert_domains_disjoint_from_aliases
from tiger.sieve import _title_color

SCHEMA = load_schema("configs/schema.yaml")
ABO_FORMS = ["silver", "gold", "beige", "navy", "teal"]


# --------------------------------------------------------------------------
# D5: the domain and the alias keys must be disjoint
# --------------------------------------------------------------------------

def test_domain_holds_no_semantic_duplicates():
    dom = SCHEMA.domain("color")
    assert len(dom) == len({SCHEMA.normalize("color", v) for v in dom}), \
        "two domain entries normalise to the same value; probes would score them as rivals"


def test_abo_forms_still_resolve_without_being_domain_members():
    for form in ABO_FORMS:
        assert form not in SCHEMA.domain("color")
        assert SCHEMA.in_domain("color", form), f"{form} must still be accepted in text"
        assert SCHEMA.normalize("color", form) in SCHEMA.domain("color")


def test_surface_forms_cover_domain_and_aliases_longest_first():
    forms = SCHEMA.surface_forms("color")
    assert set(SCHEMA.domain("color")) <= set(forms)
    assert set(ABO_FORMS) <= set(forms)
    assert forms == sorted(forms, key=lambda v: (-len(v), v))


def test_load_refuses_a_schema_that_reintroduces_the_clash():
    with pytest.raises(ValueError, match="both `values` and `aliases`"):
        _assert_domains_disjoint_from_aliases(
            {"color": {"type": "enum", "values": ["gray", "silver"],
                       "aliases": {"silver": "gray"}}})


def test_synthgen_can_render_every_domain_colour():
    """The clash made `cli synthgen` KeyError on ~a third of products."""
    from tiger.data.synthgen import COLOR_RGB
    missing = [c for c in SCHEMA.domain("color")
               if c != "multicolour" and c not in COLOR_RGB]
    assert missing == [], f"synthgen cannot render {missing}"


# --------------------------------------------------------------------------
# D6: the independent verifier compared a raw prediction to a normalised value
# --------------------------------------------------------------------------

class _FakeEncoder:
    """Encoder whose image looks most like `winner`.

    check_v2t builds one prompt-ensemble embedding per domain value and takes the
    argmax against the image, so the double must return a DIFFERENT vector per
    value. Two dimensions: [1, 0] for prompts naming the winner, [0, 1] for the
    rest; the image is [1, 0], so only the winner scores.
    """

    def __init__(self, winner: str):
        self.winner = winner

    def encode_images(self, paths):
        import numpy as np
        return np.tile([1.0, 0.0], (len(paths), 1)), [True] * len(paths)

    def encode_texts(self, texts):
        import re
        import numpy as np
        hit = [bool(re.search(rf"\b{re.escape(self.winner)}\b", t.lower())) for t in texts]
        return np.array([[1.0, 0.0] if h else [0.0, 1.0] for h in hit], dtype=float)

    def save_cache(self):
        pass


# A schema whose DOMAIN carries a non-canonical surface form. D5 removed these
# from configs/schema.yaml, so against the shipped schema `pred` is always
# already canonical and the D6 comparison cannot be observed to matter. This
# fixture is what makes the D6 assertions non-vacuous: it is the shape the
# repository was in, and the shape any future alias edit could restore.
ALIASED_SCHEMA = Schema(
    attributes={"color": {"type": "enum",
                          "values": ["gray", "silver", "red"],
                          "aliases": {"silver": "gray"}}},
    categories=["shirts"],
)


def test_the_fake_encoder_actually_discriminates():
    """Guard the double itself: a uniform encoder makes every assertion below vacuous."""
    from tiger.verify import IndependentVerifier
    iv = IndependentVerifier(_FakeEncoder("gray"), SCHEMA)
    assert iv.check_v2t("img.jpg", "shirts", "color", "gray") is True
    assert iv.check_v2t("img.jpg", "shirts", "color", "red") is False


def test_independent_verifier_accepts_an_alias_equivalent_prediction():
    """D6: the encoder predicts the raw form "silver"; the repair wrote "gray".

    Both denote the same colour. Comparing the raw prediction against the
    normalised value vetoed this correct repair.
    """
    from tiger.verify import IndependentVerifier
    iv = IndependentVerifier(_FakeEncoder("silver"), ALIASED_SCHEMA)
    assert iv.check_v2t("img.jpg", "shirts", "color", "gray") is True


def test_independent_verifier_still_vetoes_a_genuine_disagreement():
    from tiger.verify import IndependentVerifier
    iv = IndependentVerifier(_FakeEncoder("silver"), ALIASED_SCHEMA)
    assert iv.check_v2t("img.jpg", "shirts", "color", "red") is False


# --------------------------------------------------------------------------
# D7: _title_color returned a raw match, compared against a normalised value
# --------------------------------------------------------------------------

def test_title_colour_is_normalised():
    assert _title_color("Navy Cotton Shirt", SCHEMA, "") == "blue"
    assert _title_color("Gold Silk Scarf", SCHEMA, "") == "yellow"


def test_alias_title_matching_declared_alias_is_not_a_contradiction():
    """"Navy Shirt" + color=navy must agree: both are blue."""
    title_color = _title_color("Navy Shirt", SCHEMA, "")
    declared = SCHEMA.normalize("color", "navy")
    assert title_color == declared


def test_a_real_contradiction_is_still_caught_through_an_alias():
    title_color = _title_color("Navy Shirt", SCHEMA, "")
    declared = SCHEMA.normalize("color", "red")
    assert title_color and declared and title_color != declared


def test_brand_names_are_still_masked():
    assert _title_color("Navy Republic Red Shirt", SCHEMA, "Navy Republic") == "red"


# --------------------------------------------------------------------------
# D5 follow-through: the title rewriter must see surface forms too
# --------------------------------------------------------------------------

def test_patching_rewrites_an_aliased_colour_word_in_the_title():
    res = solver.apply_attr_patch(
        "Navy Cotton Shirt", "shirts",
        {"color": "navy", "material": "cotton", "size": "M", "brand": "Acme"},
        {"color": "red"}, SCHEMA)
    assert res.applied
    assert "navy" not in res.title.lower(), f"stale colour word left in {res.title!r}"
    assert "Red" in res.title
