"""Ground-truth construction for the repair ablation (finding A5).

"Restoration Accuracy" was built from `audit[field == "color"]` only, so
material and pattern patches were committed but never scored, and image repairs
were not scored at all. These pin the widened truth builders.

The audit shapes below mirror tiger/data/noise.py exactly:
  color_flip / near_color_flip   -> field="color"
  material_flip                  -> field="material"
  attribute_drop                 -> field="color",          new_value=""
  title_contradiction            -> field="title"           (attrs stay correct)
  swap_image / missing_image     -> field="image_path"
  mixed_swap_color               -> field="image_path+color", old="{img}|{colour}"
"""

import json

import pandas as pd

from tiger.eval.repair_ablation import truth_from_audit, image_provenance


def _audit(rows):
    return pd.DataFrame(rows, columns=["row_id", "field", "old_value", "new_value"])


def test_scores_every_attribute_field_not_just_colour():
    attrs, _ = truth_from_audit(_audit([
        ("r1", "color", "red", "blue"),
        ("r2", "material", "cotton", "denim"),
        ("r3", "pattern", "solid", "striped"),
    ]))
    assert attrs == {"r1": {"color": "red"},
                     "r2": {"material": "cotton"},
                     "r3": {"pattern": "solid"}}


def test_attribute_drop_records_the_dropped_value():
    attrs, _ = truth_from_audit(_audit([("r1", "color", "green", "")]))
    assert attrs["r1"]["color"] == "green"


def test_title_contradiction_is_not_an_attribute_repair():
    """The injector leaves attrs correct, so there is nothing to restore."""
    attrs, images = truth_from_audit(_audit([("r1", "title", "red", "blue")]))
    assert attrs == {} and images == {}


def test_mixed_swap_color_yields_both_sides():
    attrs, images = truth_from_audit(_audit([
        ("r1", "image_path", "img/own.jpg", "img/donor.jpg"),
        ("r2", "image_path+color", "img/r2.jpg|purple", "img/donor.jpg|orange"),
    ]))
    assert images == {"r1": "img/own.jpg", "r2": "img/r2.jpg"}
    assert attrs["r2"]["color"] == "purple", "the colour half was previously lost"


def test_empty_audit_is_handled():
    assert truth_from_audit(pd.DataFrame()) == ({}, {})
    assert truth_from_audit(None) == ({}, {})


# --------------------------------------------------------------------------
# image provenance: what a given image actually depicts
# --------------------------------------------------------------------------

def _row(rid, cat, color, path):
    return {"row_id": rid, "category": cat, "image_path": path,
            "attributes": json.dumps({"color": color})}


def test_uncorrupted_row_vouches_for_its_own_image():
    df = pd.DataFrame([_row("r1", "shirts", "red", "img/r1.jpg")])
    assert image_provenance(df, {}, {}) == {"img/r1.jpg": ("shirts", "red")}


def test_swapped_row_vouches_for_its_original_not_the_donor_image():
    """r1 now carries the donor's image; only img/r1.jpg depicts r1."""
    df = pd.DataFrame([
        _row("r1", "shirts", "red", "img/donor.jpg"),
        _row("r2", "shoes", "black", "img/donor.jpg"),
    ])
    _, images = truth_from_audit(_audit([("r1", "image_path", "img/r1.jpg", "img/donor.jpg")]))
    d = image_provenance(df, {}, images)
    assert d["img/r1.jpg"] == ("shirts", "red")
    assert d["img/donor.jpg"] == ("shoes", "black"), "donor image belongs to r2"


def test_provenance_uses_the_true_colour_for_a_mixed_row():
    """r1's text colour is corrupted; its image still depicts the true colour."""
    df = pd.DataFrame([_row("r1", "bags", "orange", "img/donor.jpg")])
    attrs, images = truth_from_audit(
        _audit([("r1", "image_path+color", "img/r1.jpg|purple", "img/donor.jpg|orange")]))
    d = image_provenance(df, attrs, images)
    assert d["img/r1.jpg"] == ("bags", "purple"), "must use pre-corruption colour"


def test_rows_without_an_image_are_skipped():
    df = pd.DataFrame([_row("r1", "hats", "white", "")])
    assert image_provenance(df, {}, {}) == {}
