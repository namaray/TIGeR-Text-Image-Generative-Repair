import json
from pathlib import Path

import pandas as pd
import pytest

from tiger import text_views
from tiger.data import noise
from tiger.schema import load_schema

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def schema():
    return load_schema(ROOT / "configs/schema.yaml")


def make_clean_df(n_products: int = 40) -> pd.DataFrame:
    rows = []
    colors = ["red", "blue", "green", "black"]
    cats = ["shirts", "shoes", "bags", "hats"]
    for i in range(n_products):
        cat = cats[i % 4]
        attrs = {"color": colors[i % 4], "material": "cotton", "pattern": "solid",
                 "size": "M" if cat == "shirts" else "one-size", "brand": "Kestrel"}
        pid = f"{cat}_{i:03d}"
        rows.append({
            "row_id": pid, "product_id": pid, "category": cat,
            "title": f"Kestrel {attrs['color'].capitalize()} Cotton Thing",
            "attributes": json.dumps(attrs),
            "canonical_text": text_views.canonical_text("t", cat, attrs),
            "image_path": f"img_{pid}.jpg", "is_image_missing": False,
            "is_text_missing": False, "split": "report",
        })
    return pd.DataFrame(rows)


CFG = {
    "seed": 3, "copies_per_row": 1, "max_redraws": 20,
    "rates": {"swap_image": 0.2, "color_flip": 0.2, "attribute_drop": 0.05,
              "title_contradiction": 0.05, "mixed_swap_color": 0.05},
}


def test_no_noop_noise(schema):
    """F12 regression: every dirty row must actually differ from its original."""
    df = make_clean_df()
    res = noise.inject(df, schema, CFG)  # inject() itself asserts; also check audit
    assert res.meta["self_verified"]
    assert len(res.audit) == res.meta["rows_noisy"]
    assert res.audit["verified_changed"].all()


def test_swap_never_same_product(schema):
    df = make_clean_df()
    res = noise.inject(df, schema, CFG)
    swaps = res.audit[res.audit["label"].isin(["swap_image", "mixed"])] \
        if "label" in res.audit.columns else res.audit[res.audit["subtype"].str.contains("swap|mixed")]
    for _, r in swaps.iterrows():
        assert r["donor_product_id"] != r["product_id"]
        assert r["new_value"].split("|")[0] != r["old_value"].split("|")[0]


def test_color_flip_excludes_current(schema):
    df = make_clean_df()
    res = noise.inject(df, schema, CFG)
    flips = res.audit[res.audit["subtype"] == "color_flip"]
    assert len(flips) > 0
    for _, r in flips.iterrows():
        assert r["old_value"] != r["new_value"]
        assert r["new_value"] in schema.domain("color")


def test_ground_truth_log_complete(schema):
    df = make_clean_df()
    res = noise.inject(df, schema, CFG)
    dirty = res.df[res.df["noise_label"] != "clean"]
    assert set(dirty["row_id"]) == set(res.audit["row_id"])
    # every audit record names the mutated field and old/new values
    assert (res.audit["field"] != "").all()


def test_brand_safe_color_replacement():
    """F10: colour replacement must not corrupt brand names."""
    title = "Red Wing Supply Red Cotton Shirt"
    out = text_views.replace_color_word_safe(title, "red", "blue", {"Red Wing Supply"})
    assert out == "Red Wing Supply Blue Cotton Shirt"


def test_find_color_word_ignores_brand():
    assert text_views.find_color_word("Red Wing Supply Shirt", ["red", "blue"], {"Red Wing Supply"}) is None
    assert text_views.find_color_word("Red Wing Supply Blue Shirt", ["red", "blue"], {"Red Wing Supply"}) == "blue"
