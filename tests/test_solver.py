from pathlib import Path

import pytest

from tiger.schema import load_schema
from tiger.solver import apply_attr_patch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def schema():
    return load_schema(ROOT / "configs/schema.yaml")


ATTRS = {"color": "red", "material": "cotton", "pattern": "solid", "size": "M",
         "brand": "Red Wing Supply"}


def test_color_patch_updates_attrs_and_title(schema):
    res = apply_attr_patch("Red Wing Supply Red Cotton Shirt", "shirts", ATTRS,
                           {"color": "blue"}, schema)
    assert res.applied
    assert res.attrs["color"] == "blue"
    # brand untouched, colour word replaced with matching case (F10)
    assert res.title == "Red Wing Supply Blue Cotton Shirt"
    assert "color=blue" in res.canonical_text


def test_out_of_domain_patch_refused(schema):
    res = apply_attr_patch("t", "shirts", ATTRS, {"color": "crimson"}, schema)
    assert not res.applied
    assert res.escalate
    assert res.attrs["color"] == "red"  # untouched
    assert "out_of_domain" in res.refusal_reason


def test_unknown_field_refused(schema):
    res = apply_attr_patch("t", "shirts", ATTRS, {"colour_way": "blue"}, schema)
    assert not res.applied and res.escalate


def test_constraint_violation_refused(schema):
    # letter size on shoes violates the cross-field rule
    shoe_attrs = {"color": "red", "size": "9"}
    res = apply_attr_patch("t", "shoes", shoe_attrs, {"size": "M"}, schema)
    assert not res.applied and res.escalate
    assert "constraint_violation" in res.refusal_reason


def test_title_without_color_word_left_alone(schema):
    res = apply_attr_patch("Red Wing Supply Oxford Shirt", "shirts", ATTRS,
                           {"color": "green"}, schema)
    assert res.applied
    # "Red" only occurs inside the brand -> title must not change
    assert res.title == "Red Wing Supply Oxford Shirt"
    assert res.attrs["color"] == "green"


def test_alias_value_normalised(schema):
    res = apply_attr_patch("t", "shirts", ATTRS, {"color": "grey"}, schema)
    assert res.applied
    assert res.attrs["color"] == "gray"


def test_noop_patch_reports_no_change(schema):
    res = apply_attr_patch("Red Shirt", "shirts", ATTRS, {"color": "red"}, schema)
    assert res.applied
    assert res.changed_fields == []
