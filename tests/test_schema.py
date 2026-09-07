from pathlib import Path

import pytest

from tiger.schema import load_schema

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def schema():
    return load_schema(ROOT / "configs/schema.yaml")


def test_enum_domain_casing(schema):
    # regression: "XS" must validate even though normalisation lowercases
    assert schema.in_domain("size", "XS")
    assert schema.in_domain("size", "xs")
    assert not schema.in_domain("size", "XXL")


def test_alias_normalisation(schema):
    assert schema.normalize("color", "grey") == "gray"
    assert schema.in_domain("color", "grey")
    assert schema.in_domain("color", "multicolor")


def test_required_attribute(schema):
    v = schema.validate_attrs("shirts", {"material": "cotton", "pattern": "solid", "size": "M"})
    assert any(x.rule == "required" and x.field == "color" for x in v)


def test_cross_field_constraint(schema):
    # letter size on shoes violates shoes_have_numeric_sizes
    v = schema.validate_attrs("shoes", {"color": "red", "size": "M"})
    assert any(x.rule == "shoes_have_numeric_sizes" for x in v)
    ok = schema.validate_attrs("shoes", {"color": "red", "size": "9"})
    assert not any(x.rule == "shoes_have_numeric_sizes" for x in ok)


def test_valid_row_passes(schema):
    attrs = {"color": "red", "material": "cotton", "pattern": "solid", "size": "M", "brand": "Kestrel"}
    assert schema.is_valid("shirts", attrs)


def test_out_of_domain_patch_rejected(schema):
    assert not schema.in_domain("color", "crimson")
