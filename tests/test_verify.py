"""Verify Eq. 27-29 acceptance-gate tests (no CLIP; synthetic scalars)."""

from pathlib import Path

import pytest

from tiger.schema import load_schema
from tiger.verify import VerifyCalibration, verify_repair

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def schema():
    return load_schema(ROOT / "configs/schema.yaml")


GOOD = {"color": "blue", "material": "cotton", "pattern": "solid", "size": "M", "brand": "Kestrel"}


def test_accept_when_all_gates_pass(schema):
    v = verify_repair("r", "shirts", GOOD, c_before=0.20, c_after=0.30,
                      tau=0.25, epsilon=0.03, schema=schema)
    assert v.accepted and v.reason == "accepted"


def test_reject_on_schema_eq27(schema):
    bad = dict(GOOD, color="crimson")  # out of Omega_color
    v = verify_repair("r", "shirts", bad, 0.20, 0.40, tau=0.25, epsilon=0.03, schema=schema)
    assert not v.accepted and not v.schema_ok
    assert "Eq27" in v.reason


def test_reject_on_threshold_eq28(schema):
    v = verify_repair("r", "shirts", GOOD, 0.10, 0.22, tau=0.25, epsilon=0.03, schema=schema)
    assert not v.accepted and not v.threshold_ok
    assert "Eq28" in v.reason


def test_reject_on_margin_eq29(schema):
    # c' clears tau but the improvement is below the rewording noise floor
    v = verify_repair("r", "shirts", GOOD, 0.28, 0.29, tau=0.25, epsilon=0.03, schema=schema)
    assert not v.accepted and not v.margin_ok
    assert "Eq29" in v.reason


def test_independent_verifier_can_reject(schema):
    v = verify_repair("r", "shirts", GOOD, 0.20, 0.30, tau=0.25, epsilon=0.03,
                      schema=schema, independent_ok=False)
    assert not v.accepted
    assert "independent_verifier" in v.reason


def test_independent_verifier_pass_recorded(schema):
    v = verify_repair("r", "shirts", GOOD, 0.20, 0.30, tau=0.25, epsilon=0.03,
                      schema=schema, independent_ok=True)
    assert v.accepted and v.independent_ok is True


def test_calibration_category_fallback():
    cal = VerifyCalibration(epsilon=0.05, epsilon_by_category={"shirts": 0.03})
    assert cal.eps_for("shirts") == 0.03
    assert cal.eps_for("unknown") == 0.05


def test_calibration_json_roundtrip():
    cal = VerifyCalibration(epsilon=0.04, epsilon_by_category={"bags": 0.02}, meta={"q": 0.95})
    cal2 = VerifyCalibration.from_json(cal.to_json())
    assert cal2.epsilon == 0.04 and cal2.eps_for("bags") == 0.02
