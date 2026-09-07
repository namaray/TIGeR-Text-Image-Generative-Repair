"""Arbiter unit tests: gamma gate, dismiss guard, policy gate, model round-trip."""

import numpy as np

from tiger import arbiter as A

CFG = {"arbiter": {"gamma": 0.60, "dismiss_threshold": 0.80,
                   "t2v_policy": {"allowed_categories": ["shirts"]}}}


def constant_model(target: str, conf: float = 0.95) -> A.ArbiterModel:
    """Model whose softmax always favours `target` with ~conf probability."""
    n = len(A.FEATURES)
    coef = [[0.0] * n for _ in A.CLASSES]
    k = len(A.CLASSES)
    other = np.log((1 - conf) / (k - 1))
    intercept = [np.log(conf) if c == target else float(other) for c in A.CLASSES]
    return A.ArbiterModel(A.FEATURES, A.CLASSES, [0.0] * n, [1.0] * n, coef, intercept)


BASE_EV = {"row_id": "r1", "category": "shirts", "allowed_directions": ["V2T", "T2V", "HUMAN"],
           "probes": {}, "image_missing": False, "text_missing": False}


def test_gamma_gate_escalates_to_e4():
    m = constant_model("E1", conf=0.40)  # below gamma
    r = A.route(dict(BASE_EV), m, CFG)
    assert r.error_type == "E4"
    assert r.action == "human_review"


def test_e1_routes_v2t():
    r = A.route(dict(BASE_EV), constant_model("E1"), CFG)
    assert (r.error_type, r.direction, r.action) == ("E1", "V2T", "v2t_patch")


def test_e2_routes_t2v():
    r = A.route(dict(BASE_EV), constant_model("E2"), CFG)
    assert (r.direction, r.action) == ("T2V", "t2v_replace_image")


def test_e3_routes_image_first_then_text():
    r = A.route(dict(BASE_EV), constant_model("E3"), CFG)
    assert r.direction == "BOTH"
    assert r.action == "t2v_replace_image_then_v2t"


def test_policy_gate_blocks_t2v_for_disallowed_category():
    ev = dict(BASE_EV, category="shoes")  # policy only allows shirts
    r = A.route(ev, constant_model("E2"), CFG)
    assert r.action == "human_review"


def test_confident_clean_dismissed():
    r = A.route(dict(BASE_EV), constant_model("CLEAN", 0.95), CFG)
    assert r.action == "dismiss"


def test_dismiss_guard_on_contrary_probe():
    ev = dict(BASE_EV, probes={"color": {"z": -3.5, "margin": -0.05, "pred": "blue"}})
    r = A.route(ev, constant_model("CLEAN", 0.95), CFG)
    assert r.action == "human_review"  # strong contrary signal blocks dismissal


def test_missing_image_never_v2t():
    ev = dict(BASE_EV, image_missing=True, allowed_directions=["T2V_ACQUIRE", "HUMAN"])
    r = A.route(ev, constant_model("E1"), CFG)
    assert r.direction == "T2V_ACQUIRE"
    assert r.action == "acquire_image"


def test_model_json_roundtrip():
    m = constant_model("E2")
    m2 = A.ArbiterModel.from_json(m.to_json())
    x = np.zeros(len(A.FEATURES))
    assert m.predict_proba(x) == m2.predict_proba(x)


def test_probs_sum_to_one():
    m = constant_model("E1")
    p = m.predict_proba(np.random.default_rng(0).normal(size=len(A.FEATURES)))
    assert abs(sum(p.values()) - 1.0) < 1e-9
