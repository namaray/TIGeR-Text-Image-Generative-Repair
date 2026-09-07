"""Regression tests for the repair-ablation harness itself (findings A1, A4).

The ablation table is the measurement instrument for every repair-side claim.
Two defects made it report configurations it was not actually running:

  A1 -- "No Gamma Gate" wrote cfg["fusion"]["gamma"], a key nothing reads, so
        the row was a duplicate of Full System.
  A4 -- all five configs came from cfg.copy(), a shallow copy sharing nested
        dicts, so a per-config edit would leak into every other config.

These pin the wiring without running the pipeline.
"""

import copy

from tiger import arbiter as A


BASE_CFG = {
    "arbiter": {"gamma": 0.60, "dismiss_threshold": 0.80,
                "t2v_policy": {"allowed_categories": ["shirts"]}},
    "verify": {"max_passes": 2},
}

BASE_EV = {"row_id": "r1", "category": "shirts",
           "allowed_directions": ["V2T", "T2V", "HUMAN"],
           "probes": {}, "image_missing": False, "text_missing": False}


def _constant_model(target: str, conf: float) -> A.ArbiterModel:
    import numpy as np
    n = len(A.FEATURES)
    k = len(A.CLASSES)
    other = float(np.log((1 - conf) / (k - 1)))
    intercept = [float(np.log(conf)) if c == target else other for c in A.CLASSES]
    return A.ArbiterModel(A.FEATURES, A.CLASSES, [0.0] * n, [1.0] * n,
                          [[0.0] * n for _ in A.CLASSES], intercept)


def test_gamma_is_read_from_the_arbiter_section():
    """A1: the key the ablation writes must be the key route() reads."""
    low = _constant_model("E1", conf=0.40)          # under the default gamma

    gated = A.route(dict(BASE_EV), low, BASE_CFG)
    assert gated.error_type == "E4", "0.40 < gamma=0.60 should escalate"

    cfg_no_gamma = copy.deepcopy(BASE_CFG)
    cfg_no_gamma["arbiter"]["gamma"] = 0.0
    ungated = A.route(dict(BASE_EV), low, cfg_no_gamma)
    assert ungated.error_type == "E1", "gamma=0 must let the route through"
    assert ungated.direction == "V2T"


def test_writing_an_unread_key_does_not_disable_the_gate():
    """A1 regression: the exact shape of the original bug must stay dead."""
    cfg_bug = copy.deepcopy(BASE_CFG)
    cfg_bug.setdefault("fusion", {})["gamma"] = 0.0
    r = A.route(dict(BASE_EV), _constant_model("E1", conf=0.40), cfg_bug)
    assert r.error_type == "E4", (
        "cfg['fusion']['gamma'] must not affect routing -- if this passes as E1, "
        "the gate is being read from the wrong place again"
    )


def test_deepcopy_isolates_per_config_gamma_edits():
    """A4: editing one config must not mutate its siblings."""
    cfg_a = copy.deepcopy(BASE_CFG)
    cfg_b = copy.deepcopy(BASE_CFG)
    cfg_b["arbiter"]["gamma"] = 0.0

    assert cfg_a["arbiter"]["gamma"] == 0.60, "sibling config was mutated"
    assert BASE_CFG["arbiter"]["gamma"] == 0.60, "source config was mutated"


def test_shallow_copy_would_have_leaked():
    """A4: documents why deepcopy is required, not merely tidier.

    Uses a local source dict so the demonstration cannot contaminate other tests.
    """
    source = copy.deepcopy(BASE_CFG)
    shallow = source.copy()
    shallow["arbiter"]["gamma"] = 0.0
    assert source["arbiter"]["gamma"] == 0.0, (
        "shallow copy shares nested dicts -- this is the hazard deepcopy removes"
    )


# --------------------------------------------------------------------------
# A3 / A3b: the random-routing baseline
# --------------------------------------------------------------------------

def _dummy(seed: int = 42):
    from tiger.eval.repair_ablation import DummyArbiter
    n = len(A.FEATURES)
    return DummyArbiter(
        feature_names=A.FEATURES, classes=A.CLASSES,
        mean=[0.0] * n, scale=[1.0] * n,
        coef=[[0.0] * n for _ in A.CLASSES], intercept=[0.0] * len(A.CLASSES),
        seed=seed,
    )


def test_random_baseline_emits_only_real_classes():
    """A3: 'E4' is a gamma-gate state, not a predictable class."""
    p = _dummy().predict_proba(None)
    assert set(p) == set(A.CLASSES)
    assert "E4" not in p, "E4 would fall through route() and be relabelled E3"


def test_random_baseline_can_choose_clean():
    """A3: hardcoding CLEAN=0.0 made the baseline non-uniform over outcomes."""
    d = _dummy(seed=7)
    tops = {max((p := d.predict_proba(None)), key=p.get) for _ in range(400)}
    assert tops == set(A.CLASSES), f"baseline never selected some class: {tops}"


def test_random_baseline_is_reproducible():
    """A3b: the paper's 3.2% row must be reproducible from a recorded seed."""
    a = [_dummy(seed=11).predict_proba(None) for _ in range(5)]
    b = [_dummy(seed=11).predict_proba(None) for _ in range(5)]
    assert a == b, "same seed must replay identically"
    assert a != [_dummy(seed=12).predict_proba(None) for _ in range(5)]


def test_random_baseline_records_its_seed():
    assert _dummy(seed=99).training_meta["dummy_arbiter_seed"] == 99


def test_random_baseline_probabilities_are_normalised():
    p = _dummy().predict_proba(None)
    assert abs(sum(p.values()) - 1.0) < 1e-9
    assert all(v >= 0.0 for v in p.values())
