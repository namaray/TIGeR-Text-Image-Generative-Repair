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
