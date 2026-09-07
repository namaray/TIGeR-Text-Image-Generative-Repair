"""apply_thresholds must honour per-signal margins and quarantine (roadmap 3.4/3.6)."""

import numpy as np
import pandas as pd

from tiger import sieve
from tiger.fusion import FusionConfig
from tiger.schema import Schema

SCHEMA = Schema(attributes={"color": {"type": "enum", "values": ["red", "blue"]}},
                categories=["shirts"], constraints=[])
CFG = {"sieve": {"quantile_q": 0.05, "global_quantile_q": 0.05,
                 "min_products_per_category": 8,
                 "probes": {"fields": ["color"], "z_margin": 2.0}}}


def _signals(n=40):
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "row_id": [f"shirts_{i}" for i in range(n)],
        "product_id": [f"shirts_{i}" for i in range(n)],
        "category": "shirts",
        "sim_full": rng.normal(0.30, 0.02, n),
        "probe_color_margin": rng.normal(0.02, 0.01, n),
        "is_image_missing": False, "is_text_missing": False,
    })


def test_quarantined_probe_does_not_flag():
    clean = _signals()
    thr = sieve.calibrate(clean, CFG, SCHEMA)
    test = _signals(6)
    test.loc[0, "probe_color_margin"] = -0.20  # would fire strongly

    normal = sieve.apply_thresholds(test, thr)
    assert bool(normal.loc[0, "flag_probe_color"])

    fusion = FusionConfig(per_signal={"flag_probe_color": {"z_margin": 2.0, "quarantined": True}})
    quar = sieve.apply_thresholds(test, thr, fusion=fusion)
    assert not bool(quar.loc[0, "flag_probe_color"])
    assert not bool(quar.loc[0, "flagged"])


def test_tighter_margin_suppresses_borderline():
    clean = _signals()
    thr = sieve.calibrate(clean, CFG, SCHEMA)
    test = _signals(6)
    # margin just past the z=2 boundary but not past z=4
    mean = thr.probes["shirts"]["color"]["mean"] if "color" in thr.probes.get("shirts", {}) \
        else thr.probes_global["color"]["mean"]
    std = thr.probes_global["color"]["std"]
    test.loc[0, "probe_color_margin"] = mean - 2.5 * std

    base = sieve.apply_thresholds(test, thr)  # z_margin 2.0 -> fires
    assert bool(base.loc[0, "flag_probe_color"])

    fusion = FusionConfig(per_signal={"flag_probe_color": {"z_margin": 4.0, "quarantined": False}})
    tight = sieve.apply_thresholds(test, thr, fusion=fusion)
    assert not bool(tight.loc[0, "flag_probe_color"])
