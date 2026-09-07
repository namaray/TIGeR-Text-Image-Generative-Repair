"""Threshold calibration/application tests -- no CLIP involved (synthetic signals)."""

import numpy as np
import pandas as pd
import pytest

from tiger import sieve
from tiger.schema import Schema

SCHEMA = Schema(
    attributes={"color": {"type": "enum", "values": ["red", "blue"]}},
    categories=["shirts", "rare"],
    constraints=[],
)

CFG = {"sieve": {"quantile_q": 0.05, "global_quantile_q": 0.05,
                 "min_products_per_category": 8,
                 "probes": {"fields": ["color"], "z_margin": 2.0, "min_margin_raw": 0.005}}}


def make_signals(n=50, cat="shirts", sim_mean=0.30, sim_std=0.02, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "row_id": [f"{cat}_{i}" for i in range(n)],
        "product_id": [f"{cat}_{i}" for i in range(n)],
        "category": cat,
        "sim_full": rng.normal(sim_mean, sim_std, n),
        "probe_color_margin": rng.normal(0.02, 0.01, n),
        "is_image_missing": False,
        "is_text_missing": False,
    })


def test_calibrate_on_clean_only_category_stats():
    df = make_signals()
    thr = sieve.calibrate(df, CFG, SCHEMA)
    assert "shirts" in thr.sim
    assert thr.sim["shirts"]["n_products"] == 50
    assert thr.sim["shirts"]["std"] > 0


def test_small_category_falls_back_to_global():
    """F4: categories below the unique-product floor must use the global threshold."""
    big = make_signals(50, "shirts")
    small = make_signals(3, "rare", sim_mean=0.10)
    thr = sieve.calibrate(pd.concat([big, small], ignore_index=True), CFG, SCHEMA)
    assert "rare" not in thr.sim
    flagged = sieve.apply_thresholds(small, thr)
    assert (flagged["threshold_source"] == "global").all()


def test_duplicates_do_not_inflate_sample_size():
    """F4: 15x duplicated products must count once for calibration."""
    df = make_signals(4, "rare")
    dup = pd.concat([df] * 15, ignore_index=True)  # same product_ids repeated
    thr = sieve.calibrate(pd.concat([make_signals(50, "shirts"), dup], ignore_index=True), CFG, SCHEMA)
    assert "rare" not in thr.sim  # 4 unique products < 8, despite 60 rows


def test_locked_thresholds_flag_low_sim():
    clean = make_signals()
    thr = sieve.calibrate(clean, CFG, SCHEMA)
    test = make_signals(10, seed=1)
    test.loc[0, "sim_full"] = 0.10  # far below tau
    flagged = sieve.apply_thresholds(test, thr)
    assert bool(flagged.loc[0, "flag_low_sim"])
    assert flagged.loc[0, "flag_reason"] == "low_sim"


def test_probe_z_margin_flags():
    clean = make_signals()
    thr = sieve.calibrate(clean, CFG, SCHEMA)
    test = make_signals(5, seed=2)
    test.loc[1, "probe_color_margin"] = -0.10  # declared colour loses badly
    flagged = sieve.apply_thresholds(test, thr)
    assert bool(flagged.loc[1, "flag_probe_color"])
    # clean-margin rows must not fire
    assert not flagged.loc[3, "flag_probe_color"]


def test_signal_ablation_interface():
    clean = make_signals()
    thr = sieve.calibrate(clean, CFG, SCHEMA)
    test = make_signals(5, seed=3)
    test.loc[0, "sim_full"] = 0.10
    only_probe = sieve.apply_thresholds(test, thr, enabled_signals=["flag_probe_color"])
    assert not bool(only_probe.loc[0, "flagged"])  # low_sim disabled
