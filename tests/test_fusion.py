"""Decision-fusion calibration tests (roadmap 3.4/3.6)."""

import numpy as np
import pandas as pd

from tiger.fusion import FusionConfig, calibrate_fusion


def _labeled(n_clean, n_dirty, dirty_z, clean_z, field="color"):
    """Frame where dirty rows have probe z=dirty_z and clean rows z=clean_z."""
    rows = []
    for _ in range(n_dirty):
        rows.append({"noise_label": "mutate_text", f"probe_{field}_z": dirty_z})
    for _ in range(n_clean):
        rows.append({"noise_label": "clean", f"probe_{field}_z": clean_z})
    return pd.DataFrame(rows)


def test_clean_separable_signal_kept_at_base_margin():
    # all dirty at z=-3, all clean at z=+1 -> firing at z<=-2 is pure
    df = _labeled(50, 20, dirty_z=-3.0, clean_z=1.0)
    fc = calibrate_fusion(df, ["color"], precision_floor=0.85)
    sig = fc.per_signal["flag_probe_color"]
    assert not sig["quarantined"]
    assert sig["z_margin"] == 2.0
    assert sig["precision"] == 1.0


def test_noisy_signal_tightened_to_meet_floor():
    # some clean rows leak in at moderate z; tighter margin should purify
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(40):
        rows.append({"noise_label": "mutate_text", "probe_color_z": -3.5})
    for _ in range(40):
        # clean rows scattered around -2.2 (would leak at z>=2.0, gone at z>=3.0)
        rows.append({"noise_label": "clean", "probe_color_z": -2.2})
    df = pd.DataFrame(rows)
    fc = calibrate_fusion(df, ["color"], precision_floor=0.9, z_grid=(2.0, 3.0))
    sig = fc.per_signal["flag_probe_color"]
    assert sig["z_margin"] == 3.0        # tightened past the clean cluster
    assert not sig["quarantined"]


def test_unsalvageable_signal_quarantined():
    # dirty and clean both fire at every margin -> cannot meet floor -> quarantine
    rows = []
    for _ in range(20):
        rows.append({"noise_label": "mutate_text", "probe_color_z": -5.0})
    for _ in range(60):
        rows.append({"noise_label": "clean", "probe_color_z": -5.0})
    df = pd.DataFrame(rows)
    fc = calibrate_fusion(df, ["color"], precision_floor=0.85, z_grid=(2.0, 3.0, 4.0))
    assert fc.per_signal["flag_probe_color"]["quarantined"]


def test_config_json_roundtrip_and_accessors():
    fc = FusionConfig(per_signal={"flag_probe_color": {"z_margin": 2.5, "quarantined": False},
                                  "flag_probe_material": {"z_margin": 4.0, "quarantined": True}},
                      precision_floor=0.85)
    fc2 = FusionConfig.from_json(fc.to_json())
    assert fc2.z_margin("flag_probe_color", 2.0) == 2.5
    assert fc2.is_quarantined("flag_probe_material")
    assert not fc2.is_quarantined("flag_probe_color")
    assert fc2.z_margin("flag_probe_unknown", 2.0) == 2.0  # default fallback
