"""Per-signal precision must measure the error the signal names (finding A8).

calibrate_fusion scored a probe as correct whenever the row was dirty for ANY
reason, so a material probe firing on a swap_image row counted as a hit. The
0.85 floor then certified "fires on dirty rows" rather than "identifies the
error it names".
"""

import pandas as pd

from tiger.fusion import PROBE_TARGETS, calibrate_fusion


def _frame(rows):
    return pd.DataFrame(rows, columns=["noise_label", "noise_subtype", "probe_material_z"])


def test_off_target_hits_no_longer_count_as_precision():
    """All fired rows are dirty, but none are material errors."""
    df = _frame([("mutate_text", "color_flip", -5.0)] * 10)
    fc = calibrate_fusion(df, ["material"], precision_floor=0.85)
    sig = fc.per_signal["flag_probe_material"]
    assert sig["precision"] == 0.0, "colour flips are not material errors"
    assert sig["precision_any_dirty"] == 1.0, "the looser figure is still reported"
    assert sig["quarantined"] is True


def test_on_target_hits_still_count():
    df = _frame([("mutate_text", "material_flip", -5.0)] * 10)
    fc = calibrate_fusion(df, ["material"], precision_floor=0.85)
    sig = fc.per_signal["flag_probe_material"]
    assert sig["precision"] == 1.0
    assert sig["quarantined"] is False


def test_clean_rows_are_never_on_target():
    df = _frame([("clean", "clean", -5.0)] * 4 + [("mutate_text", "material_flip", -5.0)] * 6)
    sig = calibrate_fusion(df, ["material"], precision_floor=0.85).per_signal["flag_probe_material"]
    assert sig["precision"] == 0.6 and sig["quarantined"] is True


def test_image_corruption_counts_for_every_probe():
    """A swapped image genuinely contradicts the declared attribute."""
    for fld in ("color", "material", "pattern"):
        assert "swap_image" in PROBE_TARGETS[fld]


def test_missing_subtype_column_preserves_old_behaviour():
    df = pd.DataFrame([("mutate_text", -5.0)] * 10, columns=["noise_label", "probe_material_z"])
    sig = calibrate_fusion(df, ["material"], precision_floor=0.85).per_signal["flag_probe_material"]
    assert sig["precision"] == sig["precision_any_dirty"] == 1.0
