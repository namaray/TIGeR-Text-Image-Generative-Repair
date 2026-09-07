"""Decision fusion calibrated to a precision floor (roadmap 3.4 + 3.6 guardrails).

The sieve OR-combines its signals. That maximises recall but lets a weak signal
(e.g. the material probe at ~0.73 precision) erode the precision story. This
module tunes each probe signal's z-margin on a LABELLED calibration set so that
every retained signal individually meets a precision floor, and QUARANTINES any
signal that cannot reach the floor even at its tightest setting.

Non-VLM half of the precision guardrails (3.6). The VLM audit of a random sample
of new flags is deferred until an API key is available; the quarantine mechanism
and the per-signal precision report exist now.

Output: a FusionConfig (per-signal z-margin + quarantine flag) consumed by
tiger.sieve.apply_thresholds. Calibrated on the calibration split only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# Which noise subtypes each probe field legitimately claims to detect (A8).
# The colour probe answers "does the image show the declared colour?", so a
# colour probe firing on a material_flip row is a coincidence, not a hit.
# Image-side corruptions are included because a swapped image genuinely does
# contradict the declared attribute -- the probe is right about that row.
PROBE_TARGETS = {
    "color": {"color_flip", "near_color_flip", "attribute_drop", "mixed_swap_color",
              "swap_image", "swap_image_same_category", "missing_image"},
    "material": {"material_flip", "mixed_swap_color",
                 "swap_image", "swap_image_same_category", "missing_image"},
    "pattern": {"mixed_swap_color",
                "swap_image", "swap_image_same_category", "missing_image"},
}


@dataclass
class FusionConfig:
    per_signal: dict = field(default_factory=dict)  # signal -> {z_margin, quarantined, precision, fired}
    precision_floor: float = 0.85
    meta: dict = field(default_factory=dict)

    def z_margin(self, field_name: str, default: float) -> float:
        return float(self.per_signal.get(field_name, {}).get("z_margin", default))

    def is_quarantined(self, signal: str) -> bool:
        return bool(self.per_signal.get(signal, {}).get("quarantined", False))

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "FusionConfig":
        return cls(**json.loads(s))


def calibrate_fusion(labeled: pd.DataFrame, probe_fields: list[str],
                     precision_floor: float = 0.85,
                     z_grid: tuple[float, ...] = (2.0, 2.5, 3.0, 3.5, 4.0),
                     zcol_fmt: str = "probe_{}_z") -> FusionConfig:
    """Tune each probe's z-margin to the smallest value meeting the precision floor.

    `labeled` must carry `noise_label`, `noise_subtype` and per-probe z columns.

    A row counts as a true positive for a probe when it is dirty AND its subtype
    is one the probe claims (PROBE_TARGETS). Scoring against `dirty` alone -- the
    previous behaviour -- certified "this signal fires on dirty rows", not "this
    signal identifies the error it names", which is what the precision-floor
    claim needs. Both figures are recorded: `precision` (on-target, enforced) and
    `precision_any_dirty` (the older, looser number, for comparison). Without a
    `noise_subtype` column the two coincide and behaviour is unchanged.

    Non-probe signals (low_sim, text checks, missing) are measured and reported
    but not swept here -- low_sim precision is governed by the locked tau, and
    the text checks are near-deterministic.
    """
    dirty = (labeled["noise_label"].astype(str) != "clean")
    subtypes = (labeled["noise_subtype"].astype(str)
                if "noise_subtype" in labeled.columns else None)
    per_signal: dict[str, dict] = {}

    for fld in probe_fields:
        zcol = zcol_fmt.format(fld)
        if zcol not in labeled.columns:
            continue
        z = labeled[zcol]

        # A8: `dirty` counts a row as a hit when it is corrupted for ANY reason,
        # so the material probe firing on a swap_image row scored as correct.
        # `on_target` restricts credit to the subtypes this probe actually
        # claims. The floor is enforced on the stricter figure; the looser one
        # is still reported so the two can be compared.
        on_target = dirty if subtypes is None else (
            dirty & subtypes.isin(PROBE_TARGETS.get(fld, set())))

        def _measure(zt: float) -> dict:
            fired = (~z.isna()) & (z <= -zt)
            n = int(fired.sum())
            return {
                "z_margin": float(zt),
                "fired": n,
                "precision": round(float(on_target[fired].mean()), 3) if n else None,
                "precision_any_dirty": round(float(dirty[fired].mean()), 3) if n else None,
            }

        chosen = None
        for zt in z_grid:
            m = _measure(zt)
            if m["fired"] and m["precision"] is not None and m["precision"] >= precision_floor:
                chosen = {**m, "quarantined": False}
                break
        if chosen is None:
            # even the tightest margin misses the floor -> quarantine
            chosen = {**_measure(z_grid[-1]), "quarantined": True}
        per_signal[f"flag_probe_{fld}"] = chosen

    # report-only precision for the non-swept signals
    for sig in ["flag_low_sim", "flag_text_out_of_domain", "flag_title_contradiction"]:
        if sig in labeled.columns:
            fired = labeled[sig].astype(bool)
            n = int(fired.sum())
            per_signal.setdefault(sig, {})
            per_signal[sig].update({"precision": round(float(dirty[fired].mean()), 3) if n else None,
                                    "precision_any_dirty": round(float(dirty[fired].mean()), 3) if n else None,
                                    "fired": n, "quarantined": False, "z_margin": None})

    return FusionConfig(
        per_signal=per_signal, precision_floor=precision_floor,
        meta={"z_grid": list(z_grid), "n_calibration_rows": int(len(labeled)),
              "n_dirty": int(dirty.sum())},
    )
