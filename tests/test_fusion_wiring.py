"""Fusion must be reachable from the live pipeline, and off by default (A7).

`detect` and `run_repair_cycle` called apply_thresholds() without a fusion
config; only `ablate` passed one. The advertised fused operating point
(P=0.888/R=0.882) was therefore an offline ablation row, while detect, analyze,
route, repair and every repair-side number ran un-fused at P=0.793/R=0.924.

Loading is opt-in because fusion trades recall for precision (mutate_text
recall 0.853 -> 0.773), which changes what reaches the repair stage.
"""

import inspect

import pytest

from tiger import cli, repair as repair_mod
from tiger.eval import repair_ablation


def test_fusion_is_off_unless_requested():
    assert cli._load_fusion(False) is None


def test_requesting_fusion_without_calibration_fails_loudly():
    with pytest.raises(FileNotFoundError, match="calibrate-fusion"):
        cli._load_fusion(True)


def test_repair_cycle_accepts_a_fusion_config():
    sig = inspect.signature(repair_mod.run_repair_cycle)
    assert "fusion" in sig.parameters, "repair cycle cannot reach the fused operating point"
    assert sig.parameters["fusion"].default is None, "must stay opt-in"


def test_repair_ablation_accepts_a_fusion_config():
    sig = inspect.signature(repair_ablation.run_repair_ablations)
    assert "fusion" in sig.parameters
    assert sig.parameters["fusion"].default is None


def test_cli_exposes_the_flag():
    src = inspect.getsource(cli.main)
    assert '"--fusion"' in src
