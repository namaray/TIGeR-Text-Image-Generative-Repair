"""Repair-planning tests: cost-minimal V2T patch, F14-safe T2V candidate pool."""

from pathlib import Path

import numpy as np
import pytest

from tiger.schema import load_schema
from tiger.solver import CandidatePool, plan_repair

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def schema():
    return load_schema(ROOT / "configs/schema.yaml")


class Route:
    def __init__(self, direction):
        self.direction = direction


def _unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


def test_v2t_plan_single_field_from_evidence(schema):
    ev = {"row_id": "r1", "category": "shirts", "product_id": "r1",
          "loo_top_field": "color", "pixel_color": "blue", "pixel_color_confidence": 0.9,
          "probes": {"color": {"z": -3.0, "pred": "blue"}}}
    row = {"attributes": '{"color": "red", "material": "cotton", "pattern": "solid", "size": "M"}',
           "title": "Red Shirt", "category": "shirts"}
    plan = plan_repair(ev, Route("V2T"), row, pool=None, cat_ids=None,
                       caption_emb=None, schema=schema)
    assert plan.plannable
    assert plan.patch == {"color": "blue"}
    assert plan.cost == 1.0


def test_v2t_unplannable_when_no_valid_value(schema):
    ev = {"row_id": "r1", "category": "shirts", "product_id": "r1",
          "loo_top_field": "color", "probes": {"color": {"z": -3.0, "pred": ""}}}
    row = {"attributes": '{"color": "red"}', "title": "Red Shirt", "category": "shirts"}
    plan = plan_repair(ev, Route("V2T"), row, None, None, None, schema)
    assert not plan.plannable


def test_v2t_skips_noop_patch(schema):
    # image already agrees with the declared value -> no repair needed
    ev = {"row_id": "r1", "category": "shirts", "product_id": "r1",
          "loo_top_field": "color", "pixel_color": "red", "pixel_color_confidence": 0.9,
          "probes": {"color": {"z": -3.0, "pred": "red"}}}
    row = {"attributes": '{"color": "red"}', "title": "Red Shirt", "category": "shirts"}
    plan = plan_repair(ev, Route("V2T"), row, None, None, None, schema)
    assert not plan.plannable


def test_candidate_pool_excludes_own_product():
    emb = np.stack([_unit(np.array([1.0, 0.0])),   # p1 (own)
                    _unit(np.array([0.98, 0.1])),  # p2
                    _unit(np.array([0.0, 1.0]))])  # p3
    pool = CandidatePool(emb, ["p1", "p2", "p3"],
                         ["a.jpg", "b.jpg", "c.jpg"], np.array([True, True, True]))
    # caption points at p1's direction; own product excluded -> must pick p2
    res = pool.best_for_text(emb[0], exclude_product="p1")
    assert res is not None
    j, _ = res
    assert j == 1


def test_t2v_plan_uses_pool(schema):
    emb = np.stack([_unit(np.array([1.0, 0.0])),
                    _unit(np.array([0.9, 0.2])),
                    _unit(np.array([0.0, 1.0]))])
    pool = CandidatePool(emb, ["r1", "r2", "r3"],
                         ["own.jpg", "cand.jpg", "other.jpg"], np.array([True, True, True]))
    cat_ids = np.array(["shirts", "shirts", "shirts"])
    ev = {"row_id": "r1", "product_id": "r1", "category": "shirts"}
    plan = plan_repair(ev, Route("T2V"), {"attributes": "{}"}, pool, cat_ids,
                       caption_emb=emb[0], schema=schema)
    assert plan.plannable
    assert plan.candidate_product_id == "r2"      # own product held out (F14)
    assert plan.candidate_image_path == "cand.jpg"


def test_t2v_unplannable_when_pool_empty(schema):
    emb = np.stack([_unit(np.array([1.0, 0.0]))])
    pool = CandidatePool(emb, ["r1"], ["own.jpg"], np.array([True]))
    ev = {"row_id": "r1", "product_id": "r1", "category": "shirts"}
    plan = plan_repair(ev, Route("T2V"), {"attributes": "{}"}, pool,
                       np.array(["shirts"]), emb[0], schema)
    assert not plan.plannable
