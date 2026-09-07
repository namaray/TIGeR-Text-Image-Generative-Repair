"""Unit tests for the Phase-1 pieces of the Analyzer: routing constraints (1.4)
and the neighbour interface with z-scored margins (1.6)."""

import numpy as np

from tiger.analyzer import NeighborIndex, allowed_directions


def test_missing_image_never_v2t():
    dirs = allowed_directions(image_missing=True, text_missing=False)
    assert "V2T" not in dirs           # F11: no visual evidence -> no V2T
    assert "T2V_ACQUIRE" in dirs


def test_missing_text_routes_v2t():
    dirs = allowed_directions(image_missing=False, text_missing=True)
    assert "V2T" in dirs
    assert "T2V" not in dirs


def test_both_missing_human_only():
    assert allowed_directions(True, True) == ["HUMAN"]


def test_nothing_missing_all_directions():
    assert set(allowed_directions(False, False)) == {"V2T", "T2V", "HUMAN"}


def _unit(v):
    return v / np.linalg.norm(v)


def test_neighbor_index_excludes_product():
    emb = np.stack([_unit(np.array([1.0, 0.0])),
                    _unit(np.array([0.99, 0.1])),
                    _unit(np.array([0.0, 1.0]))]).astype(np.float32)
    idx = NeighborIndex(emb, ["p1", "p1", "p2"], np.array([True, True, True]))
    # querying with p1's own vector while excluding p1 must return only p2
    res = idx.topk(emb[0], k=2, exclude_product="p1")
    assert [i for i, _ in res] == [2]


def test_neighbor_index_respects_valid_mask():
    emb = np.eye(3, dtype=np.float32)
    idx = NeighborIndex(emb, ["a", "b", "c"], np.array([True, False, True]))
    res = idx.topk(emb[1], k=3)
    assert 1 not in [i for i, _ in res]


def test_neighbor_index_orders_by_cosine():
    emb = np.stack([_unit(np.array([1.0, 0.0])),
                    _unit(np.array([0.9, 0.4])),
                    _unit(np.array([0.1, 1.0]))]).astype(np.float32)
    idx = NeighborIndex(emb, ["a", "b", "c"], np.ones(3, dtype=bool))
    res = idx.topk(emb[0], k=3, exclude_product="a")
    assert [i for i, _ in res] == [1, 2]
