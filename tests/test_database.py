"""Tests for motion database."""

import json
import os
import tempfile

import numpy as np
import pytest
from dhb_xr.database.motion_db import MotionDatabase, HAS_FAISS
from dhb_xr.database.similarity import invariant_distance, soft_dtw_distance


def _make_trajectory(n=30, seed=None):
    """Helper: generate a random trajectory (positions, quaternions)."""
    rng = np.random.default_rng(seed)
    positions = np.cumsum(rng.standard_normal((n, 3)) * 0.01, axis=0)
    quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    return positions, quaternions


def test_motion_db_add_retrieve():
    db = MotionDatabase(use_faiss=False)
    n = 30
    for _ in range(3):
        positions = np.cumsum(np.random.randn(n, 3) * 0.01, axis=0)
        quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
        db.add(positions, quaternions, metadata={"id": _})
    assert len(db.invariants_list) == 3
    query_pos = db.invariants_list[0]
    from dhb_xr.encoder.dhb_dr import encode_dhb_dr
    from dhb_xr.core.types import DHBMethod
    # Build a trajectory from first stored invariants (we need positions/quat for retrieve)
    pos = np.cumsum(np.random.randn(n, 3) * 0.01, axis=0)
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    results = db.retrieve(pos, quat, k=2)
    assert len(results) == 2
    inv, meta, dist = results[0]
    assert inv.shape[0] == db.invariants_list[0].shape[0]


# --- Bug 1: FAISS retrieval is dead code ---

@pytest.mark.skipif(not HAS_FAISS, reason="faiss not installed")
def test_faiss_retrieve_uses_index():
    """retrieve() with use_faiss=True should query the FAISS index, not brute-force."""
    db = MotionDatabase(use_faiss=True)
    n = 30
    # Add several trajectories with the same length
    seeds = [10, 20, 30, 40, 50]
    for i, s in enumerate(seeds):
        pos, quat = _make_trajectory(n, seed=s)
        db.add(pos, quat, metadata={"id": i})

    assert db.index is not None
    assert db.index.ntotal == len(seeds)

    # Query with one of the stored trajectories; FAISS should return results
    query_pos, query_quat = _make_trajectory(n, seed=10)
    results = db.retrieve(query_pos, query_quat, k=3)
    assert len(results) == 3
    # The closest match should be id=0 (same seed)
    _, meta, dist = results[0]
    assert meta["id"] == 0
    assert dist < results[1][2]  # closest is strictly closer


def test_faiss_retrieve_code_path_is_used():
    """When use_faiss=True and index exists, retrieve() must query the index (not brute-force)."""
    from unittest.mock import MagicMock, patch

    db = MotionDatabase(use_faiss=False)
    n = 30
    seeds = [10, 20, 30]
    for i, s in enumerate(seeds):
        pos, quat = _make_trajectory(n, seed=s)
        db.add(pos, quat, metadata={"id": i})

    # Simulate having a FAISS index
    db.use_faiss = True
    dim = db.invariants_list[0].flatten().shape[0]
    db.embedding_dim = dim

    mock_index = MagicMock()
    # Return distances and indices for k=2
    mock_index.search.return_value = (
        np.array([[0.0, 1.5]], dtype=np.float32),  # distances
        np.array([[0, 2]], dtype=np.int64),          # indices
    )
    db.index = mock_index

    query_pos, query_quat = _make_trajectory(n, seed=10)
    results = db.retrieve(query_pos, query_quat, k=2)

    # The FAISS index.search MUST have been called
    mock_index.search.assert_called_once()
    assert len(results) == 2
    # Results should come from FAISS indices (0 and 2)
    assert results[0][1] == db.metadata_list[0]
    assert results[1][1] == db.metadata_list[2]


# --- Bug 2: No persistence ---

def test_save_and_load_roundtrip():
    """MotionDatabase.save() then load() should restore invariants and metadata."""
    db = MotionDatabase(use_faiss=False)
    n = 30
    for i in range(3):
        pos, quat = _make_trajectory(n, seed=i + 100)
        db.add(pos, quat, metadata={"id": i, "name": f"traj_{i}"})

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_db")
        db.save(path)

        # Verify files exist
        assert os.path.exists(path + ".npz")
        assert os.path.exists(path + ".json")

        # Load into a new database
        db2 = MotionDatabase.load(path)
        assert len(db2.invariants_list) == 3
        assert len(db2.metadata_list) == 3
        for i in range(3):
            np.testing.assert_array_almost_equal(db.invariants_list[i], db2.invariants_list[i])
            assert db.metadata_list[i] == db2.metadata_list[i]


# --- Bug 3: Soft-DTW gamma parameter ignored ---

def test_soft_dtw_fallback_varies_with_gamma():
    """The fallback DTW implementation should produce different results for different gamma values.

    With hard DTW (gamma ignored), changing gamma has no effect.
    With soft-DTW, larger gamma -> smoother min -> larger distance.
    """
    rng = np.random.default_rng(42)
    U = rng.standard_normal((10, 4))
    V = rng.standard_normal((12, 4))

    d_small_gamma = soft_dtw_distance(U, V, gamma=0.01)
    d_large_gamma = soft_dtw_distance(U, V, gamma=10.0)

    # If gamma were ignored, these would be equal. They must differ.
    assert d_small_gamma != pytest.approx(d_large_gamma, rel=1e-6), \
        "soft_dtw_distance ignores gamma parameter (returns same value for different gammas)"


# --- Bug 4: Variable-length L2 distance ---

def test_invariant_distance_mismatched_lengths():
    """invariant_distance should handle sequences of different lengths via interpolation."""
    rng = np.random.default_rng(99)
    U = rng.standard_normal((20, 8))
    V = rng.standard_normal((30, 8))

    # Currently this would assert-fail. After fix, it should return a finite distance.
    dist = invariant_distance(U, V)
    assert np.isfinite(dist)
    assert dist > 0.0


def test_invariant_distance_equal_lengths_unchanged():
    """invariant_distance with equal-length inputs should work as before."""
    rng = np.random.default_rng(77)
    U = rng.standard_normal((20, 8))
    V = rng.standard_normal((20, 8))

    dist = invariant_distance(U, V)
    # Should match the manual L2 computation
    expected = float(np.sqrt(np.sum((U - V) ** 2)))
    assert dist == pytest.approx(expected, rel=1e-10)
