"""Tests for motion database."""

import numpy as np
import pytest
from dhb_xr.database.motion_db import MotionDatabase
from dhb_xr.database.similarity import invariant_distance


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
