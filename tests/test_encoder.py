"""Tests for DHB encoders."""

import numpy as np
import pytest
from dhb_xr.encoder.dhb_dr import encode_dhb_dr
from dhb_xr.core.types import DHBMethod


def test_encode_dhb_dr_basic():
    n = 50
    np.random.seed(42)
    positions = np.cumsum(np.random.randn(n, 3) * 0.01, axis=0)
    quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    out = encode_dhb_dr(
        positions, quaternions,
        method="pos", use_default_initial_frames=True, dhb_method=DHBMethod.DOUBLE_REFLECTION,
    )
    assert "linear_motion_invariants" in out
    assert "angular_motion_invariants" in out
    # With default padding, encoder returns (n+5-2) = (n+3) - 2 = n+1? Actually n+5 positions -> n+4 diffs -> n+2 invariant steps. So shape[0] = n+2 for n=50 -> 52.
    assert out["linear_motion_invariants"].shape[0] >= n - 2
    assert out["linear_motion_invariants"].shape[1] == 4
    assert out["angular_motion_invariants"].shape[1] == 4


def test_encode_dhb_dr_original():
    n = 30
    positions = np.cumsum(np.random.randn(n, 3) * 0.01, axis=0)
    quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    out = encode_dhb_dr(
        positions, quaternions,
        method="pos", use_default_initial_frames=True, dhb_method=DHBMethod.ORIGINAL,
    )
    assert out["linear_motion_invariants"].shape[1] == 3
    assert out["angular_motion_invariants"].shape[1] == 3
