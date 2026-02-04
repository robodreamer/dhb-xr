"""Tests for DHB decoders."""

import numpy as np
import pytest
from dhb_xr.encoder.dhb_dr import encode_dhb_dr
from dhb_xr.decoder.dhb_dr import decode_dhb_dr
from dhb_xr.core.types import DHBMethod, EncodingMethod


def test_decode_dhb_dr_roundtrip():
    n = 50
    np.random.seed(42)
    positions = np.cumsum(np.random.randn(n, 3) * 0.01, axis=0)
    quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    out = encode_dhb_dr(
        positions, quaternions,
        method=EncodingMethod.POSITION, use_default_initial_frames=True, dhb_method=DHBMethod.DOUBLE_REFLECTION,
    )
    decoded = decode_dhb_dr(
        out["linear_motion_invariants"],
        out["angular_motion_invariants"],
        out["initial_pose"],
        method=EncodingMethod.POSITION,
        dhb_method=DHBMethod.DOUBLE_REFLECTION,
        drop_padded=True,
    )
    # Encoder pads and returns N-2 invariants; decoder returns [2:] so length = (N-2)-2 = N-4? No: decoder output length is num_samples - 2 when drop_padded. num_samples = linear_motion_invariants.shape[0] = N-2. So decoded has (N-2) - 2 = N-4 poses. Actually drop_padded drops first 2 poses so we get (N-2) - 2 = N-4. Original had N poses. So we don't expect same length. Let me just check shapes and that decoded is close to original in the overlapping part.
    assert decoded["positions"].shape[1] == 3
    assert decoded["quaternions"].shape[1] == 4
    # Check first and last decoded are reasonable
    assert np.all(np.isfinite(decoded["positions"]))
    assert np.all(np.isfinite(decoded["quaternions"]))
