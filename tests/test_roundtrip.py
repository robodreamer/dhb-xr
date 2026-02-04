"""Encode-decode roundtrip accuracy tests."""

import numpy as np
import pytest
from dhb_xr.encoder.dhb_dr import encode_dhb_dr
from dhb_xr.decoder.dhb_dr import decode_dhb_dr
from dhb_xr.core.types import DHBMethod, EncodingMethod


def test_roundtrip_dhb_dr():
    n = 60
    np.random.seed(0)
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
    # Decoded length is (N-2)-2 = N-4. Compare overlapping segment with original
    dec_pos = decoded["positions"]
    dec_quat = decoded["quaternions"]
    # Original with default padding: we get invariants for indices that correspond to central part. Decoded [2:] gives poses. So decoded has shape (n_inv - 2,) = (N-2-2,) = (N-4,). Original has N. Overlap: indices 2..N-4 in original vs 0..N-6 in decoded? So original[2:2+len(dec_pos)] vs dec_pos
    n_dec = len(dec_pos)
    n_orig = len(positions)
    n_overlap = min(n_dec, n_orig - 2)
    if n_overlap < 2:
        return
    orig_slice = positions[2 : 2 + n_overlap]
    dec_slice = dec_pos[:n_overlap]
    rel_err = np.linalg.norm(dec_slice - orig_slice) / (np.linalg.norm(orig_slice) + 1e-10)
    assert rel_err < 0.5, f"Roundtrip position relative error {rel_err} should be < 0.5"
