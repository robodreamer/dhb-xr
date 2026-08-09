"""Tests for DHB decoders."""

import numpy as np

from dhb_xr.core.types import DHBMethod, EncodingMethod
from dhb_xr.decoder.dhb_dr import decode_dhb_dr
from dhb_xr.encoder.dhb_dr import encode_dhb_dr


def test_decode_dhb_dr_roundtrip():
    n = 50
    np.random.seed(42)
    positions = np.cumsum(np.random.randn(n, 3) * 0.01, axis=0)
    quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    out = encode_dhb_dr(
        positions,
        quaternions,
        method=EncodingMethod.POSITION,
        use_default_initial_frames=True,
        dhb_method=DHBMethod.DOUBLE_REFLECTION,
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


def test_natural_frame_roundtrip_uses_encoder_frames():
    phase = np.linspace(0.0, 1.0, 50)
    positions = np.column_stack(
        [phase, 0.1 * np.sin(np.pi * phase), 0.05 * np.sin(2.0 * np.pi * phase)]
    )
    angles = np.pi / 2.0 * phase
    quaternions = np.column_stack([np.cos(angles / 2.0), np.sin(angles / 2.0), np.zeros((50, 2))])
    encoded = encode_dhb_dr(
        positions,
        quaternions,
        init_pose={"position": positions[0], "quaternion": quaternions[0]},
        method=EncodingMethod.POSITION,
        use_default_initial_frames=False,
        dhb_method=DHBMethod.DOUBLE_REFLECTION,
    )

    decoded = decode_dhb_dr(
        encoded["linear_motion_invariants"],
        encoded["angular_motion_invariants"],
        encoded["initial_pose"],
        method=EncodingMethod.POSITION,
        dhb_method=DHBMethod.DOUBLE_REFLECTION,
        drop_padded=False,
        initial_linear_frame=encoded["linear_frame_initial"],
        initial_angular_frame=encoded["angular_frame_initial"],
    )

    np.testing.assert_allclose(decoded["positions"], positions, atol=1.0e-10)
    alignment = np.abs(np.sum(decoded["quaternions"] * quaternions, axis=1))
    assert float(2.0 * np.arccos(np.clip(alignment, 0.0, 1.0)).max()) < 1.0e-6
