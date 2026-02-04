"""Imitation learning: invariant-space and geodesic losses."""

import numpy as np
from dhb_xr.encoder.dhb_dr import encode_dhb_dr
from dhb_xr.core.types import DHBMethod, EncodingMethod
from dhb_xr.losses.invariant_loss import invariant_matching_loss
from dhb_xr.losses.geodesic_loss import se3_geodesic_loss_np
from dhb_xr.losses.hybrid_loss import hybrid_invariant_pose_loss

def main():
    np.random.seed(4)
    n = 25
    demo_pos = np.cumsum(np.random.randn(n, 3) * 0.02, axis=0)
    demo_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    pred_pos = demo_pos + np.random.randn(n, 3) * 0.005
    pred_quat = demo_quat.copy()

    out_demo = encode_dhb_dr(
        demo_pos, demo_quat,
        method=EncodingMethod.POSITION, use_default_initial_frames=True, dhb_method=DHBMethod.DOUBLE_REFLECTION,
    )
    out_pred = encode_dhb_dr(
        pred_pos, pred_quat,
        method=EncodingMethod.POSITION, use_default_initial_frames=True, dhb_method=DHBMethod.DOUBLE_REFLECTION,
    )
    U_demo = np.concatenate([
        out_demo["linear_motion_invariants"],
        out_demo["angular_motion_invariants"],
    ], axis=1)
    U_pred = np.concatenate([
        out_pred["linear_motion_invariants"],
        out_pred["angular_motion_invariants"],
    ], axis=1)

    loss_inv = invariant_matching_loss(U_pred, U_demo, method="dhb_dr")
    print("Invariant matching loss (L2):", loss_inv)

    loss_pose = se3_geodesic_loss_np(
        pred_pos[-1], pred_quat[-1],
        demo_pos[-1], demo_quat[-1],
        beta=1.0,
    )
    print("SE(3) geodesic loss (final pose):", loss_pose)

    loss_hybrid = hybrid_invariant_pose_loss(
        pred_pos, pred_quat, demo_pos, demo_quat,
        pred_invariants=U_pred, demo_invariants=U_demo,
        alpha=0.5, beta=1.0,
    )
    print("Hybrid (invariant + pose) loss:", loss_hybrid)

if __name__ == "__main__":
    main()
