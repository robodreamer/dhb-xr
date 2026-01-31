"""
Trajectory adaptation: resample demo and solve NLP to find invariants that match boundary poses.
Full CasADi optimization requires casadi (pip install dhb_xr[optimization]).
Without CasADi, generate_trajectory falls back to simple interpolation.
"""

import numpy as np
from typing import Dict, Any, Optional

from dhb_xr.encoder.dhb_dr import encode_dhb_dr
from dhb_xr.decoder.dhb_dr import decode_dhb_dr
from dhb_xr.core.types import DHBMethod
from dhb_xr.core import geometry as geom
from dhb_xr.utils.resampling import resample_and_smooth

try:
    import casadi as ca
    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False


def _euler_to_rot_casadi(angles):
    """Euler XYZ (extrinsic) to 3x3 rotation matrix in CasADi."""
    rx, ry, rz = angles[0], angles[1], angles[2]
    cx, sx = ca.cos(rx), ca.sin(rx)
    cy, sy = ca.cos(ry), ca.sin(ry)
    cz, sz = ca.cos(rz), ca.sin(rz)
    Rx = ca.vertcat(ca.horzcat(1, 0, 0), ca.horzcat(0, cx, -sx), ca.horzcat(0, sx, cx))
    Ry = ca.vertcat(ca.horzcat(cy, 0, sy), ca.horzcat(0, 1, 0), ca.horzcat(-sy, 0, cy))
    Rz = ca.vertcat(ca.horzcat(cz, -sz, 0), ca.horzcat(sz, cz, 0), ca.horzcat(0, 0, 1))
    return Rz @ Ry @ Rx


def _axis_angle_to_rot_casadi(rvec, use_mx=False):
    """Axis-angle to rotation matrix (Rodrigues) in CasADi."""
    th = ca.sqrt(rvec[0]**2 + rvec[1]**2 + rvec[2]**2 + 1e-12)
    k = rvec / th
    K = ca.vertcat(
        ca.horzcat(0, -k[2], k[1]),
        ca.horzcat(k[2], 0, -k[0]),
        ca.horzcat(-k[1], k[0], 0),
    )
    eye = ca.MX.eye(3) if use_mx else ca.SX.eye(3)
    return eye + ca.sin(th) * K + (1 - ca.cos(th)) * (K @ K)


def _rot_to_rvec_casadi(R):
    """Rotation matrix to axis-angle (simplified, for small angles)."""
    # Use logarithm approximation for CasADi compatibility
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    theta = ca.acos(ca.fmax(-1, ca.fmin(1, (trace - 1) / 2)))
    denom = 2 * ca.sin(theta) + 1e-12
    rx = (R[2, 1] - R[1, 2]) / denom * theta
    ry = (R[0, 2] - R[2, 0]) / denom * theta
    rz = (R[1, 0] - R[0, 1]) / denom * theta
    return ca.vertcat(rx, ry, rz)


def generate_trajectory(
    pos_data: np.ndarray,
    quat_data: np.ndarray,
    pose_target_init: Dict[str, np.ndarray],
    pose_target_final: Dict[str, np.ndarray],
    traj_length: int,
    smoothing: bool = False,
    dhb_method: DHBMethod = DHBMethod.DOUBLE_REFLECTION,
    enable_smoothing_objective: bool = False,
    enable_collision_constraints: bool = False,
    weights: Optional[np.ndarray] = None,
    use_casadi: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Generate trajectory by resampling demo and solving NLP to match boundary poses.

    If CasADi is available and use_casadi=True, solves an optimization problem:
    - Minimize ||U - U_demo||^2 (invariants close to demo)
    - Subject to: start pose = pose_target_init, end pose = pose_target_final
    - Dynamic constraints: poses are reconstructed from invariants

    Otherwise, falls back to simple interpolation.

    Args:
        pos_data: Demo positions (N, 3)
        quat_data: Demo quaternions (N, 4) wxyz
        pose_target_init: {'position': (3,), 'quaternion': (4,)} start pose
        pose_target_final: {'position': (3,), 'quaternion': (4,)} goal pose
        traj_length: Output trajectory length
        smoothing: Apply smoothing to resampled demo
        dhb_method: DHBMethod.DOUBLE_REFLECTION (4 inv) or ORIGINAL (3 inv)
        enable_smoothing_objective: Add smoothness penalty to objective
        weights: Weights for invariant terms (default: equal)
        use_casadi: Use CasADi NLP solver if available
        verbose: Print solver output

    Returns:
        Dict with adapted_pos_data, adapted_quat_data, invariants, etc.
    """
    # Resample demo
    pos_orig, quat_orig, rvec_orig, pos_resample, quat_resample, rvec_resample = resample_and_smooth(
        pos_data, quat_data, traj_length, smoothing
    )

    # Encode demo to get reference invariants
    invariants_out = encode_dhb_dr(
        pos_orig, quat_orig,
        init_pose=pose_target_init,
        method="pos",
        use_default_initial_frames=False,
        dhb_method=dhb_method,
    )
    lin_inv_demo = invariants_out["linear_motion_invariants"]
    ang_inv_demo = invariants_out["angular_motion_invariants"]
    invariants_demo = np.hstack([lin_inv_demo, ang_inv_demo])
    dim_inv = invariants_demo.shape[1]
    N = invariants_demo.shape[0]

    init_pos = np.asarray(pose_target_init["position"]).reshape(3)
    init_quat = np.asarray(pose_target_init["quaternion"]).reshape(4)
    goal_pos = np.asarray(pose_target_final["position"]).reshape(3)
    goal_quat = np.asarray(pose_target_final["quaternion"]).reshape(4)
    init_rvec = geom.quat_to_axis_angle(init_quat)
    goal_rvec = geom.quat_to_axis_angle(goal_quat)

    # Try CasADi optimization
    if HAS_CASADI and use_casadi:
        try:
            result = _solve_casadi_nlp(
                invariants_demo, N, dim_inv, traj_length,
                init_pos, init_rvec, goal_pos, goal_rvec,
                invariants_out["linear_frame_initial"],
                invariants_out["angular_frame_initial"],
                dhb_method, enable_smoothing_objective, weights, verbose,
            )
            if result is not None:
                return {
                    "linear_motion_invariant": result["invariants"][:, :dim_inv // 2],
                    "angular_motion_invariant": result["invariants"][:, dim_inv // 2:],
                    "adapted_pos_data": result["positions"],
                    "adapted_rvec_data": result["rvecs"],
                    "adapted_quat_data": np.array([geom.axis_angle_to_quat(result["rvecs"][i]) for i in range(len(result["rvecs"]))]),
                    "resampled_pos_data": pos_resample,
                    "resampled_quat_data": quat_resample,
                    "resampled_rvec_data": rvec_resample,
                    "solver": "casadi",
                }
        except Exception as e:
            if verbose:
                print(f"CasADi solver failed: {e}, falling back to interpolation")

    # Fallback: simple decode + smooth interpolation
    return _fallback_interpolation(
        lin_inv_demo, ang_inv_demo, pose_target_init, pose_target_final,
        traj_length, dhb_method, pos_resample, quat_resample, rvec_resample,
    )


def _solve_casadi_nlp(
    invariants_demo, N, dim_inv, traj_length,
    init_pos, init_rvec, goal_pos, goal_rvec,
    linear_frame_init, angular_frame_init,
    dhb_method, enable_smoothing_objective, weights, verbose,
):
    """Solve the NLP using CasADi Opti."""
    opti = ca.Opti()

    # Decision variables: invariants for each timestep
    U = opti.variable(N, dim_inv)

    # Pose variables
    P = [opti.variable(3) for _ in range(N)]
    R = [opti.variable(3) for _ in range(N)]  # rotation vectors

    # Weights
    if weights is None:
        weights = np.ones(dim_inv)
    else:
        weights = np.asarray(weights).reshape(dim_inv)

    # Normalize invariants for objective
    inv_min = invariants_demo.min(axis=0)
    inv_max = invariants_demo.max(axis=0)
    inv_range = inv_max - inv_min
    inv_range[inv_range == 0] = 1.0

    # Objective: minimize weighted deviation from demo invariants
    objective = 0
    for k in range(N):
        # U[k, :] is a row (1 x dim_inv), invariants_demo[k] is numpy array
        # Need to ensure compatible shapes for CasADi
        demo_k = invariants_demo[k, :].reshape(1, -1)  # (1, dim_inv)
        range_k = inv_range.reshape(1, -1)  # (1, dim_inv)
        weights_k = weights.reshape(1, -1)  # (1, dim_inv)
        e = (U[k, :] - demo_k) / range_k
        e_weighted = ca.sqrt(weights_k) * e
        objective += ca.sumsqr(e_weighted)

    # Smoothness penalty
    if enable_smoothing_objective:
        smooth_weight = 1e2
        for k in range(1, N - 1):
            diff = R[k + 1] - 2 * ca.vertcat(*R[k]) + R[k - 1]
            objective += smooth_weight * ca.sumsqr(diff)

    # Boundary constraints
    opti.subject_to(P[0] == init_pos)
    opti.subject_to(R[0] == init_rvec)

    # End pose constraint (on the last reconstructed pose)
    # We'll add this after setting up the dynamics

    # Dynamic constraints: reconstruct trajectory from invariants
    k_lin = dim_inv // 2  # 4 for DR, 3 for original
    # Use DM for initial constant matrices, then convert to MX-compatible operations
    linear_frame = ca.MX(linear_frame_init)
    angular_frame = ca.MX(angular_frame_init)
    rotm_accum = _axis_angle_to_rot_casadi(ca.MX(init_rvec), use_mx=True)

    for k in range(N):
        lin_inv = U[k, :k_lin]
        ang_inv = U[k, k_lin:]

        # Linear step
        mag_lin = lin_inv[0]
        euler_lin = lin_inv[1:4] if k_lin == 4 else ca.vertcat(0, lin_inv[1], lin_inv[2])
        R_lin = _euler_to_rot_casadi(euler_lin)
        trans = ca.vertcat(mag_lin, 0, 0)
        T_step = ca.vertcat(
            ca.horzcat(R_lin, trans),
            ca.horzcat(0, 0, 0, 1),
        )
        linear_frame = linear_frame @ T_step
        new_pos = linear_frame[:3, 3]

        # Angular step
        mag_ang = ang_inv[0]
        euler_ang = ang_inv[1:4] if k_lin == 4 else ca.vertcat(0, ang_inv[1], ang_inv[2])
        rvec_local = angular_frame[:3, :3] @ ca.vertcat(mag_ang, 0, 0)
        R_ang = _euler_to_rot_casadi(euler_ang)
        rotm_accum = rotm_accum @ _axis_angle_to_rot_casadi(rvec_local, use_mx=True).T
        new_rvec = _rot_to_rvec_casadi(rotm_accum)
        angular_frame = ca.vertcat(
            ca.horzcat(angular_frame[:3, :3] @ R_ang, ca.MX.zeros(3, 1)),
            ca.horzcat(0, 0, 0, 1),
        )

        # Add dynamic constraints
        opti.subject_to(P[k] == new_pos)
        opti.subject_to(R[k] == new_rvec)

    # End pose constraint
    opti.subject_to(P[-1] == goal_pos)
    opti.subject_to(R[-1] == goal_rvec)

    # Initial values
    for k in range(N):
        opti.set_initial(U[k, :], invariants_demo[k, :])

    # Minimize objective
    opti.minimize(objective)

    # Solver options
    opts = {"ipopt.print_level": 5 if verbose else 0, "print_time": verbose}
    opti.solver("ipopt", opts)

    # Solve
    sol = opti.solve()

    # Extract solution
    U_sol = sol.value(U)
    P_sol = np.array([sol.value(P[k]).flatten() for k in range(N)])
    R_sol = np.array([sol.value(R[k]).flatten() for k in range(N)])

    # Pad/trim to traj_length
    if len(P_sol) < traj_length:
        pad_n = traj_length - len(P_sol)
        P_sol = np.vstack([P_sol, np.tile(P_sol[-1], (pad_n, 1))])
        R_sol = np.vstack([R_sol, np.tile(R_sol[-1], (pad_n, 1))])
    elif len(P_sol) > traj_length:
        P_sol = P_sol[:traj_length]
        R_sol = R_sol[:traj_length]

    return {
        "invariants": U_sol,
        "positions": P_sol,
        "rvecs": R_sol,
    }


def _fallback_interpolation(
    lin_inv, ang_inv, pose_target_init, pose_target_final,
    traj_length, dhb_method, pos_resample, quat_resample, rvec_resample,
):
    """Fallback: decode + smooth interpolation when CasADi is not available."""
    decoded = decode_dhb_dr(
        lin_inv, ang_inv, pose_target_init,
        method="pos", dhb_method=dhb_method, drop_padded=False,
    )
    pos_dec = decoded["positions"]
    quat_dec = decoded["quaternions"]

    if len(pos_dec) >= traj_length:
        pos_dec = pos_dec[:traj_length].copy()
        quat_dec = quat_dec[:traj_length].copy()
    else:
        last_pos = np.tile(pos_dec[-1], (traj_length - len(pos_dec), 1))
        last_quat = np.tile(quat_dec[-1], (traj_length - len(quat_dec), 1))
        pos_dec = np.vstack([pos_dec, last_pos])
        quat_dec = np.vstack([quat_dec, last_quat])

    init_pos = np.asarray(pose_target_init["position"]).reshape(3)
    init_quat = np.asarray(pose_target_init["quaternion"]).reshape(4)
    goal_pos = np.asarray(pose_target_final["position"]).reshape(3)
    goal_quat = np.asarray(pose_target_final["quaternion"]).reshape(4)

    pos_dec[0] = init_pos.copy()
    quat_dec[0] = init_quat.copy()
    end_error = goal_pos - pos_dec[-1]

    for i in range(traj_length):
        t = i / max(1, traj_length - 1)
        s = 3 * t**2 - 2 * t**3
        pos_dec[i] = pos_dec[i] + s * end_error
        quat_dec[i] = geom.quat_slerp(quat_dec[i], goal_quat, s)

    pos_dec[0] = init_pos.copy()
    quat_dec[0] = init_quat.copy()
    pos_dec[-1] = goal_pos.copy()
    quat_dec[-1] = goal_quat.copy()

    return {
        "linear_motion_invariant": lin_inv,
        "angular_motion_invariant": ang_inv,
        "adapted_pos_data": pos_dec,
        "adapted_rvec_data": np.array([geom.quat_to_axis_angle(quat_dec[i]) for i in range(len(quat_dec))]),
        "adapted_quat_data": quat_dec,
        "resampled_pos_data": pos_resample,
        "resampled_quat_data": quat_resample,
        "resampled_rvec_data": rvec_resample,
        "solver": "interpolation",
    }
