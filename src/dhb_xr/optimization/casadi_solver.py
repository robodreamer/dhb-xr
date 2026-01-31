"""
Trajectory adaptation: resample demo and optionally optimize invariants for new boundaries.
Full CasADi optimization requires casadi + spatial_casadi (pip install dhb_xr[optimization]).
Without CasADi, generate_trajectory returns resampled + decoded with target initial pose only.
"""

import numpy as np
from typing import Dict, Any

from dhb_xr.encoder.dhb_dr import encode_dhb_dr
from dhb_xr.decoder.dhb_dr import decode_dhb_dr
from dhb_xr.core.types import DHBMethod
from dhb_xr.core import geometry as geom
from dhb_xr.utils.resampling import resample_and_smooth


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
) -> Dict[str, Any]:
    """
    Generate trajectory by resampling demo and decoding with target initial pose.
    Quaternions in wxyz. Optional: with CasADi installed, can optimize invariants
    to match both boundary poses (see dhb_xr[optimization]).
    """
    pos_orig, quat_orig, rvec_orig, pos_resample, quat_resample, rvec_resample = resample_and_smooth(
        pos_data, quat_data, traj_length, smoothing
    )
    invariants_orig = encode_dhb_dr(
        pos_orig, quat_orig,
        init_pose=pose_target_init,
        method="pos",
        use_default_initial_frames=False,
        dhb_method=dhb_method,
    )
    invariant_size = 4 if dhb_method == DHBMethod.DOUBLE_REFLECTION else 3
    # Decode with target initial pose (retargeting start only; end may not match pose_target_final)
    decoded = decode_dhb_dr(
        invariants_orig["linear_motion_invariants"],
        invariants_orig["angular_motion_invariants"],
        pose_target_init,
        method="pos",
        dhb_method=dhb_method,
        drop_padded=True,
    )
    return {
        "linear_motion_invariant": invariants_orig["linear_motion_invariants"],
        "angular_motion_invariant": invariants_orig["angular_motion_invariants"],
        "adapted_pos_data": decoded["positions"],
        "adapted_rvec_data": np.array([geom.quat_to_axis_angle(decoded["quaternions"][i]) for i in range(len(decoded["quaternions"]))]),
        "adapted_quat_data": decoded["quaternions"],
        "resampled_pos_data": pos_resample,
        "resampled_quat_data": quat_resample,
        "resampled_rvec_data": rvec_resample,
    }
