"""Batched trajectory optimizer (scipy/numpy; PyTorch optional for future autodiff)."""

import numpy as np
from typing import List, Dict, Any
from scipy.optimize import minimize

from dhb_xr.encoder.dhb_dr import encode_dhb_dr
from dhb_xr.decoder.dhb_dr import decode_dhb_dr
from dhb_xr.core.types import DHBMethod
from dhb_xr.core import geometry as geom

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class BatchedTrajectoryOptimizer:
    """
    Optimize invariants so decoded trajectory matches goal pose(s).
    Uses scipy.optimize; device is ignored (numpy backend).
    """

    def __init__(self, device: str = "cpu", dhb_method: str = "double_reflection"):
        self.device = device
        self.dhb_method = DHBMethod.DOUBLE_REFLECTION if dhb_method == "double_reflection" else DHBMethod.ORIGINAL
        self.k = 4 if self.dhb_method == DHBMethod.DOUBLE_REFLECTION else 3

    def optimize(
        self,
        demo_positions: np.ndarray,
        demo_quaternions: np.ndarray,
        init_poses: List[Dict[str, np.ndarray]],
        goal_poses: List[Dict[str, np.ndarray]],
        num_steps: int = 100,
        lr: float = 1e-2,
    ) -> tuple:
        """
        For each batch item: optimize U so decode(U, init_pose) ends at goal_pose.
        Returns (adapted_positions, adapted_quaternions) (B, N', 3), (B, N', 4).
        """
        out = encode_dhb_dr(
            demo_positions, demo_quaternions,
            method="pos", use_default_initial_frames=True, dhb_method=self.dhb_method,
        )
        U_demo = np.concatenate([
            out["linear_motion_invariants"],
            out["angular_motion_invariants"],
        ], axis=1)
        B = len(init_poses)
        n_inv, total_dim = U_demo.shape[0], U_demo.shape[1]
        pos_list = []
        quat_list = []
        for b in range(B):
            init_pos = np.asarray(init_poses[b]["position"]).reshape(3)
            init_quat = np.asarray(init_poses[b]["quaternion"]).reshape(4)
            goal_pos = np.asarray(goal_poses[b]["position"]).reshape(3)
            goal_quat = np.asarray(goal_poses[b]["quaternion"]).reshape(4)

            def loss(u_flat):
                U = u_flat.reshape(n_inv, total_dim)
                lin, ang = U[:, : self.k], U[:, self.k :]
                decoded = decode_dhb_dr(
                    lin, ang, {"position": init_pos, "quaternion": init_quat},
                    method="pos", dhb_method=self.dhb_method, drop_padded=True,
                )
                pos = decoded["positions"]
                quat = decoded["quaternions"]
                loss_p = np.sum((pos[-1] - goal_pos) ** 2)
                R_diff = geom.quat_to_rot(goal_quat).T @ geom.quat_to_rot(quat[-1])
                rvec = geom.rot_to_axis_angle(R_diff)
                loss_r = np.sum(rvec ** 2)
                return loss_p + loss_r

            res = minimize(loss, U_demo.ravel(), method="L-BFGS-B", options={"maxiter": num_steps})
            U_opt = res.x.reshape(n_inv, total_dim)
            lin, ang = U_opt[:, : self.k], U_opt[:, self.k :]
            decoded = decode_dhb_dr(
                lin, ang, {"position": init_pos, "quaternion": init_quat},
                method="pos", dhb_method=self.dhb_method, drop_padded=True,
            )
            pos_list.append(decoded["positions"])
            quat_list.append(decoded["quaternions"])
        return np.stack(pos_list), np.stack(quat_list)
