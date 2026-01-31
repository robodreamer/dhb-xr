"""Trajectory adaptation example (resample + decode with new initial pose)."""

import numpy as np
from dhb_xr.optimization.casadi_solver import generate_trajectory
from dhb_xr.core.types import DHBMethod

def main():
    np.random.seed(0)
    n = 40
    positions = np.cumsum(np.random.randn(n, 3) * 0.02, axis=0)
    quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))

    pose_init = {"position": positions[0].copy(), "quaternion": quaternions[0].copy()}
    pose_goal = {"position": positions[-1].copy() + np.array([0.1, 0.1, 0.0]), "quaternion": quaternions[-1].copy()}

    result = generate_trajectory(
        positions, quaternions,
        pose_init, pose_goal,
        traj_length=30,
        smoothing=False,
        dhb_method=DHBMethod.DOUBLE_REFLECTION,
    )
    print("Adapted positions shape:", result["adapted_pos_data"].shape)
    print("First pose:", result["adapted_pos_data"][0])
    print("Last pose:", result["adapted_pos_data"][-1])

if __name__ == "__main__":
    main()
