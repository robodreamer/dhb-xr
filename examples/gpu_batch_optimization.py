"""GPU / batched trajectory optimization example."""

import numpy as np
from dhb_xr.optimization.torch_solver import BatchedTrajectoryOptimizer
from dhb_xr.core.types import DHBMethod

def main():
    np.random.seed(1)
    n = 30
    demo_positions = np.cumsum(np.random.randn(n, 3) * 0.02, axis=0)
    demo_quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))

    # Two different start/goal pairs
    init_poses = [
        {"position": demo_positions[0].copy(), "quaternion": demo_quaternions[0].copy()},
        {"position": demo_positions[0].copy() + np.array([0.05, 0, 0]), "quaternion": demo_quaternions[0].copy()},
    ]
    goal_poses = [
        {"position": demo_positions[-1].copy(), "quaternion": demo_quaternions[-1].copy()},
        {"position": demo_positions[-1].copy() + np.array([0.1, 0.05, 0]), "quaternion": demo_quaternions[-1].copy()},
    ]

    optimizer = BatchedTrajectoryOptimizer(device="cpu", dhb_method="double_reflection")
    adapted_pos, adapted_quat = optimizer.optimize(
        demo_positions, demo_quaternions,
        init_poses, goal_poses,
        num_steps=20,
        lr=1e-2,
    )
    print("Batched optimization: 2 trajectories")
    print("Adapted positions shape:", adapted_pos.shape)
    print("Trajectory 0 last pose:", adapted_pos[0, -1])
    print("Trajectory 1 last pose:", adapted_pos[1, -1])

if __name__ == "__main__":
    main()
