"""
GPU Batched Trajectory Optimization Example

This example demonstrates batched trajectory optimization for processing multiple
trajectories simultaneously. Useful for:

- Large-scale trajectory adaptation datasets
- Real-time multi-trajectory optimization
- GPU-accelerated batch processing

The example:
1. Creates a demo trajectory (30 poses)
2. Defines two different start/goal pose pairs
3. Uses BatchedTrajectoryOptimizer to adapt the trajectory for both pairs
4. Shows the optimized results for both trajectories

For GPU acceleration, set device="cuda" (requires PyTorch CUDA).
"""

import numpy as np
from dhb_xr.optimization.torch_solver import BatchedTrajectoryOptimizer


def run_example():
    """Run the GPU batched trajectory optimization example."""
    print("DHB-XR GPU Batched Trajectory Optimization")
    print("=" * 45)

    # Create demo trajectory
    np.random.seed(1)
    n = 30
    demo_positions = np.cumsum(np.random.randn(n, 3) * 0.02, axis=0)
    demo_quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))

    print(f"Demo trajectory: {n} poses")
    print(f"Original start: {demo_positions[0]}")
    print(f"Original goal:  {demo_positions[-1]}")

    # Define two different start/goal pairs for batch optimization
    init_poses = [
        {
            "position": demo_positions[0].copy(),
            "quaternion": demo_quaternions[0].copy()
        },
        {
            "position": demo_positions[0].copy() + np.array([0.05, 0, 0]),
            "quaternion": demo_quaternions[0].copy()
        },
    ]
    goal_poses = [
        {
            "position": demo_positions[-1].copy(),
            "quaternion": demo_quaternions[-1].copy()
        },
        {
            "position": demo_positions[-1].copy() + np.array([0.1, 0.05, 0]),
            "quaternion": demo_quaternions[-1].copy()
        },
    ]

    print("
Batch optimization setup:")
    print(f"Trajectory 0: {init_poses[0]['position']} → {goal_poses[0]['position']}")
    print(f"Trajectory 1: {init_poses[1]['position']} → {goal_poses[1]['position']}")

    # Run batched optimization
    print("\nRunning batched optimization...")
    optimizer = BatchedTrajectoryOptimizer(device="cpu", dhb_method="double_reflection")
    adapted_pos, adapted_quat = optimizer.optimize(
        demo_positions, demo_quaternions,
        init_poses, goal_poses,
        num_steps=20,
        lr=1e-2,
    )

    print(f"✅ Optimization complete")
    print(f"Adapted positions shape: {adapted_pos.shape}")
    print(f"Adapted quaternions shape: {adapted_quat.shape}")

    print("
Results:")
    for i in range(len(init_poses)):
        start_pos = adapted_pos[i, 0]
        goal_pos = adapted_pos[i, -1]
        target_goal = goal_poses[i]['position']

        goal_error = np.linalg.norm(goal_pos - target_goal)
        print(f"Trajectory {i}:")
        print(f"  Start: {start_pos}")
        print(f"  Goal:  {goal_pos}")
        print(".2e")

    return adapted_pos, adapted_quat


def main():
    """Main function for command-line execution."""
    run_example()


if __name__ == "__main__":
    main()
