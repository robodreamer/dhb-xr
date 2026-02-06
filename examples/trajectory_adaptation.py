"""
Trajectory Adaptation Example

This example demonstrates trajectory adaptation: retargeting a demonstration
trajectory to start and end at different poses while preserving the motion shape.

The process:
1. Create a demo trajectory (40 poses with random walk)
2. Define new start and goal poses (offset from original)
3. Use DHB encoding/decoding to adapt the trajectory to the new boundary conditions
4. Show that the adapted trajectory starts/ends at the desired poses

This is the foundation for robust trajectory execution under pose perturbations.
"""

import numpy as np
from dhb_xr.optimization.casadi_solver import generate_trajectory
from dhb_xr.core.types import DHBMethod


def run_example():
    """Run the trajectory adaptation example."""
    print("DHB-XR Trajectory Adaptation Example")
    print("=" * 40)

    # Create demo trajectory
    np.random.seed(0)
    n = 40
    positions = np.cumsum(np.random.randn(n, 3) * 0.02, axis=0)
    quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))

    print(f"Demo trajectory: {n} poses")
    print(f"Original start: {positions[0]}")
    print(f"Original goal:  {positions[-1]}")

    # Define new boundary conditions
    pose_init = {"position": positions[0].copy(), "quaternion": quaternions[0].copy()}
    pose_goal = {
        "position": positions[-1].copy() + np.array([0.1, 0.1, 0.0]),
        "quaternion": quaternions[-1].copy()
    }

    print("
New goal (offset by [0.1, 0.1, 0]):"
    print(f"Target goal: {pose_goal['position']}")

    # Adapt trajectory
    print("\nAdapting trajectory...")
    result = generate_trajectory(
        positions, quaternions,
        pose_init, pose_goal,
        traj_length=30,
        smoothing=False,
        dhb_method=DHBMethod.DOUBLE_REFLECTION,
    )

    adapted_positions = result["adapted_pos_data"]
    print(f"Adapted trajectory: {len(adapted_positions)} poses")
    print(f"Adapted start: {adapted_positions[0]}")
    print(f"Adapted goal:  {adapted_positions[-1]}")

    # Check boundary condition satisfaction
    start_error = np.linalg.norm(adapted_positions[0] - pose_init["position"])
    goal_error = np.linalg.norm(adapted_positions[-1] - pose_goal["position"])

    print("
Boundary condition errors:")
    print(".2e")
    print(".2e")

    if start_error < 1e-10 and goal_error < 1e-10:
        print("✅ Perfect boundary condition satisfaction")
    else:
        print("⚠️  Boundary conditions not perfectly satisfied")

    return result


def main():
    """Main function for command-line execution."""
    run_example()


if __name__ == "__main__":
    main()
