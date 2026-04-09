#!/usr/bin/env python
"""Test waypoint PD controller on LIBERO simulation.

This script validates the WaypointPDController by:
1. Loading a LIBERO demo and extracting the EE trajectory from obs.
2. Replaying the SAME trajectory as waypoints (0 cm offset) -- should
   succeed near 100% since it's just tracking the original demo path.
3. Replaying a DHB-adapted trajectory with a 10 cm position offset to
   test robustness to spatial perturbation.
4. Printing success rate and per-trial metrics.

Requirements:
    MUJOCO_GL=egl ~/miniforge3/envs/libero/bin/python \
        examples/integration/test_waypoint_controller.py

Author: Andy Park
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# ---- paths for LIBERO and dhb_xr ----
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, "/home/andypark/Projects/repos/LIBERO")
sys.path.insert(0, "/home/andypark/Projects/repos/LIBERO-PRO")

import numpy as np

# ---- LIBERO imports ----
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# ---- DHB-XR imports ----
from dhb_xr.integration.vla.waypoint_controller import WaypointPDController

# ===========================================================================
# Helpers
# ===========================================================================

def load_demo_trajectory(dataset_path: str, demo_id: int = 0) -> dict:
    """Load EE trajectory + actions from a LIBERO HDF5 demo."""
    import h5py

    with h5py.File(dataset_path, "r") as f:
        demo_key = f"demo_{demo_id}"
        if demo_key not in f["data"]:
            raise ValueError(f"Demo {demo_id} not found in {dataset_path}")

        demo = f["data"][demo_key]
        actions = np.array(demo["actions"])

        # EE position from obs
        ee_pos = np.array(demo["obs"]["ee_pos"])  # (N, 3)

        # EE orientation: robot_states[:, 5:9] is (w,x,y,z)
        # Convert to xyzw for scipy / robosuite convention
        robot_states = np.array(demo["robot_states"])
        ee_quat_wxyz = robot_states[:, 5:9]
        ee_quat_xyzw = ee_quat_wxyz[:, [1, 2, 3, 0]]

        # Gripper actions (last dim of actions)
        gripper = actions[:, -1]

    return {
        "ee_pos": ee_pos,
        "ee_quat_xyzw": ee_quat_xyzw,
        "gripper": gripper,
        "actions": actions,
        "num_frames": len(actions),
    }


def create_env(task_id: int = 0, camera_size: int = 128) -> tuple:
    """Create LIBERO environment and return (env, task, init_states)."""
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_spatial"]()
    task = task_suite.get_task(task_id)

    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )

    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": camera_size,
        "camera_widths": camera_size,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)

    init_states = task_suite.get_task_init_states(task_id)
    return env, task, init_states


def reset_env(env, init_states, init_state_id: int = 0) -> dict:
    """Reset environment to a specific initial state, return obs."""
    env.reset()
    obs = env.set_init_state(init_states[init_state_id])
    return obs


def apply_offset(positions: np.ndarray, offset_m: float) -> np.ndarray:
    """Apply a uniform offset (meters) to all XYZ positions.

    The offset is applied along the x-axis to simulate a spatial
    perturbation (e.g., object or robot base shifted).
    """
    offset = np.zeros(3)
    offset[0] = offset_m  # shift along x
    return positions + offset


# ===========================================================================
# Evaluation
# ===========================================================================

def evaluate_waypoint_controller(
    env,
    init_states,
    demo: dict,
    offset_m: float = 0.0,
    num_trials: int = 1,
    init_state_id: int = 0,
    Kp_pos: float = 8.0,
    Kp_ori: float = 3.0,
    waypoint_threshold: float = 0.015,
    max_steps_per_waypoint: int = 20,
    verbose: bool = True,
) -> dict:
    """Run the waypoint controller and collect metrics."""

    ctrl = WaypointPDController(
        Kp_pos=Kp_pos,
        Kp_ori=Kp_ori,
        waypoint_threshold=waypoint_threshold,
        max_steps_per_waypoint=max_steps_per_waypoint,
    )

    # Prepare waypoints (with optional offset)
    wp_pos = demo["ee_pos"].copy()
    if offset_m != 0.0:
        wp_pos = apply_offset(wp_pos, offset_m)
    wp_ori = demo["ee_quat_xyzw"].copy()
    wp_grip = demo["gripper"].copy()

    results = []
    for trial in range(num_trials):
        obs = reset_env(env, init_states, init_state_id)

        t0 = time.time()
        result = ctrl.execute_trajectory(
            env,
            waypoint_positions=wp_pos,
            waypoint_orientations=wp_ori,
            gripper_commands=wp_grip,
            obs_pos_key="ee_pos",
            obs_ori_key="ee_states",
            obs_ori_slice=(3, 7),
            initial_obs=obs,
            verbose=verbose,
        )
        elapsed = time.time() - t0

        result["trial"] = trial
        result["elapsed_s"] = elapsed
        results.append(result)

        if verbose:
            reached_pct = (
                result["waypoints_reached"] / result["total_waypoints"] * 100
            )
            print(
                f"  Trial {trial}: reward={result['final_reward']:.2f}, "
                f"steps={result['total_steps']}, "
                f"waypoints={result['waypoints_reached']}/{result['total_waypoints']} "
                f"({reached_pct:.0f}%), "
                f"time={elapsed:.1f}s"
            )

    # Aggregate
    rewards = [r["final_reward"] for r in results]
    successes = [1 if r["final_reward"] > 0 else 0 for r in results]
    total_steps = [r["total_steps"] for r in results]

    return {
        "trials": results,
        "success_rate": np.mean(successes),
        "mean_reward": np.mean(rewards),
        "mean_steps": np.mean(total_steps),
        "num_trials": num_trials,
    }


# ===========================================================================
# Baseline: replay original actions
# ===========================================================================

def evaluate_action_replay(
    env,
    init_states,
    demo: dict,
    init_state_id: int = 0,
    verbose: bool = True,
) -> dict:
    """Baseline: replay original demo actions directly."""
    obs = reset_env(env, init_states, init_state_id)

    total_steps = 0
    final_reward = 0.0
    for step in range(demo["num_frames"]):
        action = demo["actions"][step]
        obs, reward, done, info = env.step(action)
        total_steps += 1
        final_reward = reward
        if done:
            break

    success = 1 if final_reward > 0 else 0
    if verbose:
        print(
            f"  Action replay: reward={final_reward:.2f}, "
            f"steps={total_steps}, success={bool(success)}"
        )
    return {
        "final_reward": final_reward,
        "total_steps": total_steps,
        "success": success,
    }


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test WaypointPDController on LIBERO"
    )
    parser.add_argument(
        "--task_id", type=int, default=0, help="LIBERO spatial task ID"
    )
    parser.add_argument(
        "--demo_id", type=int, default=0, help="Demo ID within task dataset"
    )
    parser.add_argument(
        "--offset_m",
        type=float,
        default=0.10,
        help="Position offset in meters for adapted trajectory test",
    )
    parser.add_argument(
        "--num_trials", type=int, default=1, help="Number of trials per test"
    )
    parser.add_argument(
        "--Kp_pos", type=float, default=8.0, help="Position P gain"
    )
    parser.add_argument(
        "--Kp_ori", type=float, default=3.0, help="Orientation P gain"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress per-step output"
    )
    args = parser.parse_args()

    verbose = not args.quiet

    print("=" * 65)
    print("  Waypoint PD Controller -- LIBERO Simulation Test")
    print("=" * 65)
    print()

    # --- Create environment ---
    print(f"Creating LIBERO environment (task_id={args.task_id})...")
    env, task, init_states = create_env(task_id=args.task_id)
    print(f"  Task: {task.name}")
    print(f"  Language: {task.language}")
    print()

    # --- Load demo ---
    dataset_path = os.path.join(
        get_libero_path("datasets"),
        "libero_spatial",
        f"{task.name}_demo.hdf5",
    )
    print(f"Loading demo {args.demo_id} from: {dataset_path}")
    demo = load_demo_trajectory(dataset_path, demo_id=args.demo_id)
    print(f"  Frames: {demo['num_frames']}")
    print(f"  EE pos range: {demo['ee_pos'].min(axis=0)} -> {demo['ee_pos'].max(axis=0)}")
    print()

    # === Test 1: Baseline action replay ===
    print("-" * 65)
    print("Test 1: Baseline -- replay original demo actions")
    print("-" * 65)
    baseline = evaluate_action_replay(
        env, init_states, demo, verbose=verbose
    )
    print()

    # === Test 2: Waypoint controller at 0 cm offset (identity test) ===
    print("-" * 65)
    print("Test 2: Waypoint controller -- 0 cm offset (track original demo)")
    print("-" * 65)
    result_0cm = evaluate_waypoint_controller(
        env,
        init_states,
        demo,
        offset_m=0.0,
        num_trials=args.num_trials,
        Kp_pos=args.Kp_pos,
        Kp_ori=args.Kp_ori,
        verbose=verbose,
    )
    print()

    # === Test 3: Waypoint controller at N cm offset ===
    offset_cm = args.offset_m * 100
    print("-" * 65)
    print(f"Test 3: Waypoint controller -- {offset_cm:.0f} cm offset")
    print("-" * 65)
    result_offset = evaluate_waypoint_controller(
        env,
        init_states,
        demo,
        offset_m=args.offset_m,
        num_trials=args.num_trials,
        Kp_pos=args.Kp_pos,
        Kp_ori=args.Kp_ori,
        verbose=verbose,
    )
    print()

    # === Summary ===
    print("=" * 65)
    print("  Summary")
    print("=" * 65)
    print(f"  Baseline (action replay):     reward={baseline['final_reward']:.2f}, "
          f"success={bool(baseline['success'])}")
    print(f"  Waypoint ctrl (0 cm):         success_rate={result_0cm['success_rate']*100:.0f}%, "
          f"mean_reward={result_0cm['mean_reward']:.2f}, "
          f"mean_steps={result_0cm['mean_steps']:.0f}")
    print(f"  Waypoint ctrl ({offset_cm:.0f} cm offset): "
          f"success_rate={result_offset['success_rate']*100:.0f}%, "
          f"mean_reward={result_offset['mean_reward']:.2f}, "
          f"mean_steps={result_offset['mean_steps']:.0f}")
    print("=" * 65)

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
