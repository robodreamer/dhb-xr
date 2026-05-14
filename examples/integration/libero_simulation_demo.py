#!/usr/bin/env python
"""LIBERO Simulation Demo with DHB-XR.

This script demonstrates:
1. Setting up a LIBERO environment
2. Loading a demo trajectory from HDF5
3. Encoding/decoding with DHB invariants
4. Replaying the trajectory in simulation

Requirements:
    conda create -n libero python=3.10
    conda activate libero
    pip install robosuite mujoco h5py numpy
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
    cd LIBERO && pip install -e .
    pip install dhb_xr

Usage:
    python examples/integration/libero_simulation_demo.py --task_id 0 --render

Author: Andy Park
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np

# Check for LIBERO installation
try:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    HAS_LIBERO = True
except ImportError:
    HAS_LIBERO = False
    print("LIBERO not installed. Install with:")
    print("  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git")
    print("  cd LIBERO && pip install -e .")

# Check for robosuite
try:
    import robosuite
    HAS_ROBOSUITE = True
except ImportError:
    HAS_ROBOSUITE = False
    print("robosuite not installed. Install with: pip install robosuite mujoco")


def load_demo_from_hdf5(dataset_path: str, demo_id: int = 0):
    """Load a demo trajectory from LIBERO HDF5 dataset."""
    import h5py

    with h5py.File(dataset_path, "r") as f:
        demo_key = f"demo_{demo_id}"
        if demo_key not in f["data"]:
            raise ValueError(f"Demo {demo_id} not found in {dataset_path}")

        demo = f["data"][demo_key]

        # Extract actions
        actions = np.array(demo["actions"])

        # Extract end-effector pose from robot_states
        robot_states = np.array(demo["robot_states"])
        ee_pos = robot_states[:, 2:5]  # columns 2-4
        ee_quat = robot_states[:, 5:9]  # columns 5-8 (w, x, y, z)

        return {
            "actions": actions,
            "positions": ee_pos,
            "quaternions": ee_quat,
            "num_frames": len(actions),
        }


def encode_demo_with_dhb(positions: np.ndarray, quaternions: np.ndarray):
    """Encode trajectory with DHB-DR."""
    from dhb_xr.encoder.dhb_dr import encode_dhb_dr
    from dhb_xr.core.types import EncodingMethod, DHBMethod

    result = encode_dhb_dr(
        positions, quaternions,
        method=EncodingMethod.POSITION,
        dhb_method=DHBMethod.DOUBLE_REFLECTION,
    )

    return {
        "linear_invariants": result["linear_motion_invariants"],
        "angular_invariants": result["angular_motion_invariants"],
        "initial_pose": result["initial_pose"],
    }


def decode_demo_with_dhb(
    linear_inv: np.ndarray,
    angular_inv: np.ndarray,
    initial_pose: dict,
):
    """Decode DHB invariants back to trajectory."""
    from dhb_xr.decoder.dhb_dr import decode_dhb_dr
    from dhb_xr.core.types import EncodingMethod, DHBMethod

    result = decode_dhb_dr(
        linear_inv, angular_inv,
        initial_pose,
        method=EncodingMethod.POSITION,
        dhb_method=DHBMethod.DOUBLE_REFLECTION,
        drop_padded=True,
    )

    return {
        "positions": result["positions"],
        "quaternions": result["quaternions"],
    }


def run_libero_demo(task_id: int = 0, demo_id: int = 0, render: bool = False):
    """Run a LIBERO task with a demo trajectory."""

    if not HAS_LIBERO or not HAS_ROBOSUITE:
        print("\nCannot run simulation without LIBERO and robosuite.")
        print("Running DHB encoding/decoding demo only...\n")

        # Use our downloaded dataset for DHB demo
        dataset_dir = Path(os.environ.get(
            "LIBERO_DATA_DIR",
            "/home/andypark/Projects/data/libero/libero_spatial"
        ))

        if not dataset_dir.exists():
            print(f"Dataset not found at {dataset_dir}")
            print("Download with:")
            print("  wget -O libero_spatial.zip 'https://utexas.box.com/shared/static/04k94hyizn4huhbv5sz4ev9p2h1p6s7f.zip'")
            return

        # Find first HDF5 file
        hdf5_files = list(dataset_dir.glob("*.hdf5"))
        if not hdf5_files:
            print(f"No HDF5 files found in {dataset_dir}")
            return

        dataset_path = str(hdf5_files[0])
        print(f"Loading demo from: {dataset_path}")

        demo = load_demo_from_hdf5(dataset_path, demo_id)
        print(f"Loaded demo with {demo['num_frames']} frames")
        print(f"  Positions shape: {demo['positions'].shape}")
        print(f"  Quaternions shape: {demo['quaternions'].shape}")

        # Encode with DHB
        print("\nEncoding with DHB-DR...")
        encoded = encode_demo_with_dhb(demo["positions"], demo["quaternions"])
        print(f"  Linear invariants: {encoded['linear_invariants'].shape}")
        print(f"  Angular invariants: {encoded['angular_invariants'].shape}")

        # Decode back
        print("\nDecoding back to trajectory...")
        decoded = decode_demo_with_dhb(
            encoded["linear_invariants"],
            encoded["angular_invariants"],
            encoded["initial_pose"],
        )
        print(f"  Decoded positions: {decoded['positions'].shape}")
        print(f"  Decoded quaternions: {decoded['quaternions'].shape}")

        # Compute reconstruction error
        pos_error = np.linalg.norm(demo["positions"] - decoded["positions"][:len(demo["positions"])], axis=1)
        print(f"\nReconstruction error:")
        print(f"  Position: mean={pos_error.mean():.6f}m, max={pos_error.max():.6f}m")

        print("\n=== DHB encoding/decoding successful! ===")
        print("\nTo run full simulation, install LIBERO:")
        print("  conda create -n libero python=3.10")
        print("  conda activate libero")
        print("  pip install robosuite mujoco")
        print("  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git")
        print("  cd LIBERO && pip install -e .")
        return

    # ========== Full LIBERO simulation ==========
    print(f"Setting up LIBERO task {task_id}...")

    # Get benchmark and task
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_spatial"]()

    task = task_suite.get_task(task_id)
    print(f"Task: {task.name}")
    print(f"Description: {task.language}")

    # Get BDDL file for environment setup
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file
    )

    # Create environment
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 128,
        "camera_widths": 128,
    }

    if render:
        env_args["has_renderer"] = True
        env_args["has_offscreen_renderer"] = False

    env = OffScreenRenderEnv(**env_args)
    env.seed(0)

    # Get initial states for benchmarking
    init_states = task_suite.get_task_init_states(task_id)
    init_state_id = 0

    # Reset and set initial state
    env.reset()
    env.set_init_state(init_states[init_state_id])

    # Load demo actions from dataset
    dataset_path = os.path.join(
        get_libero_path("datasets"),
        "libero_spatial",
        f"{task.name}_demo.hdf5"
    )

    print(f"Loading demo from: {dataset_path}")
    demo = load_demo_from_hdf5(dataset_path, demo_id)

    # Encode with DHB
    print("Encoding demo with DHB-DR...")
    encoded = encode_demo_with_dhb(demo["positions"], demo["quaternions"])

    print(f"Running {demo['num_frames']} steps in simulation...")

    # Execute demo actions
    for step in range(demo["num_frames"]):
        action = demo["actions"][step]
        obs, reward, done, info = env.step(action)

        if render:
            env.render()

        if done:
            print(f"Task completed at step {step}!")
            break

    # Check final reward
    if reward > 0:
        print("SUCCESS: Task completed!")
    else:
        print("Task not completed within demo length.")

    env.close()
    print("\nSimulation finished.")


def main():
    parser = argparse.ArgumentParser(description="LIBERO + DHB-XR Demo")
    parser.add_argument("--task_id", type=int, default=0, help="LIBERO task ID")
    parser.add_argument("--demo_id", type=int, default=0, help="Demo ID within task")
    parser.add_argument("--render", action="store_true", help="Render simulation")
    args = parser.parse_args()

    print("=" * 60)
    print("LIBERO Simulation Demo with DHB-XR")
    print("=" * 60)
    print()

    run_libero_demo(
        task_id=args.task_id,
        demo_id=args.demo_id,
        render=args.render,
    )


if __name__ == "__main__":
    main()
