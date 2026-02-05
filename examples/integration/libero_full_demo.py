#!/usr/bin/env python
"""LIBERO / LIBERO-PRO Full Simulation Demo with DHB-XR.

This script demonstrates the complete DHB-XR + LIBERO integration:

1. Loading demo trajectories from LIBERO HDF5 datasets
2. Encoding trajectories to SE(3)-invariant DHB representations
3. Trajectory adaptation (retargeting to new initial poses)
4. Running adapted trajectories in the LIBERO simulation
5. Motion retrieval using the DHB database

Compatible with both LIBERO and LIBERO-PRO (drop-in replacement with perturbation support).
For perturbation robustness testing, see libero_pro_dhb_demo.py.

Requirements:
    # In a conda environment (libero):
    source ~/miniforge3/bin/activate libero

    # Or create new:
    conda create -n libero python=3.10
    conda activate libero
    pip install robosuite==1.4.0 mujoco bddl==1.0.1 robomimic==0.2.0
    pip install future easydict hydra-core cloudpickle gym==0.25.2
    # LIBERO-PRO (recommended, includes perturbation benchmarks):
    git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git
    cd LIBERO-PRO && pip install -e . --config-settings editable_mode=compat
    pip install dhb_xr

Usage:
    # Run with libero conda environment
    ~/miniforge3/bin/mamba run -n libero python examples/integration/libero_full_demo.py

    # With rendering (requires display)
    ~/miniforge3/bin/mamba run -n libero python examples/integration/libero_full_demo.py --render

    # Specific task
    ~/miniforge3/bin/mamba run -n libero python examples/integration/libero_full_demo.py --task_id 3

Author: Andy Park
Date: 2026-01-30
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np

# -----------------------------------------------------------------------------
# Check dependencies
# -----------------------------------------------------------------------------

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    from libero.libero.benchmark import get_benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    HAS_LIBERO = True
except ImportError:
    HAS_LIBERO = False

try:
    import robosuite
    HAS_ROBOSUITE = True
except ImportError:
    HAS_ROBOSUITE = False

# DHB-XR imports
from dhb_xr.encoder.dhb_dr import encode_dhb_dr
from dhb_xr.decoder.dhb_dr import decode_dhb_dr
from dhb_xr.core.types import EncodingMethod, DHBMethod
from dhb_xr.core.geometry import rot_to_quat, quat_to_rot

# Solver availability
try:
    from dhb_xr.optimization.fatrop_solver import generate_trajectory_fatrop
    HAS_FATROP = True
except ImportError:
    HAS_FATROP = False

try:
    from dhb_xr.optimization.casadi_solver import generate_trajectory
    HAS_CASADI_SOLVER = True
except ImportError:
    HAS_CASADI_SOLVER = False


# -----------------------------------------------------------------------------
# Data loading utilities
# -----------------------------------------------------------------------------

def load_demo_from_hdf5(dataset_path: str, demo_id: int = 0):
    """Load a demo trajectory from LIBERO HDF5 dataset.

    LIBERO stores robot states as:
        robot_states[:, 0:2]  - gripper state
        robot_states[:, 2:5]  - end-effector position
        robot_states[:, 5:9]  - end-effector quaternion (w, x, y, z)
    """
    with h5py.File(dataset_path, "r") as f:
        demo_key = f"demo_{demo_id}"
        if demo_key not in f["data"]:
            available = [k for k in f["data"].keys() if k.startswith("demo_")]
            raise ValueError(f"Demo {demo_id} not found. Available: {available[:5]}...")

        demo = f["data"][demo_key]

        # Extract actions
        actions = np.array(demo["actions"])

        # Extract end-effector pose from robot_states
        robot_states = np.array(demo["robot_states"])
        ee_pos = robot_states[:, 2:5].astype(np.float64)
        ee_quat_wxyz = robot_states[:, 5:9].astype(np.float64)

        # Convert from (w, x, y, z) to (x, y, z, w) format
        ee_quat = ee_quat_wxyz[:, [1, 2, 3, 0]]

        # Normalize quaternions
        ee_quat = ee_quat / np.linalg.norm(ee_quat, axis=1, keepdims=True)

        return {
            "actions": actions,
            "positions": ee_pos,
            "quaternions": ee_quat,
            "num_frames": len(actions),
            "task_name": Path(dataset_path).stem,
        }


def list_demos_in_dataset(dataset_path: str):
    """List all demo IDs in a dataset."""
    with h5py.File(dataset_path, "r") as f:
        demos = sorted([k for k in f["data"].keys() if k.startswith("demo_")])
        return [int(d.split("_")[1]) for d in demos]


# -----------------------------------------------------------------------------
# DHB encoding/decoding
# -----------------------------------------------------------------------------

def encode_trajectory(positions: np.ndarray, quaternions: np.ndarray):
    """Encode trajectory to DHB-DR invariants."""
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


def decode_trajectory(linear_inv: np.ndarray, angular_inv: np.ndarray, initial_pose: dict):
    """Decode DHB invariants back to trajectory."""
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


def adapt_trajectory(
    positions: np.ndarray,
    quaternions: np.ndarray,
    new_start_pos: np.ndarray,
    new_start_quat: np.ndarray,
):
    """Adapt trajectory to a new starting pose using solver-based optimization.

    Uses Fatrop (preferred) or CasADi/IPOPT to find an adapted trajectory
    whose DHB invariants stay close to the demo while satisfying boundary
    constraints. Falls back to simple encode-decode if no solver is available.

    Solver priority: Fatrop (~7ms) > CasADi/IPOPT (~50ms) > encode-decode
    """
    # Goal: shift by same offset as start
    offset = new_start_pos - positions[0]
    new_goal_pos = positions[-1] + offset
    new_goal_quat = quaternions[-1].copy()
    traj_length = min(len(positions), 50)

    pose_init = {"position": new_start_pos.copy(), "quaternion": new_start_quat.copy()}
    pose_goal = {"position": new_goal_pos.copy(), "quaternion": new_goal_quat.copy()}

    # Try Fatrop first (fastest)
    if HAS_FATROP:
        try:
            result = generate_trajectory_fatrop(
                positions, quaternions,
                pose_target_init=pose_init,
                pose_target_final=pose_goal,
                traj_length=traj_length,
                use_fatrop=True,
                verbose=False,
            )
            if result.get("success", False):
                return {
                    "positions": result["positions"],
                    "quaternions": result["quaternions"],
                    "solver": "fatrop",
                }
        except Exception:
            pass

    # Try CasADi/IPOPT
    if HAS_CASADI_SOLVER:
        try:
            result = generate_trajectory(
                positions, quaternions,
                pose_target_init=pose_init,
                pose_target_final=pose_goal,
                traj_length=traj_length,
                dhb_method=DHBMethod.DOUBLE_REFLECTION,
                use_casadi=True,
                verbose=False,
            )
            if result.get("solver") == "casadi":
                return {
                    "positions": result["adapted_pos_data"],
                    "quaternions": result["adapted_quat_data"],
                    "solver": "casadi",
                }
        except Exception:
            pass

    # Fallback: pure encode-decode (no goal constraint)
    encoded = encode_trajectory(positions, quaternions)
    new_initial_pose = {
        "position": new_start_pos.copy(),
        "quaternion": new_start_quat.copy(),
    }
    adapted = decode_trajectory(
        encoded["linear_invariants"],
        encoded["angular_invariants"],
        new_initial_pose,
    )
    return {
        "positions": adapted["positions"],
        "quaternions": adapted["quaternions"],
        "solver": "decode",
    }


# -----------------------------------------------------------------------------
# Demo functions
# -----------------------------------------------------------------------------

def run_dhb_only_demo():
    """Run DHB encoding/decoding demo without simulation."""
    print("\n" + "=" * 60)
    print("DHB-XR Encoding/Decoding Demo (No Simulation)")
    print("=" * 60)

    # Find dataset
    dataset_dir = Path(os.environ.get(
        "LIBERO_DATA_DIR",
        "/home/andypark/Projects/data/libero/libero_spatial"
    ))

    if not dataset_dir.exists():
        print(f"\nDataset not found at: {dataset_dir}")
        print("\nDownload LIBERO-Spatial dataset:")
        print("  mkdir -p ~/Projects/data/libero && cd ~/Projects/data/libero")
        print("  wget -O libero_spatial.zip 'https://utexas.box.com/shared/static/04k94hyizn4huhbv5sz4ev9p2h1p6s7f.zip'")
        print("  unzip libero_spatial.zip")
        return False

    # Find HDF5 files
    hdf5_files = sorted(dataset_dir.glob("*.hdf5"))
    if not hdf5_files:
        print(f"No HDF5 files found in {dataset_dir}")
        return False

    print(f"\nFound {len(hdf5_files)} task files")
    print(f"Using: {hdf5_files[0].name}")

    # Load demo
    demo = load_demo_from_hdf5(str(hdf5_files[0]), demo_id=0)
    print(f"\nLoaded demo with {demo['num_frames']} frames")
    print(f"  Position range: [{demo['positions'].min():.3f}, {demo['positions'].max():.3f}]")

    # Encode
    print("\n--- Encoding with DHB-DR ---")
    encoded = encode_trajectory(demo["positions"], demo["quaternions"])
    print(f"Linear invariants shape: {encoded['linear_invariants'].shape}")
    print(f"Angular invariants shape: {encoded['angular_invariants'].shape}")

    # Decode (reconstruction test)
    print("\n--- Decoding (reconstruction) ---")
    decoded = decode_trajectory(
        encoded["linear_invariants"],
        encoded["angular_invariants"],
        encoded["initial_pose"],
    )

    # Compute reconstruction error
    N = min(len(demo["positions"]), len(decoded["positions"]))
    pos_error = np.linalg.norm(demo["positions"][:N] - decoded["positions"][:N], axis=1)
    print(f"Position reconstruction error:")
    print(f"  Mean: {pos_error.mean() * 1000:.3f} mm")
    print(f"  Max:  {pos_error.max() * 1000:.3f} mm")

    # Trajectory adaptation demo
    print("\n--- Trajectory Adaptation (solver-based) ---")
    print(f"Solver availability: Fatrop={'YES' if HAS_FATROP else 'NO'}, CasADi={'YES' if HAS_CASADI_SOLVER else 'NO'}")
    print("Adapting trajectory to a new starting pose...")

    # Create perturbed starting pose
    new_start_pos = demo["positions"][0] + np.array([0.05, 0.02, 0.01])
    new_start_quat = demo["quaternions"][0]  # Same orientation

    adapted = adapt_trajectory(
        demo["positions"], demo["quaternions"],
        new_start_pos, new_start_quat,
    )

    solver_used = adapted.get("solver", "unknown")
    print(f"Solver used: {solver_used}")
    print(f"Original start: {demo['positions'][0]}")
    print(f"Adapted start:  {adapted['positions'][0]}")
    print(f"Offset applied: {new_start_pos - demo['positions'][0]}")

    # Verify the shape is preserved
    orig_displacement = demo["positions"][-1] - demo["positions"][0]
    adapted_displacement = adapted["positions"][-1] - adapted["positions"][0]
    shape_error = np.linalg.norm(orig_displacement - adapted_displacement)
    print(f"\nTrajectory shape preserved (displacement error): {shape_error * 1000:.3f} mm")

    # Visualization
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(12, 5))

        # 3D trajectory plot
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(demo["positions"][:, 0], demo["positions"][:, 1], demo["positions"][:, 2],
                 'b-', linewidth=2, label='Original')
        ax1.plot(adapted["positions"][:, 0], adapted["positions"][:, 1], adapted["positions"][:, 2],
                 'r--', linewidth=2, label='Adapted')
        ax1.scatter(*demo["positions"][0], c='blue', s=100, marker='o', label='Original start')
        ax1.scatter(*adapted["positions"][0], c='red', s=100, marker='o', label='Adapted start')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('Trajectory Adaptation Demo')
        ax1.legend()

        # Invariants plot
        ax2 = fig.add_subplot(122)
        ax2.plot(encoded["linear_invariants"][:, 0], label='Linear magnitude', linewidth=2)
        ax2.plot(encoded["angular_invariants"][:, 0], label='Angular magnitude', linewidth=2)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Invariant value')
        ax2.set_title('DHB Invariants')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/tmp/dhb_demo_plot.png', dpi=150)
        print(f"\nPlot saved to: /tmp/dhb_demo_plot.png")

        # Show if display available
        try:
            plt.show(block=False)
            plt.pause(3)  # Show for 3 seconds
            plt.close()
        except:
            pass

    except ImportError:
        print("\nMatplotlib not available for visualization. Install with: pip install matplotlib")

    print("\n" + "=" * 60)
    print("DHB demo completed successfully!")
    print("=" * 60)

    return True


def run_simulation_demo(task_id: int = 0, demo_id: int = 0, render: bool = False, save_video: str = None):
    """Run full LIBERO simulation with DHB-XR."""
    print("\n" + "=" * 60)
    print("LIBERO Simulation Demo with DHB-XR")
    print("=" * 60)

    # Get benchmark
    benchmark = get_benchmark("libero_spatial")()
    print(f"\nLIBERO-Spatial benchmark: {benchmark.n_tasks} tasks")

    # Get task info
    task = benchmark.get_task(task_id)
    print(f"\nTask {task_id}: {task.name}")

    # Get BDDL file for environment setup
    task_bddl_file = benchmark.get_task_bddl_file_path(task_id)
    print(f"BDDL file: {task_bddl_file}")

    # Create environment
    print("\nCreating environment...")

    cv2_available = False

    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 256,  # Higher resolution for display/video
        "camera_widths": 256,
    }

    env = OffScreenRenderEnv(**env_args)

    # For live display, use OpenCV to show camera frames
    # (LIBERO's OffScreenRenderEnv doesn't support native MuJoCo viewer)
    if render:
        try:
            import cv2
            cv2_available = True
            print("Using OpenCV for real-time display")
            print("(Press 'q' to quit, window shows agentview camera)")
        except ImportError:
            print("OpenCV not available for live display.")
            print("Install with: pip install opencv-python")
            print("Or use --save-video to record instead.")
    print("Environment created!")

    # Reset environment
    obs = env.reset()
    print(f"\nObservation keys: {list(obs.keys())[:5]}...")

    # Get initial end-effector pose from observation
    initial_ee_pos = obs["robot0_eef_pos"]
    initial_ee_quat = obs["robot0_eef_quat"]  # (x, y, z, w) format in robosuite
    print(f"Initial EE position: {initial_ee_pos}")
    print(f"Initial EE quaternion: {initial_ee_quat}")

    # Load demo trajectory from dataset
    datasets_path = get_libero_path("datasets")
    dataset_file = f"{task.name}_demo.hdf5"
    dataset_path = os.path.join(datasets_path, "libero_spatial", dataset_file)

    # Check custom dataset location
    if not os.path.exists(dataset_path):
        dataset_path = f"/home/andypark/Projects/data/libero/libero_spatial/{dataset_file}"

    print(f"\nLoading demo from: {dataset_path}")

    if not os.path.exists(dataset_path):
        print(f"Dataset file not found: {dataset_path}")
        env.close()
        return False

    demo = load_demo_from_hdf5(dataset_path, demo_id)
    print(f"Demo loaded: {demo['num_frames']} frames")

    # Encode demo trajectory
    print("\n--- DHB Encoding ---")
    encoded = encode_trajectory(demo["positions"], demo["quaternions"])
    print(f"Encoded to {encoded['linear_invariants'].shape[0]} invariant frames")

    # Execute demo actions in simulation
    print(f"\n--- Executing {demo['num_frames']} actions ---")

    total_reward = 0
    success = False
    frames = []  # For video recording
    quit_requested = False

    for step in range(demo["num_frames"]):
        action = demo["actions"][step]
        obs, reward, done, info = env.step(action)

        total_reward += reward

        # Get image from agentview camera (LIBERO flips images vertically)
        frame = obs.get("agentview_image", None)
        if frame is not None:
            frame = frame[::-1]  # Flip vertically (LIBERO convention)

        # Display with OpenCV
        if render and cv2_available and frame is not None:
            import cv2
            # Convert RGB to BGR for OpenCV
            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # Resize for better visibility
            display_frame = cv2.resize(display_frame, (512, 512))
            cv2.imshow("LIBERO Simulation (press 'q' to quit)", display_frame)
            key = cv2.waitKey(30)  # 30ms delay (~33 fps)
            if key == ord('q'):
                print("\n  Quit requested by user")
                quit_requested = True
                break

        # Capture frame for video
        if save_video and frame is not None:
            frames.append(frame)

        if step % 20 == 0:
            current_ee = obs["robot0_eef_pos"]
            print(f"  Step {step:3d}: EE pos = [{current_ee[0]:.3f}, {current_ee[1]:.3f}, {current_ee[2]:.3f}]")

        if done or reward > 0:
            success = reward > 0
            print(f"\n  Task {'COMPLETED' if success else 'ended'} at step {step}!")
            break

    # Close OpenCV window
    if render and cv2_available:
        import cv2
        cv2.destroyAllWindows()

    # Save video if requested (using imageio like LIBERO does)
    if save_video and frames:
        try:
            import imageio
            video_path = save_video if save_video.endswith('.mp4') else f"{save_video}.mp4"
            print(f"\nSaving video to: {video_path}")
            # Use fps=30 for smooth playback (LIBERO uses 60 for demos, 30 is fine for viewing)
            video_writer = imageio.get_writer(video_path, fps=30)
            for frame in frames:
                video_writer.append_data(frame)
            video_writer.close()
            print(f"Video saved! ({len(frames)} frames, {len(frames)/30:.1f}s at 30fps)")
        except ImportError:
            print("\nCannot save video: imageio not installed. Run: pip install imageio imageio-ffmpeg")

    # Final status
    print("\n" + "-" * 40)
    if success:
        print("SUCCESS! Task completed.")
    else:
        print("Task not completed within demo length.")

    final_ee = obs["robot0_eef_pos"]
    print(f"Final EE position: [{final_ee[0]:.3f}, {final_ee[1]:.3f}, {final_ee[2]:.3f}]")

    env.close()
    print("\nSimulation finished.")

    return success


def run_motion_retrieval_demo():
    """Demonstrate motion retrieval using DHB database."""
    print("\n" + "=" * 60)
    print("Motion Retrieval Demo with DHB-XR")
    print("=" * 60)

    from dhb_xr.database.motion_db import MotionDatabase

    # Find dataset
    dataset_dir = Path("/home/andypark/Projects/data/libero/libero_spatial")
    if not dataset_dir.exists():
        print(f"Dataset not found at {dataset_dir}")
        return False

    hdf5_files = sorted(dataset_dir.glob("*.hdf5"))[:5]  # Use first 5 tasks
    if len(hdf5_files) < 2:
        print("Need at least 2 task files for retrieval demo")
        return False

    print(f"\nBuilding database from {len(hdf5_files)} task files...")

    # Build database
    db = MotionDatabase(dhb_method="double_reflection")

    all_demos = []
    for hdf5_file in hdf5_files:
        task_name = hdf5_file.stem.replace("_demo", "")
        demos = list_demos_in_dataset(str(hdf5_file))[:3]  # First 3 demos per task

        for demo_id in demos:
            demo = load_demo_from_hdf5(str(hdf5_file), demo_id)
            demo["task_name"] = task_name
            all_demos.append(demo)

            db.add(
                positions=demo["positions"],
                quaternions=demo["quaternions"],
                metadata={"task": task_name, "demo_id": demo_id},
            )

    print(f"Database contains {len(all_demos)} trajectories")

    # Query with one of the demos
    query_demo = all_demos[0]
    print(f"\nQuerying with: {query_demo['task_name']} (demo 0)")

    results = db.retrieve(
        query_positions=query_demo["positions"],
        query_quaternions=query_demo["quaternions"],
        k=5,
        use_dtw=True,
    )

    print("\nTop 5 matches:")
    for i, (inv, metadata, distance) in enumerate(results):
        print(f"  {i+1}. {metadata['task'][:50]:50s} (demo {metadata['demo_id']}) - distance: {distance:.4f}")

    print("\n" + "=" * 60)
    print("Motion retrieval demo completed!")
    print("=" * 60)

    return True


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LIBERO + DHB-XR Full Integration Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # DHB demo only (no simulation required)
    python libero_full_demo.py --dhb-only

    # Full simulation
    ~/miniforge3/bin/mamba run -n libero python libero_full_demo.py

    # With rendering
    ~/miniforge3/bin/mamba run -n libero python libero_full_demo.py --render

    # Motion retrieval demo
    python libero_full_demo.py --retrieval
        """
    )
    parser.add_argument("--task_id", type=int, default=0, help="LIBERO task ID (0-9)")
    parser.add_argument("--demo_id", type=int, default=0, help="Demo ID within task")
    parser.add_argument("--render", action="store_true", help="Render simulation (requires display)")
    parser.add_argument("--save-video", type=str, default=None, help="Save video to file (e.g., demo.mp4)")
    parser.add_argument("--dhb-only", action="store_true", help="Run DHB demo only (no simulation)")
    parser.add_argument("--retrieval", action="store_true", help="Run motion retrieval demo")
    args = parser.parse_args()

    print("=" * 60)
    print("LIBERO + DHB-XR Full Integration Demo")
    print("=" * 60)

    # Check dependencies
    print("\nDependency check:")
    print(f"  h5py:      {'OK' if HAS_H5PY else 'MISSING'}")
    print(f"  LIBERO:    {'OK' if HAS_LIBERO else 'MISSING'}")
    print(f"  robosuite: {'OK' if HAS_ROBOSUITE else 'MISSING'}")

    if args.retrieval:
        run_motion_retrieval_demo()
        return

    if args.dhb_only or not (HAS_LIBERO and HAS_ROBOSUITE):
        if not args.dhb_only and not HAS_LIBERO:
            print("\nLIBERO not installed. Running DHB-only demo...")
            print("\nTo install LIBERO for full simulation:")
            print("  source ~/miniforge3/bin/activate libero")
            print("  # or: conda create -n libero python=3.10 && conda activate libero")
            print("  pip install robosuite==1.4.0 mujoco bddl==1.0.1 robomimic==0.2.0")
            print("  pip install future easydict hydra-core cloudpickle gym==0.25.2")
            print("  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git")
            print("  cd LIBERO && pip install -e . --config-settings editable_mode=compat")

        run_dhb_only_demo()
    else:
        run_simulation_demo(
            task_id=args.task_id,
            demo_id=args.demo_id,
            render=args.render,
            save_video=args.save_video,
        )


if __name__ == "__main__":
    main()
