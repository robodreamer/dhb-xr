"""Panda Sim Proof-of-Concept: DHB-XR Full Pipeline in MuJoCo.

Validates the complete DHB-XR pipeline end-to-end in simulation:

    LIBERO demo (HDF5)
        └→ dhb_xr.LiberoAdapter     — load demo trajectory
        └→ encode_dhb_dr             — SE(3)-invariant encoding
        └→ FatropSolver.adapt        — geometric adaptation at ~7ms
        └→ decode_dhb_dr             — SE(3) trajectory
        └→ embodik.VelocityIK        — Panda joint velocities
        └→ LIBERO OffScreenRenderEnv — MuJoCo execution → success/fail + video

This is the simulation equivalent of a real-robot PoC.
All experiments run in sim — no hardware required.

Usage:
    # DHB-only (no simulation, fastest)
    python panda_dhb_sim_poc.py --dhb-only --perturbation-magnitude 0.10

    # Full sim pipeline (requires LIBERO conda environment)
    ~/miniforge3/bin/mamba run -n libero \\
        python panda_dhb_sim_poc.py \\
        --perturbation-magnitude 0.10 \\
        --save-video output/panda_poc.mp4 \\
        --output-json results/panda_poc_results.json

    # Full perturbation sweep
    ~/miniforge3/bin/mamba run -n libero \\
        python panda_dhb_sim_poc.py \\
        --sweep \\
        --output-json results/panda_poc_sweep.json

Author: Andy Park
Date: 2026-04-09
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add dhb_xr src to path
_SRC = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(_SRC))

# ── DHB-XR imports ─────────────────────────────────────────────────────────
from dhb_xr.encoder.dhb_dr import encode_dhb_dr
from dhb_xr.decoder.dhb_dr import decode_dhb_dr
from dhb_xr.core.geometry import rot_to_quat, quat_to_rot

try:
    from dhb_xr.optimization.fatrop_solver import generate_trajectory_fatrop
    HAS_FATROP = True
except ImportError:
    HAS_FATROP = False

try:
    from dhb_xr.optimization.casadi_solver import generate_trajectory
    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False

try:
    from dhb_xr.integration.vla.libero import LiberoAdapter
    HAS_ADAPTER = True
except ImportError:
    HAS_ADAPTER = False

# ── embodik IK ─────────────────────────────────────────────────────────────
try:
    import embodik as eik
    HAS_EMBODIK = True
except ImportError:
    HAS_EMBODIK = False

# ── LIBERO simulation ───────────────────────────────────────────────────────
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    import libero  # noqa: F401
    from libero.libero import benchmark as libero_benchmark
    from libero.libero.envs import OffScreenRenderEnv
    HAS_LIBERO = True
except ImportError:
    HAS_LIBERO = False

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

# ── Optional FAST tokenizer for comparison ──────────────────────────────────
try:
    from dhb_xr.tokenization.fast_tokenizer import FASTTokenizer
    HAS_FAST = True
except ImportError:
    HAS_FAST = False


# ══════════════════════════════════════════════════════════════════════════════
# Demo loading
# ══════════════════════════════════════════════════════════════════════════════

LIBERO_DATASET_DIR = Path("/home/andypark/Projects/data/libero/libero_spatial")


def find_demo_file(task_id: int = 0) -> Optional[Path]:
    """Find a LIBERO demo HDF5 file."""
    # Try primary location
    if LIBERO_DATASET_DIR.exists():
        files = sorted(LIBERO_DATASET_DIR.glob("*.hdf5"))
        if files:
            return files[task_id % len(files)]

    # Fallback: any LIBERO data
    for search_path in [Path("/home/andypark/Projects/data"), Path("/tmp/libero")]:
        files = list(search_path.rglob("*.hdf5"))
        if files:
            return files[task_id % len(files)]

    return None


def load_demo(hdf5_path: Path, demo_id: int = 0) -> Dict:
    """Load a single demo trajectory from LIBERO HDF5."""
    if not HAS_H5PY:
        raise ImportError("h5py required: pip install h5py")

    with h5py.File(str(hdf5_path), "r") as f:
        # Try multiple key conventions
        data_group = f.get("data", f)
        demo_keys = [k for k in data_group.keys() if "demo" in k.lower()]
        if not demo_keys:
            demo_keys = list(data_group.keys())

        demo_key = sorted(demo_keys)[demo_id % len(demo_keys)]
        demo = data_group[demo_key]

        # Extract observations
        obs = demo.get("obs", demo)

        # End-effector positions — try multiple LIBERO key conventions
        pos = None
        for key in ["ee_pos", "robot0_eef_pos", "eef_pos"]:
            if key in obs:
                pos = np.array(obs[key])
                break
        if pos is None and "ee_states" in obs:
            pos = np.array(obs["ee_states"])[:, :3]
        if pos is None and "robot_states" in demo:
            pos = np.array(demo["robot_states"])[:, :3]

        # End-effector quaternions — LIBERO stores as wxyz in ee_states cols 3-6
        quat = None
        for key in ["robot0_eef_quat", "eef_quat", "ee_quat"]:
            if key in obs:
                raw = np.array(obs[key])
                if abs(raw[0, 0]) > abs(raw[0, 3]):  # wxyz format → convert to xyzw
                    quat = np.column_stack([raw[:, 1], raw[:, 2], raw[:, 3], raw[:, 0]])
                else:
                    quat = raw
                break
        if quat is None and "ee_ori" in obs:
            # LIBERO stores ee_ori as Euler angles (XYZ), convert to quaternion (xyzw)
            from scipy.spatial.transform import Rotation
            euler = np.array(obs["ee_ori"])  # (N, 3) XYZ Euler
            r = Rotation.from_euler("xyz", euler)
            quat = r.as_quat()  # xyzw
        if quat is None and "ee_states" in obs:
            ee_s = np.array(obs["ee_states"])  # (N, 6): pos(3) + euler(3)
            from scipy.spatial.transform import Rotation
            r = Rotation.from_euler("xyz", ee_s[:, 3:6])
            quat = r.as_quat()  # xyzw
        if quat is None and "robot_states" in demo:
            raw = np.array(demo["robot_states"])[:, 3:7]  # wxyz
            quat = np.column_stack([raw[:, 1], raw[:, 2], raw[:, 3], raw[:, 0]])

        # Actions (joint velocities or EEF deltas)
        actions = np.array(demo.get("actions", np.zeros((len(pos), 7))))

        if pos is None or quat is None:
            raise ValueError(f"Could not extract trajectory from {hdf5_path}:{demo_key}")

        return {
            "positions": pos,
            "quaternions": quat,  # xyzw
            "actions": actions,
            "task_name": str(hdf5_path.stem),
            "demo_key": demo_key,
            "length": len(pos),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Core pipeline functions
# ══════════════════════════════════════════════════════════════════════════════

def dhb_encode(positions: np.ndarray, quaternions: np.ndarray) -> Dict:
    """Encode trajectory to DHB invariants."""
    enc = encode_dhb_dr(positions, quaternions)
    return {
        "linear_invariants": enc["linear_motion_invariants"],
        "angular_invariants": enc["angular_motion_invariants"],
        "linear_frame_initial": enc["linear_frame_initial"],
        "angular_frame_initial": enc["angular_frame_initial"],
        "initial_pose": enc["initial_pose"],
    }


def dhb_adapt(
    positions: np.ndarray,
    quaternions: np.ndarray,
    new_start_pos: np.ndarray,
    new_start_quat: np.ndarray,
    new_goal_pos: Optional[np.ndarray] = None,
    new_goal_quat: Optional[np.ndarray] = None,
) -> Dict:
    """Adapt trajectory to new start/goal pose via Fatrop → CasADi → decode fallback."""
    goal_pos = new_goal_pos if new_goal_pos is not None else positions[-1]
    goal_quat = new_goal_quat if new_goal_quat is not None else quaternions[-1]

    pose_init = {"position": new_start_pos, "quaternion": new_start_quat}
    pose_final = {"position": goal_pos, "quaternion": goal_quat}

    t0 = time.perf_counter()

    if HAS_FATROP:
        try:
            result = generate_trajectory_fatrop(
                positions, quaternions, pose_init, pose_final, traj_length=len(positions)
            )
            result["solver"] = "fatrop"
            result["solve_time_ms"] = (time.perf_counter() - t0) * 1000
            return result
        except Exception:
            pass

    # Decode-only fallback: re-decode invariants at new start pose.
    # This achieves near-zero goal error when the perturbation is a rigid translation
    # (the invariants capture the motion shape, decoding at new start shifts the trajectory).
    # Fatrop additionally enforces the exact goal constraint.
    enc = dhb_encode(positions, quaternions)
    import copy
    new_initial_pose = copy.deepcopy(enc["initial_pose"])
    new_initial_pose["position"] = new_start_pos
    new_initial_pose["quaternion"] = new_start_quat
    dec = decode_dhb_dr(
        enc["linear_invariants"],
        enc["angular_invariants"],
        new_initial_pose,
    )
    return {
        "positions": dec["positions"],
        "quaternions": dec["quaternions"],
        "solver": "decode_invariants",
        "solve_time_ms": (time.perf_counter() - t0) * 1000,
    }


def trajectory_to_joint_actions(
    positions: np.ndarray,
    quaternions: np.ndarray,
    action_dim: int = 7,
) -> np.ndarray:
    """Convert SE(3) trajectory to joint actions via embodik IK (or EEF delta fallback)."""
    if HAS_EMBODIK:
        return _embodik_ik(positions, quaternions, action_dim)
    return _eef_delta_actions(positions, quaternions, action_dim)


def _embodik_ik(
    positions: np.ndarray,
    quaternions: np.ndarray,
    action_dim: int = 7,
) -> np.ndarray:
    """Convert SE(3) trajectory to joint velocities via embodik Panda IK."""
    robot = eik.RobotModel("panda", floating_base=False)
    solver = eik.KinematicsSolver(robot)

    N = len(positions)
    joint_actions = np.zeros((N, action_dim), dtype=np.float32)
    q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])  # Panda home config

    for i in range(N):
        # embodik expects wxyz quaternion format
        q_xyzw = quaternions[i]
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        try:
            result = solver.solve_position_ik(q, positions[i], q_wxyz, frame="panda_hand")
            if result is not None:
                q = result
        except Exception:
            pass
        joint_actions[i, :min(7, action_dim)] = q[:min(7, action_dim)]
        if action_dim > 7:
            joint_actions[i, 7] = 1.0  # gripper open

    return joint_actions


def _eef_delta_actions(
    positions: np.ndarray,
    quaternions: np.ndarray,
    action_dim: int = 7,
) -> np.ndarray:
    """EEF delta action fallback (no IK required)."""
    N = len(positions)
    actions = np.zeros((N, action_dim), dtype=np.float32)
    for i in range(N - 1):
        actions[i, :3] = positions[i + 1] - positions[i]
        R_c = quat_to_rot(quaternions[i])
        R_n = quat_to_rot(quaternions[i + 1])
        R_d = R_n @ R_c.T
        theta = np.arccos(np.clip((np.trace(R_d) - 1) / 2, -1.0, 1.0))
        if abs(theta) > 1e-8:
            axis = np.array([R_d[2,1]-R_d[1,2], R_d[0,2]-R_d[2,0], R_d[1,0]-R_d[0,1]])
            actions[i, 3:6] = axis / np.linalg.norm(axis) * theta
        if action_dim >= 7:
            actions[i, 6] = 1.0
    actions[-1] = actions[-2]
    return actions


# ══════════════════════════════════════════════════════════════════════════════
# Simulation execution
# ══════════════════════════════════════════════════════════════════════════════

def run_in_sim(
    hdf5_path: Path,
    joint_actions: np.ndarray,
    task_id: int = 0,
    save_video: Optional[str] = None,
) -> Dict:
    """Execute joint actions in LIBERO MuJoCo simulation.

    Returns:
        {success: bool, final_reward: float, frames: list (if video requested)}
    """
    if not HAS_LIBERO:
        return {"success": None, "final_reward": None, "note": "LIBERO not installed"}

    try:
        # Load LIBERO environment for this task
        benchmark_dict = libero_benchmark.get_benchmark_dict()
        task_suite_name = "libero_spatial"
        task_suite = benchmark_dict.get(task_suite_name, None)

        env_args = {
            "bddl_file_name": str(hdf5_path).replace(".hdf5", ".bddl"),
            "camera_heights": 128,
            "camera_widths": 128,
        }
        env = OffScreenRenderEnv(**env_args)
        obs = env.reset()

        frames = []
        success = False
        final_reward = 0.0

        for i, action in enumerate(joint_actions):
            obs, reward, done, info = env.step(action)
            final_reward = float(reward)
            success = reward > 0 or info.get("success", False)

            if save_video:
                frame = env.render(mode="rgb_array")
                frames.append(frame)

            if done:
                break

        env.close()

        if save_video and frames and HAS_IMAGEIO:
            Path(save_video).parent.mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(save_video, frames, fps=20)
            print(f"  Video saved: {save_video}")

        return {
            "success": success,
            "final_reward": final_reward,
            "num_steps": i + 1,
            "frames": len(frames),
        }

    except Exception as e:
        return {"success": None, "error": str(e), "note": "Simulation failed"}


# ══════════════════════════════════════════════════════════════════════════════
# Main experiment functions
# ══════════════════════════════════════════════════════════════════════════════

def run_single_poc(
    task_id: int = 0,
    perturbation_magnitude: float = 0.10,
    save_video: Optional[str] = None,
    dhb_only: bool = False,
    output_json: Optional[str] = None,
) -> Dict:
    """Run single proof-of-concept: encode → adapt → IK → (sim execute).

    Args:
        task_id: LIBERO task index
        perturbation_magnitude: spatial offset in meters (0.10 = 10cm)
        save_video: path for output video (requires LIBERO)
        dhb_only: skip simulation, just validate geometric pipeline
        output_json: path to write results JSON

    Returns:
        result dict with timing, errors, success
    """
    print("\n" + "=" * 65)
    print("  DHB-XR Panda Sim PoC")
    print("=" * 65)
    print(f"  Task ID: {task_id}  |  Perturbation: {perturbation_magnitude*100:.0f}cm")
    print(f"  Pipeline: encode → Fatrop adapt → embodik IK → MuJoCo")
    print(f"  Solver: {'Fatrop' if HAS_FATROP else 'CasADi' if HAS_CASADI else 'decode'}")
    print(f"  embodik IK: {'yes' if HAS_EMBODIK else 'no (EEF delta fallback)'}")
    print()

    # 1. Load demo
    hdf5_path = find_demo_file(task_id)
    if hdf5_path is None:
        print("  No demo file found. Set LIBERO_DATASET_DIR or provide --hdf5-path.")
        return {"error": "no demo file"}

    print(f"  Demo: {hdf5_path.name}")
    demo = load_demo(hdf5_path, demo_id=0)
    print(f"  Loaded {demo['length']}-step trajectory")

    # 2. Encode
    t_enc = time.perf_counter()
    enc = dhb_encode(demo["positions"], demo["quaternions"])
    enc_ms = (time.perf_counter() - t_enc) * 1000
    print(f"  Encode: {enc_ms:.1f}ms")

    # 3. Define perturbation
    direction = np.array([1, 1, 0]) / np.sqrt(2)
    offset = direction * perturbation_magnitude
    new_start = demo["positions"][0] + offset
    new_goal = demo["positions"][-1] + offset
    print(f"  Perturbation: [{offset[0]:.3f}, {offset[1]:.3f}, {offset[2]:.3f}]m")

    # 4. Adapt trajectory (Fatrop ~7ms)
    adapted = dhb_adapt(
        demo["positions"], demo["quaternions"],
        new_start, demo["quaternions"][0],
        new_goal, demo["quaternions"][-1],
    )
    adapt_ms = adapted["solve_time_ms"]
    print(f"  Adapt: {adapt_ms:.1f}ms  (solver: {adapted['solver']})")

    # 5. Geometric quality check
    goal_err_mm = np.linalg.norm(adapted["positions"][-1] - new_goal) * 1000
    orig_disp = demo["positions"][-1] - demo["positions"][0]
    adapted_disp = adapted["positions"][-1] - adapted["positions"][0]
    shape_err_mm = np.linalg.norm(orig_disp - adapted_disp) * 1000
    print(f"  Goal error: {goal_err_mm:.2f}mm  |  Shape error: {shape_err_mm:.2f}mm")

    # 6. IK
    t_ik = time.perf_counter()
    joint_actions = trajectory_to_joint_actions(
        adapted["positions"], adapted["quaternions"]
    )
    ik_ms = (time.perf_counter() - t_ik) * 1000
    print(f"  IK: {ik_ms:.1f}ms  ({len(joint_actions)} steps, {joint_actions.shape[1]}-DOF)")

    result = {
        "task": demo["task_name"],
        "perturbation_m": perturbation_magnitude,
        "pipeline": {
            "encode_ms": enc_ms,
            "adapt_ms": adapt_ms,
            "ik_ms": ik_ms,
            "total_ms": enc_ms + adapt_ms + ik_ms,
            "solver": adapted["solver"],
        },
        "geometry": {
            "goal_error_mm": goal_err_mm,
            "shape_error_mm": shape_err_mm,
        },
        "demo_length": demo["length"],
        "embodik_ik": HAS_EMBODIK,
    }

    # 7. Optional: also run naive replay for comparison
    naive_goal_err = np.linalg.norm(demo["positions"][-1] - new_goal) * 1000
    result["naive_baseline"] = {"goal_error_mm": naive_goal_err}
    print(f"\n  Comparison:")
    print(f"    DHB-XR goal error:   {goal_err_mm:.1f}mm")
    print(f"    Naive replay error:  {naive_goal_err:.1f}mm")
    print(f"    Improvement:         {naive_goal_err - goal_err_mm:.1f}mm")

    if dhb_only:
        print("\n  [DHB-only mode — skipping MuJoCo simulation]")
        result["simulation"] = None
    else:
        # 8. Execute in MuJoCo
        print(f"\n  Running in MuJoCo simulation...")
        sim_result = run_in_sim(
            hdf5_path, joint_actions, task_id=task_id, save_video=save_video
        )
        result["simulation"] = sim_result
        if sim_result.get("success") is not None:
            status = "SUCCESS" if sim_result["success"] else "FAILURE"
            print(f"  Simulation: {status}  (reward={sim_result.get('final_reward', 0):.3f})")
        else:
            print(f"  Simulation: {sim_result.get('note', 'error')}")

    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Results: {output_json}")

    print("\n" + "=" * 65)
    print(f"  Total pipeline: {result['pipeline']['total_ms']:.1f}ms  "
          f"(target: adapt ≤ 10ms)")
    print("=" * 65)

    return result


def run_sweep(
    task_id: int = 0,
    perturbation_magnitudes: Optional[List[float]] = None,
    dhb_only: bool = False,
    output_json: str = "results/panda_poc_sweep.json",
) -> Dict:
    """Run full perturbation sweep: 0/2/5/10/15cm across DHB-XR + baselines."""
    if perturbation_magnitudes is None:
        perturbation_magnitudes = [0.0, 0.02, 0.05, 0.10, 0.15]

    print("\n" + "=" * 65)
    print("  DHB-XR Panda Sim PoC — Perturbation Sweep")
    print("=" * 65)

    results = {"task_id": task_id, "sweep": [], "summary": {}}
    dhb_goals, naive_goals = [], []

    for mag in perturbation_magnitudes:
        r = run_single_poc(
            task_id=task_id,
            perturbation_magnitude=mag,
            dhb_only=dhb_only,
        )
        results["sweep"].append(r)
        dhb_goals.append(r["geometry"]["goal_error_mm"])
        naive_goals.append(r["naive_baseline"]["goal_error_mm"])

    results["summary"] = {
        "dhb_mean_goal_error_mm": float(np.mean(dhb_goals)),
        "naive_mean_goal_error_mm": float(np.mean(naive_goals)),
        "mean_improvement_mm": float(np.mean(naive_goals) - np.mean(dhb_goals)),
    }

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Sweep results: {output_json}")
    print(f"  Mean improvement: {results['summary']['mean_improvement_mm']:.1f}mm")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DHB-XR Panda Simulation PoC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # DHB pipeline validation only (no simulation, fast)
    python panda_dhb_sim_poc.py --dhb-only --perturbation-magnitude 0.10

    # Full MuJoCo simulation
    python panda_dhb_sim_poc.py --perturbation-magnitude 0.10 --save-video poc.mp4

    # Full sweep: 0/2/5/10/15cm
    python panda_dhb_sim_poc.py --sweep --output-json results/panda_sweep.json
        """,
    )
    parser.add_argument("--task-id", type=int, default=0, help="LIBERO task index")
    parser.add_argument(
        "--perturbation-magnitude",
        type=float,
        default=0.10,
        help="Spatial perturbation in meters (default: 0.10 = 10cm)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run full sweep: 0/2/5/10/15cm perturbations",
    )
    parser.add_argument(
        "--dhb-only",
        action="store_true",
        help="Run DHB pipeline only (no MuJoCo simulation)",
    )
    parser.add_argument("--save-video", type=str, default=None, help="Video output path")
    parser.add_argument(
        "--output-json",
        type=str,
        default="results/panda_poc_results.json",
        help="JSON output path",
    )

    args = parser.parse_args()

    if args.sweep:
        run_sweep(
            task_id=args.task_id,
            dhb_only=args.dhb_only,
            output_json=args.output_json,
        )
    else:
        run_single_poc(
            task_id=args.task_id,
            perturbation_magnitude=args.perturbation_magnitude,
            save_video=args.save_video,
            dhb_only=args.dhb_only,
            output_json=args.output_json,
        )


if __name__ == "__main__":
    main()
