#!/usr/bin/env python
"""Test blended controller for LIBERO OSC_POSE action space.

The KEY INSIGHT: LIBERO's OSC_POSE controller applies variable per-step
scaling (20x-577x). A pure PD controller computing action[:3] = Kp * error
doesn't match the OSC dynamics. Instead, we BLEND the original demo actions
with a position correction:

    action[:3] = original_action[:3] + alpha * Kp * (adapted_target - current_ee)

At 0cm offset, correction ~= 0, so action ~= original -> should get ~100%.
At 10cm offset, correction compensates for the position shift.

IMPORTANT: Runtime obs keys differ from HDF5 obs keys:
  - Runtime: robot0_eef_pos, robot0_eef_quat (from env.step / env.reset)
  - HDF5:    ee_pos, ee_ori (stored in demo files)

Also: must use HDF5 states[0] as init_state, NOT benchmark init_states.
Benchmark init_states do not match the demo trajectory.

This script tests:
1. Pure naive replay (baseline, should be ~100% at 0cm)
2. Pure PD controller Kp=12 (broken, ~0% at 0cm)
3. Blended controller (original + correction, target ~100% at 0cm)
4. Blended controller at 5cm and 10cm offsets

Usage:
    MUJOCO_GL=egl ~/miniforge3/envs/libero/bin/python \
        examples/integration/test_blended_controller.py

Author: Andy Park
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# ---- paths for LIBERO and dhb_xr ----
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, "/home/andypark/Projects/repos/LIBERO")
sys.path.insert(0, "/home/andypark/Projects/repos/LIBERO-PRO")

import numpy as np

# ---- Fix torch.load for PyTorch 2.6+ (LIBERO uses numpy arrays in init states) ----
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# ---- LIBERO imports ----
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# Runtime obs key for EE position (NOT the HDF5 key "ee_pos")
OBS_EEF_POS_KEY = "robot0_eef_pos"

# ===========================================================================
# Helpers
# ===========================================================================


def load_demo(dataset_path: str, demo_id: int = 0) -> dict:
    """Load demo actions, EE trajectory, and init state from LIBERO HDF5."""
    import h5py

    with h5py.File(dataset_path, "r") as f:
        demo_key = f"demo_{demo_id}"
        if demo_key not in f["data"]:
            raise ValueError(f"Demo {demo_id} not found in {dataset_path}")

        demo = f["data"][demo_key]
        actions = np.array(demo["actions"])
        # HDF5 obs keys (different from runtime obs keys!)
        ee_pos = np.array(demo["obs"]["ee_pos"])      # (N, 3)
        ee_ori = np.array(demo["obs"]["ee_ori"])        # (N, 3) axis-angle
        init_state = np.array(demo["states"][0])        # (92,) sim state

    return {
        "actions": actions,
        "ee_pos": ee_pos,
        "ee_ori": ee_ori,
        "init_state": init_state,
        "num_frames": len(actions),
    }


def create_env(task_id: int = 0, camera_size: int = 128):
    """Create LIBERO environment and return (env, task, dataset_path)."""
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

    dataset_path = os.path.join(
        get_libero_path("datasets"),
        "libero_spatial",
        f"{task.name}_demo.hdf5",
    )

    return env, task, dataset_path


def reset_env_to_demo(env, init_state: np.ndarray) -> dict:
    """Reset environment using the HDF5 demo init state, return obs."""
    env.reset()
    obs = env.set_init_state(init_state)
    return obs


def apply_offset(positions: np.ndarray, offset_m: float) -> np.ndarray:
    """Apply a uniform x-axis offset (meters) to positions."""
    offset = np.zeros(3)
    offset[0] = offset_m
    return positions + offset


# ===========================================================================
# Controllers
# ===========================================================================


def run_naive_replay(env, demo: dict, verbose: bool = True) -> dict:
    """Baseline: replay original demo actions directly."""
    obs = reset_env_to_demo(env, demo["init_state"])

    final_reward = 0.0
    for step in range(demo["num_frames"]):
        action = demo["actions"][step]
        obs, reward, done, info = env.step(action)
        final_reward = reward
        if done:
            break

    success = final_reward > 0
    if verbose:
        print(f"    Naive replay: reward={final_reward:.2f}, "
              f"steps={step + 1}, success={success}")
    return {"reward": final_reward, "success": success, "steps": step + 1}


def run_pure_pd(env, demo: dict, Kp: float = 12.0,
                offset_m: float = 0.0, verbose: bool = True) -> dict:
    """Pure PD controller: action[:3] = Kp * (target - current). Broken for OSC."""
    obs = reset_env_to_demo(env, demo["init_state"])

    target_positions = demo["ee_pos"].copy()
    if offset_m != 0.0:
        target_positions = apply_offset(target_positions, offset_m)

    final_reward = 0.0
    for step in range(demo["num_frames"]):
        current_ee = np.array(obs[OBS_EEF_POS_KEY])
        target_pos = target_positions[step]

        # Pure PD: replace position action entirely
        pos_error = target_pos - current_ee
        action = demo["actions"][step].copy()
        action[:3] = np.clip(Kp * pos_error, -1.0, 1.0)
        # Keep original orientation and gripper from demo

        obs, reward, done, info = env.step(action)
        final_reward = reward
        if done:
            break

    success = final_reward > 0
    if verbose:
        print(f"    Pure PD (Kp={Kp}): reward={final_reward:.2f}, "
              f"steps={step + 1}, success={success}")
    return {"reward": final_reward, "success": success, "steps": step + 1}


def run_blended_controller(
    env,
    demo: dict,
    offset_m: float = 0.0,
    Kp_correction: float = 2.0,
    alpha: float = 0.5,
    verbose: bool = True,
) -> dict:
    """Blended controller: original action + alpha * Kp * correction.

    At 0cm offset, correction ~= 0, so action ~= original.
    At larger offsets, the correction steers toward the adapted target.
    """
    obs = reset_env_to_demo(env, demo["init_state"])

    # Adapted target trajectory
    adapted_positions = demo["ee_pos"].copy()
    if offset_m != 0.0:
        adapted_positions = apply_offset(adapted_positions, offset_m)

    final_reward = 0.0
    tracking_errors = []

    for step in range(demo["num_frames"]):
        current_ee = np.array(obs[OBS_EEF_POS_KEY])
        adapted_target = adapted_positions[step]

        # Position correction: how far are we from where we should be?
        correction = adapted_target - current_ee
        tracking_errors.append(np.linalg.norm(correction))

        # Blend: original action + correction
        action = demo["actions"][step].copy()
        action[:3] = action[:3] + alpha * Kp_correction * correction
        action[:3] = np.clip(action[:3], -1.0, 1.0)
        # Keep original orientation and gripper from demo

        obs, reward, done, info = env.step(action)
        final_reward = reward
        if done:
            break

    success = final_reward > 0
    if verbose:
        mean_err = np.mean(tracking_errors)
        max_err = np.max(tracking_errors)
        print(f"    Blended (Kp={Kp_correction}, alpha={alpha}, offset={offset_m*100:.0f}cm): "
              f"reward={final_reward:.2f}, steps={step + 1}, success={success}, "
              f"mean_track_err={mean_err:.4f}m, max_track_err={max_err:.4f}m")
    return {
        "reward": final_reward,
        "success": success,
        "steps": step + 1,
        "mean_tracking_error": np.mean(tracking_errors),
        "max_tracking_error": np.max(tracking_errors),
    }


# ===========================================================================
# Main
# ===========================================================================


def main():
    print("=" * 70)
    print("  Blended Controller Test -- LIBERO OSC_POSE")
    print("=" * 70)
    print()

    # --- Setup ---
    task_id = 0
    print(f"Creating LIBERO environment (task_id={task_id})...")
    env, task, dataset_path = create_env(task_id=task_id)
    print(f"  Task: {task.name}")
    print(f"  Language: {task.language}")
    print(f"  Dataset: {dataset_path}")
    print()

    print("Loading demo 0...")
    demo = load_demo(dataset_path, demo_id=0)
    print(f"  Frames: {demo['num_frames']}")
    print(f"  Actions shape: {demo['actions'].shape}")
    print(f"  EE pos range: {demo['ee_pos'].min(axis=0)} -> {demo['ee_pos'].max(axis=0)}")
    print(f"  Init state shape: {demo['init_state'].shape}")
    print()

    results = {}

    # ==================================================================
    # TEST 1: Pure naive replay at 0cm (baseline)
    # ==================================================================
    print("-" * 70)
    print("TEST 1: Pure naive replay at 0cm (baseline, should be ~100%)")
    print("-" * 70)
    results["naive_replay_0cm"] = run_naive_replay(env, demo)
    print()

    # ==================================================================
    # TEST 2a: Pure PD controller Kp=12 at 0cm
    # ==================================================================
    print("-" * 70)
    print("TEST 2a: Pure PD controller Kp=12 at 0cm")
    print("-" * 70)
    results["pure_pd_0cm"] = run_pure_pd(env, demo, Kp=12.0, offset_m=0.0)
    print()

    # ==================================================================
    # TEST 2b: Pure PD controller at various offsets
    # ==================================================================
    print("-" * 70)
    print("TEST 2b: Pure PD controller Kp=12 at various offsets")
    print("-" * 70)
    for offset_cm in [3, 5, 7, 10]:
        key = f"pure_pd_{offset_cm}cm"
        results[key] = run_pure_pd(
            env, demo, Kp=12.0, offset_m=offset_cm / 100.0,
        )
    print()

    # ==================================================================
    # TEST 3: Blended controller at 0cm -- param sweep
    # ==================================================================
    print("-" * 70)
    print("TEST 3: Blended controller at 0cm -- param sweep")
    print("-" * 70)
    for alpha in [0.3, 0.5, 1.0]:
        for Kp in [1.0, 2.0, 4.0]:
            key = f"blended_0cm_a{alpha}_Kp{Kp}"
            results[key] = run_blended_controller(
                env, demo,
                offset_m=0.0,
                Kp_correction=Kp,
                alpha=alpha,
            )
    print()

    # ==================================================================
    # TEST 4: Blended controller at various offsets -- param sweep
    # ==================================================================
    print("-" * 70)
    print("TEST 4: Blended controller at offsets -- sweep")
    print("-" * 70)
    for offset_cm in [3, 5, 7, 10]:
        for alpha, Kp in [(0.3, 1.0), (0.3, 2.0), (0.5, 1.0), (1.0, 2.0)]:
            key = f"blended_{offset_cm}cm_a{alpha}_Kp{Kp}"
            results[key] = run_blended_controller(
                env, demo,
                offset_m=offset_cm / 100.0,
                Kp_correction=Kp,
                alpha=alpha,
            )
    print()

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print()

    # Group by offset
    for offset_cm in [0, 3, 5, 7, 10]:
        print(f"  --- Offset: {offset_cm} cm ---")
        print(f"  {'Method':<50} {'Success':>8} {'Reward':>8}")
        print(f"  {'-'*50} {'-'*8} {'-'*8}")

        if offset_cm == 0:
            # Naive replay
            r = results["naive_replay_0cm"]
            print(f"  {'Naive replay':<50} {str(r['success']):>8} {r['reward']:>8.2f}")
            # Pure PD
            r = results["pure_pd_0cm"]
            print(f"  {'Pure PD Kp=12':<50} {str(r['success']):>8} {r['reward']:>8.2f}")
        else:
            key = f"pure_pd_{offset_cm}cm"
            if key in results:
                r = results[key]
                print(f"  {'Pure PD Kp=12':<50} {str(r['success']):>8} {r['reward']:>8.2f}")

        # Blended results for this offset
        prefix = f"blended_{offset_cm}cm_"
        for k in sorted(results.keys()):
            if k.startswith(prefix):
                label = "Blended " + k[len(prefix):]
                r = results[k]
                print(f"  {label:<50} {str(r['success']):>8} {r['reward']:>8.2f}")
        print()

    print("=" * 70)

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
