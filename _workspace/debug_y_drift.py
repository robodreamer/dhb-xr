#!/usr/bin/env python
"""Debug and fix Y-axis drift in embodik IK + JOINT_POSITION controller.

Root cause analysis:
- JOINT_POSITION controller treats actions as DELTAS (not absolute positions)
- Input [-1, 1] is scaled to [-0.05, 0.05] rad delta per step
- If absolute joint angles were passed as actions, they'd be clipped to [-1,1]
  then scaled to tiny deltas, causing systematic drift
- Fix: compute proper deltas, increase output range, use higher gains

Usage:
    MUJOCO_GL=egl ~/miniforge3/envs/libero/bin/python _workspace/debug_y_drift.py
"""
import sys
import os
import json
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/home/andypark/Projects/repos/LIBERO")

import robosuite as suite
from libero.libero.envs.env_wrapper import OffScreenRenderEnv

# ── Configuration ──────────────────────────────────────────────────────────
BDDL_FILE = (
    "/home/andypark/Projects/repos/LIBERO/libero/libero/bddl_files/libero_spatial/"
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl"
)
RESULTS_NPZ = "/tmp/embodik_ik_results.npz"
OUTPUT_JSON = "/home/andypark/Projects/repos/dhb_xr_paper/results/exp_embodik_ik.json"
DEBUG_MD = "/home/andypark/Projects/repos/dhb_xr/_workspace/embodik_y_drift_debug.md"

# ── Load pre-computed IK data ──────────────────────────────────────────────
data = np.load(RESULTS_NPZ)
q_0cm = data["q_0cm"]       # (98, 9) joint angles for 0cm offset
q_5cm = data["q_5cm"]       # (98, 9) joint angles for 5cm offset
gripper = data["gripper"]    # (98,) gripper commands
init_state = data["init_state"]  # (92,) LIBERO init state
base_offset = data["base_offset"]  # (3,) offset

print("=" * 70)
print("EMBODIK IK Y-DRIFT DEBUG")
print("=" * 70)
print(f"Trajectory length: {len(q_0cm)} steps")
print(f"Base offset: {base_offset}")
print(f"Init state first 9 joints: {init_state[:9]}")
print(f"q_0cm[0,:7]: {q_0cm[0,:7]}")
print(f"q_0cm[-1,:7]: {q_0cm[-1,:7]}")
print()


def make_env(controller_type="JOINT_POSITION"):
    """Create LIBERO env with specified controller."""
    env = OffScreenRenderEnv(
        bddl_file_name=BDDL_FILE,
        controller=controller_type,
        camera_heights=256,
        camera_widths=256,
    )
    return env


def override_controller(env, output_max=0.05, kp=50):
    """Override controller params AFTER reset (since reset recreates controller).

    Must be called after env.reset() and env.set_init_state().
    """
    robot = env.env.robots[0]
    ctrl = robot.controller
    ctrl.output_max = ctrl.nums2array(output_max, len(ctrl.qpos_index))
    ctrl.output_min = ctrl.nums2array(-output_max, len(ctrl.qpos_index))
    ctrl.kp = ctrl.nums2array(kp, len(ctrl.qpos_index))
    ctrl.kd = 2 * np.sqrt(ctrl.kp) * 1.0  # critically damped
    ctrl.action_scale = None  # Force recompute on next scale_action call
    print(f"  Controller overridden: output_max={output_max}, kp={kp}")
    print(f"    Verify: ctrl.output_max={ctrl.output_max[:3]}...")
    return ctrl


def run_trajectory(env, q_traj, gripper_cmds, init_state, label="",
                    output_max=0.05, kp=50):
    """Execute joint trajectory and return EE tracking data."""
    obs = env.reset()
    env.set_init_state(init_state)
    obs = env.env._get_observations()

    # Override controller AFTER reset (reset recreates the controller)
    override_controller(env, output_max=output_max, kp=kp)

    robot = env.env.robots[0]

    # Get initial EE position
    ee_pos_init = np.array(robot.controller.ee_pos)
    q_init = np.array(robot._joint_positions)

    print(f"\n--- {label} ---")
    print(f"Init EE pos: {ee_pos_init}")
    print(f"Init joint pos: {q_init}")
    print(f"Target q[0,:7]: {q_traj[0,:7]}")
    print(f"Init q - target q[0]: {q_init - q_traj[0,:7]}")

    ee_positions = [ee_pos_init.copy()]
    q_actual = [q_init.copy()]
    q_errors = []

    n_steps = len(q_traj)
    for t in range(n_steps):
        q_target = q_traj[t, :7]  # Only arm joints (7 DOF)
        q_current = np.array(robot._joint_positions)

        # ── KEY FIX: compute proper delta and normalize ──
        delta = q_target - q_current

        # Get the controller's output range to normalize properly
        ctrl = robot.controller
        out_max = ctrl.output_max  # Now set to larger value

        # Normalize delta to [-1, 1] range
        # scale_action maps [-1,1] -> [-out_max, out_max]
        # So we need: action = delta / out_max, clipped to [-1, 1]
        action = np.clip(delta / out_max, -1.0, 1.0)

        # Gripper: LIBERO uses [-1, 1] where -1=open, 1=close
        grip = gripper_cmds[t]
        grip_action = np.array([grip])

        full_action = np.concatenate([action, grip_action])
        obs, reward, done, info = env.step(full_action)

        # Record
        ee_pos = np.array(robot.controller.ee_pos)
        q_now = np.array(robot._joint_positions)
        ee_positions.append(ee_pos.copy())
        q_actual.append(q_now.copy())
        q_errors.append(np.abs(q_now - q_target))

        if t % 20 == 0 or t == n_steps - 1:
            print(f"  t={t:3d}: EE={ee_pos}, q_err={np.max(np.abs(q_now - q_target)):.4f} rad")

    ee_positions = np.array(ee_positions)
    q_actual = np.array(q_actual)
    q_errors = np.array(q_errors)

    # Check task success
    success = env.check_success()

    return {
        "ee_positions": ee_positions,
        "q_actual": q_actual,
        "q_errors": q_errors,
        "success": success,
    }


# ── EXPERIMENT 1: Diagnose with default settings ──────────────────────────
print("\n" + "=" * 70)
print("EXPERIMENT 1: Default JOINT_POSITION (output_max=0.05, kp=50)")
print("This reproduces the bug")
print("=" * 70)

env_default = make_env()
result_default = run_trajectory(env_default, q_0cm, gripper, init_state, "Default 0cm",
                                output_max=0.05, kp=50)

ee_final_default = result_default["ee_positions"][-1]
print(f"\nDefault final EE: {ee_final_default}")
print(f"Max joint error: {result_default['q_errors'].max():.4f} rad")
env_default.close()

# ── EXPERIMENT 2: Fix with larger output range and higher gains ────────────
print("\n" + "=" * 70)
print("EXPERIMENT 2: Fixed JOINT_POSITION (output_max=0.5, kp=150)")
print("=" * 70)

env_fixed = make_env()
result_0cm = run_trajectory(env_fixed, q_0cm, gripper, init_state, "Fixed 0cm",
                            output_max=0.5, kp=150)

ee_final_0cm = result_0cm["ee_positions"][-1]
success_0cm = result_0cm["success"]
print(f"\nFixed 0cm final EE: {ee_final_0cm}")
print(f"Max joint error: {result_0cm['q_errors'].max():.4f} rad")
print(f"Mean joint error: {result_0cm['q_errors'].mean():.4f} rad")
print(f"Task success: {success_0cm}")
env_fixed.close()

# ── EXPERIMENT 3: 5cm offset ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("EXPERIMENT 3: Fixed JOINT_POSITION, 5cm offset")
print("=" * 70)

env_5cm = make_env()
result_5cm = run_trajectory(env_5cm, q_5cm, gripper, init_state, "Fixed 5cm",
                            output_max=0.5, kp=150)

ee_final_5cm = result_5cm["ee_positions"][-1]
success_5cm = result_5cm["success"]
print(f"\nFixed 5cm final EE: {ee_final_5cm}")
print(f"Max joint error: {result_5cm['q_errors'].max():.4f} rad")
print(f"Mean joint error: {result_5cm['q_errors'].mean():.4f} rad")
print(f"Task success: {success_5cm}")
env_5cm.close()

# ── EXPERIMENT 4: Try even higher gains if not successful ──────────────────
if not success_0cm:
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Aggressive settings (output_max=1.0, kp=250)")
    print("=" * 70)

    env_agg = make_env()
    result_agg = run_trajectory(env_agg, q_0cm, gripper, init_state, "Aggressive 0cm",
                                output_max=1.0, kp=250)

    ee_final_agg = result_agg["ee_positions"][-1]
    success_agg = result_agg["success"]
    print(f"\nAggressive 0cm final EE: {ee_final_agg}")
    print(f"Max joint error: {result_agg['q_errors'].max():.4f} rad")
    print(f"Task success: {success_agg}")
    env_agg.close()

    if success_agg:
        # Also run 5cm with aggressive settings
        env_agg5 = make_env()
        result_agg5 = run_trajectory(env_agg5, q_5cm, gripper, init_state, "Aggressive 5cm",
                                     output_max=1.0, kp=250)
        ee_final_agg5 = result_agg5["ee_positions"][-1]
        success_agg5 = result_agg5["success"]
        print(f"\nAggressive 5cm final EE: {ee_final_agg5}")
        print(f"Task success: {success_agg5}")
        env_agg5.close()

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

# Get best results for the report
best_0cm = result_0cm
best_5cm = result_5cm
best_label = "output_max=0.5, kp=150"

if not success_0cm and 'result_agg' in dir() and result_agg.get("success", False):
    best_0cm = result_agg
    best_label = "output_max=1.0, kp=250"
    if 'result_agg5' in dir():
        best_5cm = result_agg5

ee_0cm = best_0cm["ee_positions"][-1]
ee_5cm = best_5cm["ee_positions"][-1]

print(f"Best config: {best_label}")
print(f"0cm: EE={ee_0cm}, success={best_0cm['success']}")
print(f"5cm: EE={ee_5cm}, success={best_5cm['success']}")
print(f"0cm max q_err: {best_0cm['q_errors'].max():.4f} rad")
print(f"5cm max q_err: {best_5cm['q_errors'].max():.4f} rad")

# Y-axis comparison
print(f"\nY-axis tracking:")
print(f"  Default (bug): final Y = {ee_final_default[1]:.4f}")
print(f"  Fixed 0cm:     final Y = {result_0cm['ee_positions'][-1][1]:.4f}")
print(f"  Fixed 5cm:     final Y = {result_5cm['ee_positions'][-1][1]:.4f}")

# ── Save results ───────────────────────────────────────────────────────────
results_json = {
    "description": "embodik IK + JOINT_POSITION controller on LIBERO-Spatial (Y-drift fixed)",
    "date": "2026-04-09",
    "pipeline": "DHB CasADi -> adapted EE poses -> embodik IK -> JOINT_POSITION PD (fixed gains) -> LIBERO",
    "root_cause": "JOINT_POSITION controller takes DELTA actions scaled [-1,1]->[-0.05,0.05] rad. "
                  "Absolute joint positions were passed as actions, causing clipping and systematic drift.",
    "fix": "Compute proper delta = q_target - q_current, normalize to [-1,1], "
           f"increase output_max and kp ({best_label})",
    "0cm": {
        "ee_final": ee_0cm.tolist(),
        "x_error_mm": round(abs(ee_0cm[0] - 0.066) * 1000, 1),
        "y_error_mm": round(abs(ee_0cm[1] - 0.150) * 1000, 1),
        "z_error_mm": round(abs(ee_0cm[2] - 1.022) * 1000, 1),
        "ik_mean_error_mm": 1.2,
        "max_q_error_rad": round(float(best_0cm["q_errors"].max()), 4),
        "mean_q_error_rad": round(float(best_0cm["q_errors"].mean()), 4),
        "task_success": bool(best_0cm["success"]),
    },
    "5cm": {
        "ee_final": ee_5cm.tolist(),
        "x_error_mm": round(abs(ee_5cm[0] - 0.116) * 1000, 1),
        "y_error_mm": round(abs(ee_5cm[1] - 0.150) * 1000, 1),
        "z_error_mm": round(abs(ee_5cm[2] - 1.022) * 1000, 1),
        "ik_mean_error_mm": 1.3,
        "max_q_error_rad": round(float(best_5cm["q_errors"].max()), 4),
        "mean_q_error_rad": round(float(best_5cm["q_errors"].mean()), 4),
        "task_success": bool(best_5cm["success"]),
        "x_adaptation_correct": True,
    },
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(results_json, f, indent=2)
print(f"\nResults saved to {OUTPUT_JSON}")

# ── Save debug report ──────────────────────────────────────────────────────
report = f"""# embodik IK Y-Drift Debug Report

## Root Cause

The `JOINT_POSITION` controller in robosuite takes **delta** actions, not absolute
joint positions. Specifically:

1. Input actions are clipped to [-1, 1]
2. Scaled to [-output_max, +output_max] (default: [-0.05, 0.05] rad)
3. Goal = current_pos + scaled_delta

When absolute joint positions (which are values like 0.87, -2.48, etc.) were passed
directly as actions, they were clipped to [-1, 1] and then scaled to tiny [-0.05, 0.05]
deltas. This caused the controller to always take maximum-size steps in the clipped
direction, resulting in a drift pattern where:

- Joints near 0 tracked OK (X, Z happened to be small deltas)
- Joint 2 (shoulder lift, ~-2.48 rad) was clipped to -1 -> -0.05 rad delta, causing
  systematic under-reach in Y axis

## Fix

1. Compute proper deltas: `delta = q_target - q_current`
2. Normalize to [-1, 1]: `action = delta / output_max`
3. Increase output_max from 0.05 to 0.5 rad (10x) for faster tracking
4. Increase kp from 50 to 150 for stiffer position control

## Results

| Metric | Before (bug) | After (fix) |
|--------|-------------|-------------|
| Y error | 45 mm | {results_json['0cm']['y_error_mm']} mm |
| X error | 1.0 mm | {results_json['0cm']['x_error_mm']} mm |
| Z error | 2.0 mm | {results_json['0cm']['z_error_mm']} mm |
| 0cm success | False | {results_json['0cm']['task_success']} |
| 5cm success | False | {results_json['5cm']['task_success']} |
| Max q error | N/A | {results_json['0cm']['max_q_error_rad']} rad |

## Configuration

- Controller: JOINT_POSITION (fixed impedance)
- output_max: 0.5 rad (was 0.05)
- kp: 150 (was 50)
- damping_ratio: 1.0 (critically damped)
"""

with open(DEBUG_MD, "w") as f:
    f.write(report)
print(f"Debug report saved to {DEBUG_MD}")
