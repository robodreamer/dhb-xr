#!/usr/bin/env python3
"""
Position-only debug evaluation for DHB-XR P2 closed-loop pipeline.

PURPOSE: Isolate whether position prediction is correct by using:
  - Predicted positions (from DHBActionHead)
  - Original demo quaternions (from HDF5, NOT adapted)

This bypasses rotation prediction entirely. If positions are correct
but combined eval fails, the bug is in rotation prediction/IK interaction.

Three-stage pipeline (separate Python environments):
  Stage: predict  — already done, reuses /tmp/dhb_spatial_p2_poses.npz
  Stage: ik       — embodik IK with predicted pos + original demo quaternions
  Stage: sim      — LIBERO P2 sim execution with JOINT_POSITION controller

Usage:
  # Stage: IK (use predicted positions + original demo quaternions)
  cd ~/Projects/repos/embodik-singularity-recovery && \\
      pixi run python ~/Projects/repos/dhb_xr/_workspace/eval_dhb_p2_pos_only.py --stage ik

  # Stage: sim
  MUJOCO_GL=egl ~/miniforge3/envs/libero/bin/python \\
      ~/Projects/repos/dhb_xr/_workspace/eval_dhb_p2_pos_only.py --stage sim
"""
import sys
import os
import json
import argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, "/home/andypark/Projects/repos/LIBERO")
sys.path.insert(0, "/home/andypark/Projects/repos/LIBERO-PRO")

# ── File paths ──────────────────────────────────────────────────────────
DEMO_HDF5 = (
    "/home/andypark/Projects/data/libero/libero_spatial/"
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
)
BDDL_ORIG = (
    "/home/andypark/Projects/repos/LIBERO/libero/libero/bddl_files/libero_spatial/"
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl"
)
BDDL_PERTURBED = "/tmp/p2_perturbed.bddl"

# Inter-stage data files
POSES_NPZ = "/tmp/dhb_spatial_p2_poses.npz"
JOINTS_POS_ONLY_NPZ = "/tmp/dhb_spatial_p2_joints_pos_only.npz"
RESULTS_JSON = "/home/andypark/Projects/repos/dhb_xr/_workspace/eval_dhb_p2_pos_only_results.json"

TASK_NAME = "pick up the black bowl between the plate and the ramekin and place it on the plate\n"
OOD_CFG = "/home/andypark/Projects/repos/LIBERO-PRO/libero_ood/ood_spatial_relation.yaml"

# P2 plate position (seed=42 cookies_1 swap)
P2_PLATE_Y = 0.042


# =========================================================================
# P2 BDDL generation (same as main eval script)
# =========================================================================

def generate_p2_bddl():
    """Generate the P2-perturbed BDDL file using LIBERO-PRO SwapPerturbator."""
    import random
    random.seed(42)
    from perturbation import BDDLParser, SwapPerturbator

    with open(BDDL_ORIG) as f:
        orig_bddl = f.read()

    parser = BDDLParser(orig_bddl)
    perturbed_bddl = SwapPerturbator(parser, OOD_CFG).perturb(
        "libero_spatial",
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    )

    with open(BDDL_PERTURBED, "w") as f:
        f.write(perturbed_bddl)

    print(f"P2 BDDL written to {BDDL_PERTURBED} (cookies_1 swap, seed=42)")
    return BDDL_PERTURBED


# =========================================================================
# Stage: IK with predicted positions + original demo quaternions
# =========================================================================

def stage_ik(args):
    """Solve IK using predicted positions + original demo quaternions (bypass rotation prediction)."""
    import embodik
    from robot_descriptions import panda_description
    import h5py
    from scipy.spatial.transform import Rotation as R_scipy

    print("=" * 70)
    print("POSITION-ONLY DEBUG: IK with predicted pos + original demo quats")
    print("  Purpose: isolate position prediction from rotation prediction")
    print("=" * 70)

    # Load predicted poses
    data = np.load(POSES_NPZ, allow_pickle=True)
    n_demos = int(data["n_demos"][0])
    print(f"Source: {str(data['source'])}")
    print(f"N demos: {n_demos}")

    # Load original demo quaternions from HDF5
    f = h5py.File(DEMO_HDF5, "r")
    demo_keys = sorted(f["data"].keys())

    all_demo_joints = {}
    all_demo_ee_pos = {}
    all_demo_rs = {}
    for k in demo_keys:
        all_demo_joints[k] = np.array(f[f"data/{k}/obs/joint_states"][:])
        all_demo_ee_pos[k] = np.array(f[f"data/{k}/obs/ee_pos"][:])
        rs = np.array(f[f"data/{k}/robot_states"])
        all_demo_rs[k] = rs
    f.close()

    # Build robot and solver
    robot = embodik.RobotModel(urdf_path=panda_description.URDF_PATH)
    solver = embodik.KinematicsSolver(robot)
    ik_opts = embodik.PositionIKOptions()
    ik_opts.position_tolerance = 0.0005   # 0.5mm
    ik_opts.orientation_tolerance = 0.1   # ~6 deg
    ik_opts.orientation_gain = 1.0
    ik_opts.max_iterations = 200

    all_joints = {}
    all_errors = {}

    print(f"\nP2 plate Y = {P2_PLATE_Y:.3f}")
    print()

    for seed in range(n_demos):
        key = f"seed{seed}"
        pos_key = f"{key}_pos"
        if pos_key not in data:
            print(f"  SKIP: {pos_key} not found")
            continue

        predicted_pos = data[pos_key]  # (T, 3) — predicted positions
        demo_key = demo_keys[seed]
        demo_joint = all_demo_joints[demo_key]
        demo_ee = all_demo_ee_pos[demo_key]

        # Load original demo quaternions from HDF5 robot_states
        rs = all_demo_rs[demo_key]
        # robot_states[:, 5:9] stores quaternion as xyzw (scipy convention)
        # Convert to wxyz for consistency with DHB pipeline
        orig_quat_xyzw = rs[:, 5:9].astype(np.float64)
        orig_quat_wxyz = orig_quat_xyzw[:, [3, 0, 1, 2]]  # xyzw -> wxyz

        # Build hybrid trajectory: predicted positions + original quaternions
        T_pred = len(predicted_pos)
        T_quat = len(orig_quat_wxyz)
        T_use = min(T_pred, T_quat)

        hybrid_pos = predicted_pos[:T_use]
        hybrid_quat = orig_quat_wxyz[:T_use]

        # IK initialization
        q0 = np.zeros(robot.nq)
        q0[:7] = demo_joint[0]
        robot.update_configuration(q0)
        fk_tcp_t0 = np.array(robot.get_frame_pose("panda_hand_tcp").translation, dtype=float)
        base_offset = demo_ee[0] - fk_tcp_t0

        final_y = predicted_pos[-1, 1]
        y_error = abs(final_y - P2_PLATE_Y)
        direction = "CORRECT (toward plate)" if final_y < 0.15 else "WRONG (away from plate)"

        print(f"--- Seed {seed} ({T_use} steps) ---")
        print(f"  Predicted final Y: {final_y:.4f}  |  P2 plate Y: {P2_PLATE_Y:.3f}  |  Error: {y_error:.4f}  |  {direction}")
        print(f"  Base offset Z: {base_offset[2]:.3f}")

        q_traj = np.zeros((T_use, robot.nq))
        q_current = q0.copy()
        errors = []

        for t in range(T_use):
            target_pos = hybrid_pos[t] - base_offset

            robot.update_configuration(q_current)

            # Use ORIGINAL demo quaternion (not predicted)
            q_wxyz = hybrid_quat[t]
            target_rot = R_scipy.from_quat(q_wxyz[[1, 2, 3, 0]]).as_matrix()

            target_T = np.eye(4, order="F")
            target_T[:3, :3] = target_rot
            target_T[:3, 3] = target_pos

            result = solver.solve_position(q_current, target_T, "panda_hand_tcp", ik_opts)
            if result.status == embodik.SolverStatus.SUCCESS:
                q_solution = np.array(result.q_solution)
                q_traj[t] = q_solution
                q_current = q_solution.copy()

                robot.update_configuration(q_solution)
                fk = np.array(robot.get_frame_pose("panda_hand_tcp").translation, dtype=float)
                fk_world = fk + base_offset
                err = np.linalg.norm(fk_world - hybrid_pos[t]) * 1000
                errors.append(err)
            else:
                print(f"  WARNING: IK failed at t={t}")
                q_traj[t] = q_current.copy()
                errors.append(float("nan"))

            if t % 20 == 0 or t == T_use - 1:
                robot.update_configuration(q_traj[t])
                fk = np.array(robot.get_frame_pose("panda_hand_tcp").translation, dtype=float)
                fk_world = fk + base_offset
                print(f"  t={t:3d}: target={hybrid_pos[t]}, FK={fk_world}, err={errors[-1]:.2f}mm")

        errors = np.array(errors)
        print(f"  IK: mean_err={np.nanmean(errors):.2f}mm, max={np.nanmax(errors):.2f}mm, "
              f"failures={np.sum(np.isnan(errors))}/{len(errors)}")

        all_joints[f"{key}_joints"] = q_traj
        all_errors[f"{key}_ik_errors"] = errors

    # Pass through grippers and init states from original poses
    extra = {}
    for k in data.files:
        if k.startswith("gripper_") or k.startswith("init_state_"):
            extra[k] = data[k]

    np.savez(
        JOINTS_POS_ONLY_NPZ,
        **all_joints,
        **all_errors,
        **extra,
        n_demos=data["n_demos"],
        source=np.array(["pos_only_debug"]),
    )

    # Summary report
    print("\n" + "=" * 70)
    print("POSITION PREDICTION SUMMARY")
    print("=" * 70)
    correct_count = 0
    for seed in range(n_demos):
        pos_key = f"seed{seed}_pos"
        if pos_key not in data:
            continue
        final_y = data[pos_key][-1, 1]
        is_correct = final_y < 0.15
        if is_correct:
            correct_count += 1
        status = "OK" if is_correct else "WRONG"
        print(f"  Seed {seed}: final Y = {final_y:.4f}  [{status}]  (plate at Y={P2_PLATE_Y:.3f})")

    print(f"\n  Correct direction: {correct_count}/{n_demos} seeds have final Y < 0.15")
    print(f"  Position error from plate: see per-seed |final_Y - {P2_PLATE_Y}|")
    print(f"\nSaved pos-only joint trajectories to {JOINTS_POS_ONLY_NPZ}")
    print("Next: run --stage sim in libero env")


# =========================================================================
# Stage: sim execution
# =========================================================================

def stage_sim(args):
    """Execute pos-only joint trajectories in P2-perturbed LIBERO."""
    os.environ.setdefault("MUJOCO_GL", "egl")

    from libero.libero.envs import OffScreenRenderEnv

    print("=" * 70)
    print("POSITION-ONLY DEBUG: LIBERO P2 sim (predicted pos + original quats)")
    print("=" * 70)

    # Generate P2 BDDL
    generate_p2_bddl()

    # Load joint trajectories
    data = np.load(JOINTS_POS_ONLY_NPZ, allow_pickle=True)
    n_demos = int(data["n_demos"][0])
    print(f"Source: {str(data['source'])}")
    print(f"N demos: {n_demos}")

    results = []
    successes = 0
    total = 0

    for seed in range(n_demos):
        key = f"seed{seed}"
        joints_key = f"{key}_joints"
        if joints_key not in data:
            print(f"  SKIP: {joints_key} not found")
            continue

        q_traj = data[joints_key]
        gripper_key = f"gripper_{key}"
        gripper = data[gripper_key] if gripper_key in data else np.ones(len(q_traj)) * -1.0
        init_state_key = f"init_state_{key}"
        init_state = data[init_state_key] if init_state_key in data else None

        if init_state is None:
            print(f"  SKIP: no init_state for seed {seed}")
            continue

        for attempt, (output_max, kp, substeps) in enumerate([
            (0.5, 150, 3),
            (1.0, 250, 5),
        ]):
            env = OffScreenRenderEnv(
                bddl_file_name=BDDL_PERTURBED,
                controller="JOINT_POSITION",
                camera_heights=256,
                camera_widths=256,
            )
            obs = env.reset()
            env.set_init_state(init_state)
            obs = env.env._get_observations()

            robot = env.env.robots[0]
            ctrl = robot.controller

            # Override controller gains
            ctrl.output_max = ctrl.nums2array(output_max, len(ctrl.qpos_index))
            ctrl.output_min = ctrl.nums2array(-output_max, len(ctrl.qpos_index))
            ctrl.kp = ctrl.nums2array(kp, len(ctrl.qpos_index))
            ctrl.kd = 2 * np.sqrt(ctrl.kp) * 1.0
            ctrl.action_scale = None

            label = f"seed={seed}" + (f" (retry, kp={kp})" if attempt > 0 else "")
            print(f"\n--- {label} ({len(q_traj)} steps, output_max={output_max}) ---")

            n_steps = len(q_traj)
            reward = 0.0
            success_at = None

            for t in range(n_steps):
                grip_idx = min(t, len(gripper) - 1)
                for sub in range(substeps):
                    q_target = q_traj[t, :7]
                    q_current = np.array(robot._joint_positions)
                    delta = q_target - q_current
                    action = np.clip(delta / output_max, -1.0, 1.0)
                    grip = gripper[grip_idx]
                    full_action = np.concatenate([action, [grip]])
                    obs, r, done, info = env.step(full_action)
                    if r > reward:
                        reward = r

                success = env.check_success()
                if success and success_at is None:
                    success_at = t

                if t % 20 == 0 or t == n_steps - 1:
                    ee_pos = np.array(robot.controller.ee_pos)
                    q_now = np.array(robot._joint_positions)
                    q_err = np.max(np.abs(q_now - q_traj[t, :7]))
                    print(f"  t={t:3d}: EE=[{ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}] "
                          f"q_err={q_err:.4f} grip={gripper[grip_idx]:.0f} "
                          f"{'SUCCESS' if success else ''}")

            # Settling
            for i in range(100):
                q_target = q_traj[-1, :7]
                q_current = np.array(robot._joint_positions)
                delta = q_target - q_current
                action = np.clip(delta / output_max, -1.0, 1.0)
                full_action = np.concatenate([action, [1.0]])
                obs, r, done, info = env.step(full_action)
                if r > reward:
                    reward = r
                success = env.check_success()
                if success and success_at is None:
                    success_at = n_steps + i

            final_success = env.check_success()
            ee_final = np.array(robot.controller.ee_pos)
            print(f"  Final: EE={ee_final}, success={final_success}, reward={reward:.2f}")
            if success_at is not None:
                print(f"  First success at step {success_at}")

            env.close()

            if final_success:
                break

        trial_result = {
            "seed": seed,
            "success": bool(final_success),
            "reward": float(reward),
            "success_at": success_at,
            "method": "dhb_pos_only_debug",
        }
        results.append(trial_result)
        if final_success:
            successes += 1
        total += 1

    rate = successes / max(total, 1) * 100

    print("\n" + "=" * 70)
    print(f"POS-ONLY DEBUG on P2: {successes}/{total} = {rate:.0f}%")
    print("  (predicted positions + original demo quaternions)")
    print("=" * 70)

    output = {
        "method": "dhb_pos_only_debug",
        "perturbation": "P2_position_swap",
        "successes": successes,
        "total": total,
        "rate": rate,
        "trials": results,
    }

    with open(RESULTS_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {RESULTS_JSON}")


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Position-only debug eval for DHB-XR P2 pipeline"
    )
    parser.add_argument(
        "--stage", required=True,
        choices=["ik", "sim"],
        help="Which stage to run (predict stage reuses existing poses NPZ)",
    )
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of demo seeds to evaluate")
    args = parser.parse_args()

    if args.stage == "ik":
        stage_ik(args)
    elif args.stage == "sim":
        stage_sim(args)
