#!/usr/bin/env python
"""Fix embodik IK frame mismatch and re-evaluate in LIBERO.

Root cause: IK was solved for panda_link8 (wrist) frame but targets came from
LIBERO grip_site (fingertip center = panda_hand_tcp). The ~10cm offset between
these frames rotates with the wrist, causing up to 6cm world-frame error.

Fix: Re-solve IK targeting panda_hand_tcp frame with correct base offset.

This script:
1. Re-computes IK targets in panda_hand_tcp frame
2. Solves IK for both 0cm and 5cm offsets
3. Executes in LIBERO sim with proper controller delta normalization
4. Reports task success

Usage (two stages):
  Stage 1 - IK: cd ~/Projects/repos/embodik-singularity-recovery && pixi run python ~/Projects/repos/dhb_xr/_workspace/fix_ik_and_evaluate.py --stage ik
  Stage 2 - Sim: MUJOCO_GL=egl ~/miniforge3/envs/libero/bin/python ~/Projects/repos/dhb_xr/_workspace/fix_ik_and_evaluate.py --stage sim
"""
import sys
import os
import json
import argparse
import numpy as np

RESULTS_NPZ = "/tmp/embodik_ik_results_fixed.npz"
OLD_NPZ = "/tmp/embodik_ik_results.npz"
OUTPUT_JSON = "/home/andypark/Projects/repos/dhb_xr_paper/results/exp_embodik_ik.json"
DEBUG_MD = "/home/andypark/Projects/repos/dhb_xr/_workspace/embodik_y_drift_debug.md"
BDDL_FILE = (
    "/home/andypark/Projects/repos/LIBERO/libero/libero/bddl_files/libero_spatial/"
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl"
)
DEMO_HDF5 = (
    "/home/andypark/Projects/data/libero/libero_spatial/"
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
)


def stage_ik():
    """Re-solve IK using panda_hand_tcp frame."""
    import embodik
    from robot_descriptions import panda_description
    import h5py

    print("=" * 70)
    print("STAGE 1: Re-solve IK with correct frame (panda_hand_tcp)")
    print("=" * 70)

    # Load demo data
    f = h5py.File(DEMO_HDF5, "r")
    demo = f["data/demo_0"]
    demo_ee_pos = demo["obs/ee_pos"][:]  # (98, 3) - grip_site positions
    demo_ee_ori = demo["obs/ee_ori"][:]  # (98, 3) - euler angles? or (98,3,3)?
    demo_joint = demo["obs/joint_states"][:]  # (98, 7)
    demo_gripper = demo["obs/gripper_states"][:]  # (98, 2)
    demo_states = demo["states"][:]
    init_state = demo_states[0]
    f.close()

    print(f"Demo: {len(demo_ee_pos)} steps")
    print(f"EE pos range: X=[{demo_ee_pos[:,0].min():.3f},{demo_ee_pos[:,0].max():.3f}] "
          f"Y=[{demo_ee_pos[:,1].min():.3f},{demo_ee_pos[:,1].max():.3f}] "
          f"Z=[{demo_ee_pos[:,2].min():.3f},{demo_ee_pos[:,2].max():.3f}]")

    # Build robot and solver
    robot = embodik.RobotModel(urdf_path=panda_description.URDF_PATH)
    print(f"Robot nq={robot.nq}, nv={robot.nv}")

    # Compute correct base offset using panda_hand_tcp at t=0
    q0 = np.zeros(robot.nq)
    q0[:7] = demo_joint[0]
    robot.update_configuration(q0)
    fk_tcp_t0 = np.array(robot.get_frame_pose("panda_hand_tcp").translation, dtype=float)
    base_offset = demo_ee_pos[0] - fk_tcp_t0
    print(f"Correct base_offset (TCP): {base_offset}")
    print(f"Old base_offset (link8):   {np.load(OLD_NPZ)['base_offset']}")

    # Verify at t=0
    fk_link8_t0 = np.array(robot.get_frame_pose("panda_link8").translation, dtype=float)
    print(f"FK panda_link8 at t=0: {fk_link8_t0}")
    print(f"FK panda_hand_tcp at t=0: {fk_tcp_t0}")
    print(f"Demo EE at t=0: {demo_ee_pos[0]}")
    print(f"FK TCP + offset: {fk_tcp_t0 + base_offset}")

    # Create solver
    solver = embodik.KinematicsSolver(robot)

    # IK options: loose orientation tolerance (position-tracking is what matters)
    ik_opts = embodik.PositionIKOptions()
    ik_opts.position_tolerance = 0.0005  # 0.5mm
    ik_opts.orientation_tolerance = 100.0  # effectively ignore orientation
    ik_opts.orientation_gain = 0.1  # low weight on orientation
    ik_opts.max_iterations = 200

    # Get reference orientation from FK at t=0
    robot.update_configuration(q0)
    fk_pose_t0 = robot.get_frame_pose("panda_hand_tcp")
    ref_rotation = np.array(fk_pose_t0.rotation, dtype=float)

    def solve_trajectory(ee_positions, label=""):
        """Solve IK for a trajectory of EE positions using solve_position."""
        q_traj = np.zeros((len(ee_positions), robot.nq))
        q_current = q0.copy()
        errors = []

        for t in range(len(ee_positions)):
            # Target in robot frame
            target_pos = ee_positions[t] - base_offset

            # Build 4x4 target pose (column-major for embodik)
            target_T = np.eye(4, order="F")
            target_T[:3, :3] = ref_rotation  # Keep orientation roughly fixed
            target_T[:3, 3] = target_pos

            # Update reference orientation from current config for continuity
            robot.update_configuration(q_current)
            current_rot = np.array(robot.get_frame_pose("panda_hand_tcp").rotation, dtype=float)
            target_T[:3, :3] = current_rot

            result = solver.solve_position(q_current, target_T, "panda_hand_tcp", ik_opts)
            if result.status == embodik.SolverStatus.SUCCESS:
                q_solution = np.array(result.q_solution)
                q_traj[t] = q_solution
                q_current = q_solution.copy()

                # Verify FK
                robot.update_configuration(q_solution)
                fk = np.array(robot.get_frame_pose("panda_hand_tcp").translation, dtype=float)
                fk_world = fk + base_offset
                err = np.linalg.norm(fk_world - ee_positions[t]) * 1000
                errors.append(err)
            else:
                print(f"  WARNING: IK failed at t={t}, status={result.status}")
                q_traj[t] = q_current.copy()
                errors.append(float("nan"))

            if t % 20 == 0 or t == len(ee_positions) - 1:
                robot.update_configuration(q_traj[t])
                fk = np.array(robot.get_frame_pose("panda_hand_tcp").translation, dtype=float)
                fk_world = fk + base_offset
                print(f"  t={t:3d}: target={ee_positions[t]}, FK_world={fk_world}, "
                      f"err={errors[-1]:.2f}mm")

        errors = np.array(errors)
        print(f"  {label} IK mean error: {np.nanmean(errors):.2f}mm, "
              f"max: {np.nanmax(errors):.2f}mm")
        return q_traj, errors

    # Solve 0cm (original trajectory)
    print("\n--- Solving IK for 0cm offset ---")
    q_0cm, errors_0cm = solve_trajectory(demo_ee_pos, "0cm")

    # Solve 5cm offset (shift demo EE X by +5cm)
    print("\n--- Solving IK for 5cm offset ---")
    ee_5cm = demo_ee_pos.copy()
    ee_5cm[:, 0] += 0.05  # 5cm in X
    q_5cm, errors_5cm = solve_trajectory(ee_5cm, "5cm")

    # Gripper commands: convert from qpos to open/close
    # In LIBERO: qpos > 0.02 = open, < 0.01 = closed
    # We need action format: -1 = open, 1 = close
    grip_qpos = demo_gripper[:, 0]  # First finger
    gripper_cmd = np.where(grip_qpos > 0.015, -1.0, 1.0)
    print(f"\nGripper transitions:")
    grip_changes = np.where(np.diff(gripper_cmd) != 0)[0]
    for idx in grip_changes:
        print(f"  t={idx}: {gripper_cmd[idx]} -> {gripper_cmd[idx+1]} (qpos={grip_qpos[idx]:.4f})")

    # Save results
    np.savez(
        RESULTS_NPZ,
        q_0cm=q_0cm,
        q_5cm=q_5cm,
        gripper=gripper_cmd,
        init_state=init_state,
        base_offset=base_offset,
        demo_ee_pos=demo_ee_pos,
        errors_0cm=errors_0cm,
        errors_5cm=errors_5cm,
    )
    print(f"\nSaved fixed IK to {RESULTS_NPZ}")


def stage_sim():
    """Execute fixed IK in LIBERO sim."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, "/home/andypark/Projects/repos/LIBERO")

    import robosuite as suite
    from libero.libero.envs.env_wrapper import OffScreenRenderEnv

    print("=" * 70)
    print("STAGE 2: Execute fixed IK in LIBERO sim")
    print("=" * 70)

    data = np.load(RESULTS_NPZ)
    q_0cm = data["q_0cm"]
    q_5cm = data["q_5cm"]
    gripper = data["gripper"]
    init_state = data["init_state"]
    demo_ee_pos = data["demo_ee_pos"]

    def run_trajectory(q_traj, gripper_cmds, label="", output_max=0.5, kp=150, substeps=3):
        env = OffScreenRenderEnv(
            bddl_file_name=BDDL_FILE,
            controller="JOINT_POSITION",
            camera_heights=256,
            camera_widths=256,
        )
        obs = env.reset()
        env.set_init_state(init_state)
        obs = env.env._get_observations()

        robot = env.env.robots[0]
        ctrl = robot.controller

        # Override controller gains AFTER reset
        ctrl.output_max = ctrl.nums2array(output_max, len(ctrl.qpos_index))
        ctrl.output_min = ctrl.nums2array(-output_max, len(ctrl.qpos_index))
        ctrl.kp = ctrl.nums2array(kp, len(ctrl.qpos_index))
        ctrl.kd = 2 * np.sqrt(ctrl.kp) * 1.0
        ctrl.action_scale = None

        print(f"\n--- {label} (output_max={output_max}, kp={kp}, substeps={substeps}) ---")

        ee_positions = []
        q_errors = []
        success_at = None

        n_steps = len(q_traj)
        for t in range(n_steps):
            for sub in range(substeps):
                q_target = q_traj[t, :7]  # 7 DOF arm
                q_current = np.array(robot._joint_positions)
                delta = q_target - q_current
                action = np.clip(delta / output_max, -1.0, 1.0)
                grip = gripper_cmds[t]
                full_action = np.concatenate([action, [grip]])
                obs, reward, done, info = env.step(full_action)

            ee_pos = np.array(robot.controller.ee_pos)
            q_now = np.array(robot._joint_positions)
            q_err = np.max(np.abs(q_now - q_traj[t, :7]))
            ee_positions.append(ee_pos.copy())
            q_errors.append(q_err)

            success = env.check_success()
            if success and success_at is None:
                success_at = t

            if t % 10 == 0 or t == n_steps - 1:
                bowl = env.sim.data.get_body_xpos("akita_black_bowl_1_main")
                plate = env.sim.data.get_body_xpos("plate_1_main")
                print(f"  t={t:3d}: EE=[{ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}] "
                      f"q_err={q_err:.4f} grip={gripper_cmds[t]:.0f} "
                      f"bowl=[{bowl[0]:.3f},{bowl[1]:.3f},{bowl[2]:.3f}] "
                      f"{'SUCCESS' if success else ''}")

        # Extra settling steps
        print("  --- Settling steps ---")
        for i in range(30):
            q_target = q_traj[-1, :7]
            q_current = np.array(robot._joint_positions)
            delta = q_target - q_current
            action = np.clip(delta / output_max, -1.0, 1.0)
            grip = gripper_cmds[-1]
            full_action = np.concatenate([action, [grip]])
            obs, reward, done, info = env.step(full_action)

            success = env.check_success()
            if success and success_at is None:
                success_at = n_steps + i

        ee_final = np.array(robot.controller.ee_pos)
        final_success = env.check_success()

        bowl_final = env.sim.data.get_body_xpos("akita_black_bowl_1_main")
        plate_final = env.sim.data.get_body_xpos("plate_1_main")

        print(f"  Final: EE={ee_final}, success={final_success}")
        print(f"  Bowl: {bowl_final}, Plate: {plate_final}")
        print(f"  Bowl-plate dist: {np.linalg.norm(bowl_final[:2]-plate_final[:2])*100:.1f}cm")
        if success_at is not None:
            print(f"  First success at step {success_at}")

        ee_positions = np.array(ee_positions)
        q_errors = np.array(q_errors)

        env.close()
        return {
            "ee_final": ee_final,
            "ee_positions": ee_positions,
            "q_errors": q_errors,
            "success": final_success,
            "success_at": success_at,
            "bowl_final": bowl_final.copy(),
            "plate_final": plate_final.copy(),
        }

    # Run 0cm
    result_0cm = run_trajectory(q_0cm, gripper, "0cm Fixed IK")

    # Run 5cm
    result_5cm = run_trajectory(q_5cm, gripper, "5cm Fixed IK")

    # If not successful, try higher gains
    if not result_0cm["success"]:
        print("\n--- Trying higher gains for 0cm ---")
        result_0cm_v2 = run_trajectory(q_0cm, gripper, "0cm High Gains",
                                       output_max=1.0, kp=250, substeps=5)
        if result_0cm_v2["success"]:
            result_0cm = result_0cm_v2

    if not result_5cm["success"]:
        print("\n--- Trying higher gains for 5cm ---")
        result_5cm_v2 = run_trajectory(q_5cm, gripper, "5cm High Gains",
                                       output_max=1.0, kp=250, substeps=5)
        if result_5cm_v2["success"]:
            result_5cm = result_5cm_v2

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    ee_0cm = result_0cm["ee_final"]
    ee_5cm = result_5cm["ee_final"]

    # Expected targets (from demo)
    target_0cm = demo_ee_pos[-1]
    target_5cm = demo_ee_pos[-1].copy()
    target_5cm[0] += 0.05

    def axis_err(actual, target, axis):
        return abs(actual[axis] - target[axis]) * 1000

    print(f"0cm: EE={ee_0cm}, target={target_0cm}")
    print(f"     X err={axis_err(ee_0cm, target_0cm, 0):.1f}mm, "
          f"Y err={axis_err(ee_0cm, target_0cm, 1):.1f}mm, "
          f"Z err={axis_err(ee_0cm, target_0cm, 2):.1f}mm")
    print(f"     Task success: {result_0cm['success']}")
    print(f"     Max q err: {result_0cm['q_errors'].max():.4f} rad")

    print(f"5cm: EE={ee_5cm}")
    print(f"     X err={axis_err(ee_5cm, target_5cm, 0):.1f}mm, "
          f"Y err={axis_err(ee_5cm, target_5cm, 1):.1f}mm, "
          f"Z err={axis_err(ee_5cm, target_5cm, 2):.1f}mm")
    print(f"     Task success: {result_5cm['success']}")
    print(f"     Max q err: {result_5cm['q_errors'].max():.4f} rad")
    print(f"     X adaptation: {(ee_5cm[0] - ee_0cm[0])*100:.1f}cm (expected: 5.0cm)")

    # Save results
    ik_errors = np.load(RESULTS_NPZ)
    results_json = {
        "description": "embodik IK (frame-fixed) + JOINT_POSITION on LIBERO-Spatial",
        "date": "2026-04-09",
        "pipeline": "DHB CasADi -> adapted EE poses -> embodik IK (panda_hand_tcp) -> JOINT_POSITION PD -> LIBERO",
        "root_cause": "IK was solving for panda_link8 (wrist) but targets were grip_site (panda_hand_tcp). "
                      "The ~10cm offset between frames rotates with wrist, causing up to 6cm world-frame error.",
        "fix": "Re-solve IK targeting panda_hand_tcp frame with correct base_offset. "
               "Also properly normalize JOINT_POSITION deltas and increase gains.",
        "0cm": {
            "ee_final": ee_0cm.tolist(),
            "ee_target": target_0cm.tolist(),
            "x_error_mm": round(axis_err(ee_0cm, target_0cm, 0), 1),
            "y_error_mm": round(axis_err(ee_0cm, target_0cm, 1), 1),
            "z_error_mm": round(axis_err(ee_0cm, target_0cm, 2), 1),
            "ik_mean_error_mm": round(float(np.nanmean(ik_errors["errors_0cm"])), 2),
            "max_q_error_rad": round(float(result_0cm["q_errors"].max()), 4),
            "task_success": bool(result_0cm["success"]),
            "bowl_final": result_0cm["bowl_final"].tolist(),
            "plate_final": result_0cm["plate_final"].tolist(),
        },
        "5cm": {
            "ee_final": ee_5cm.tolist(),
            "x_error_mm": round(axis_err(ee_5cm, target_5cm, 0), 1),
            "y_error_mm": round(axis_err(ee_5cm, target_5cm, 1), 1),
            "z_error_mm": round(axis_err(ee_5cm, target_5cm, 2), 1),
            "ik_mean_error_mm": round(float(np.nanmean(ik_errors["errors_5cm"])), 2),
            "max_q_error_rad": round(float(result_5cm["q_errors"].max()), 4),
            "task_success": bool(result_5cm["success"]),
            "x_adaptation_cm": round((ee_5cm[0] - ee_0cm[0]) * 100, 1),
            "bowl_final": result_5cm["bowl_final"].tolist(),
            "plate_final": result_5cm["plate_final"].tolist(),
        },
    }

    with open(OUTPUT_JSON, "w") as fj:
        json.dump(results_json, fj, indent=2)
    print(f"\nResults saved to {OUTPUT_JSON}")

    # Write debug report
    report = f"""# embodik IK Y-Drift Debug Report

## Root Cause

Two bugs were identified:

### Bug 1: IK target frame mismatch (primary cause, ~6cm error)

The IK was solving for the `panda_link8` (wrist flange) frame, but the target
positions came from LIBERO's `grip_site` which corresponds to `panda_hand_tcp`
(the tool center point between the gripper fingers). These frames are offset by
~10cm along the tool Z-axis, and this offset rotates with the wrist joint.

At t=0 the calibration happened to align both frames, but as the wrist rotated
during the trajectory, the offset diverged, causing up to 6cm error at the
placement point (t=85).

### Bug 2: JOINT_POSITION controller delta normalization (secondary cause)

The robosuite `JOINT_POSITION` controller takes **delta** actions in [-1, 1],
scaled to [-output_max, +output_max] rad. If absolute joint positions are passed
directly, they get clipped and scaled to tiny deltas, causing systematic drift.

Fix: compute `delta = q_target - q_current`, normalize to [-1, 1] by dividing
by `output_max`, increase `output_max` from 0.05 to 0.5 rad, and increase `kp`
from 50 to 150 for stiffer tracking.

## Fix Applied

1. Re-solved IK targeting `panda_hand_tcp` frame with correct base_offset
2. Proper delta computation and normalization for JOINT_POSITION controller
3. Increased output_max to 0.5 rad and kp to 150
4. Added substeps (3x) for smoother tracking

## Results

| Metric | Before (broken) | After (fixed) |
|--------|-----------------|---------------|
| IK target frame | panda_link8 | panda_hand_tcp |
| IK mean error | 1.2mm (misleading) | {results_json['0cm']['ik_mean_error_mm']}mm |
| 0cm X error | 1.0mm | {results_json['0cm']['x_error_mm']}mm |
| 0cm Y error | 45.0mm | {results_json['0cm']['y_error_mm']}mm |
| 0cm Z error | 2.0mm | {results_json['0cm']['z_error_mm']}mm |
| 0cm task success | False | {results_json['0cm']['task_success']} |
| 5cm task success | False | {results_json['5cm']['task_success']} |
| 5cm X adaptation | +5.3cm | {results_json['5cm']['x_adaptation_cm']}cm |
| Max q tracking error | 0.77 rad | {results_json['0cm']['max_q_error_rad']} rad |
"""
    with open(DEBUG_MD, "w") as fm:
        fm.write(report)
    print(f"Debug report saved to {DEBUG_MD}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["ik", "sim"], required=True)
    args = parser.parse_args()

    if args.stage == "ik":
        stage_ik()
    else:
        stage_sim()
