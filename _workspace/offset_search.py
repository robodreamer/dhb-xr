#!/usr/bin/env python
"""Search for LIBERO task + offset configs where DHB-XR + embodik IK achieves 100% success.

Two-stage pipeline:
  Stage 1 (IK): pixi run python (Python 3.12 with embodik)
  Stage 2 (sim): libero env python (Python 3.10 with robosuite)

Usage:
  # Stage 1 - IK solve
  cd ~/Projects/repos/embodik-singularity-recovery && \
    pixi run python ~/Projects/repos/dhb_xr/_workspace/offset_search.py \
      --stage ik --task 0 --offset-axis Y --offset-m 0.03

  # Stage 2 - Sim evaluate
  MUJOCO_GL=egl ~/miniforge3/envs/libero/bin/python \
    ~/Projects/repos/dhb_xr/_workspace/offset_search.py \
      --stage sim --task 0 --offset-axis Y --offset-m 0.03
"""
import sys
import os
import json
import argparse
import numpy as np

# LIBERO spatial tasks (index -> description, hdf5 name, bddl name)
LIBERO_SPATIAL_DIR = "/home/andypark/Projects/data/libero/libero_spatial"
BDDL_DIR = "/home/andypark/Projects/repos/LIBERO/libero/libero/bddl_files/libero_spatial"

TASKS = [
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
]

AXIS_MAP = {"X": 0, "Y": 1, "Z": 2}


def get_paths(task_idx):
    name = TASKS[task_idx]
    return {
        "hdf5": os.path.join(LIBERO_SPATIAL_DIR, f"{name}_demo.hdf5"),
        "bddl": os.path.join(BDDL_DIR, f"{name}.bddl"),
    }


def npz_path(task_idx, axis, offset_m, demo_idx=0):
    return f"/tmp/offset_search_t{task_idx}_{axis}_{int(offset_m*1000)}mm_d{demo_idx}.npz"


def stage_ik(task_idx, axis, offset_m, demo_idx=0):
    """Solve IK for original + offset trajectories."""
    import embodik
    from robot_descriptions import panda_description
    import h5py

    paths = get_paths(task_idx)
    axis_idx = AXIS_MAP[axis]

    print("=" * 70)
    print(f"IK SOLVE: task={task_idx}, axis={axis}, offset={offset_m*100:.1f}cm, demo={demo_idx}")
    print(f"  Task: {TASKS[task_idx]}")
    print("=" * 70)

    # Load demo
    f = h5py.File(paths["hdf5"], "r")
    demo = f[f"data/demo_{demo_idx}"]
    demo_ee_pos = demo["obs/ee_pos"][:]
    demo_joint = demo["obs/joint_states"][:]
    demo_gripper = demo["obs/gripper_states"][:]
    demo_states = demo["states"][:]
    init_state = demo_states[0]
    f.close()

    print(f"Demo: {len(demo_ee_pos)} steps")
    print(f"EE pos range: X=[{demo_ee_pos[:,0].min():.3f},{demo_ee_pos[:,0].max():.3f}] "
          f"Y=[{demo_ee_pos[:,1].min():.3f},{demo_ee_pos[:,1].max():.3f}] "
          f"Z=[{demo_ee_pos[:,2].min():.3f},{demo_ee_pos[:,2].max():.3f}]")

    # Build robot
    robot = embodik.RobotModel(urdf_path=panda_description.URDF_PATH)
    q0 = np.zeros(robot.nq)
    q0[:7] = demo_joint[0]
    robot.update_configuration(q0)
    fk_tcp_t0 = np.array(robot.get_frame_pose("panda_hand_tcp").translation, dtype=float)
    base_offset = demo_ee_pos[0] - fk_tcp_t0
    print(f"Base offset: {base_offset}")

    solver = embodik.KinematicsSolver(robot)

    ik_opts = embodik.PositionIKOptions()
    ik_opts.position_tolerance = 0.0005
    ik_opts.orientation_tolerance = 100.0
    ik_opts.orientation_gain = 0.1
    ik_opts.max_iterations = 200

    def solve_trajectory(ee_positions, label=""):
        q_traj = np.zeros((len(ee_positions), robot.nq))
        q_current = q0.copy()
        errors = []

        for t in range(len(ee_positions)):
            target_pos = ee_positions[t] - base_offset
            target_T = np.eye(4, order="F")

            robot.update_configuration(q_current)
            current_rot = np.array(robot.get_frame_pose("panda_hand_tcp").rotation, dtype=float)
            target_T[:3, :3] = current_rot
            target_T[:3, 3] = target_pos

            result = solver.solve_position(q_current, target_T, "panda_hand_tcp", ik_opts)
            if result.status == embodik.SolverStatus.SUCCESS:
                q_solution = np.array(result.q_solution)
                q_traj[t] = q_solution
                q_current = q_solution.copy()
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
                print(f"  t={t:3d}: target={ee_positions[t]}, FK={fk_world}, err={errors[-1]:.2f}mm")

        errors = np.array(errors)
        n_fail = np.sum(np.isnan(errors))
        print(f"  {label} mean err: {np.nanmean(errors):.2f}mm, max: {np.nanmax(errors):.2f}mm, "
              f"IK failures: {n_fail}/{len(errors)}")
        return q_traj, errors

    # Original trajectory
    print("\n--- Solving IK for 0cm (baseline) ---")
    q_0cm, errors_0cm = solve_trajectory(demo_ee_pos, "0cm")

    # Offset trajectory
    print(f"\n--- Solving IK for {offset_m*100:.1f}cm {axis}-offset ---")
    ee_offset = demo_ee_pos.copy()
    ee_offset[:, axis_idx] += offset_m
    q_off, errors_off = solve_trajectory(ee_offset, f"{offset_m*100:.1f}cm {axis}")

    # Gripper commands
    grip_qpos = demo_gripper[:, 0]
    gripper_cmd = np.where(grip_qpos > 0.015, -1.0, 1.0)

    out = npz_path(task_idx, axis, offset_m, demo_idx)
    np.savez(
        out,
        q_0cm=q_0cm,
        q_offset=q_off,
        gripper=gripper_cmd,
        init_state=init_state,
        base_offset=base_offset,
        demo_ee_pos=demo_ee_pos,
        errors_0cm=errors_0cm,
        errors_offset=errors_off,
        offset_axis=axis_idx,
        offset_m=offset_m,
    )
    print(f"\nSaved to {out}")

    # Quick check: any IK failures for offset?
    n_fail = np.sum(np.isnan(errors_off))
    if n_fail > 0:
        print(f"WARNING: {n_fail} IK failures in offset trajectory - may not succeed in sim")
    return n_fail == 0


def stage_sim(task_idx, axis, offset_m, demo_idx=0):
    """Execute IK trajectory in LIBERO sim."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, "/home/andypark/Projects/repos/LIBERO")

    from libero.libero.envs.env_wrapper import OffScreenRenderEnv

    paths = get_paths(task_idx)
    axis_idx = AXIS_MAP[axis]
    data_path = npz_path(task_idx, axis, offset_m, demo_idx)

    print("=" * 70)
    print(f"SIM EVAL: task={task_idx}, axis={axis}, offset={offset_m*100:.1f}cm, demo={demo_idx}")
    print(f"  Task: {TASKS[task_idx]}")
    print("=" * 70)

    data = np.load(data_path)
    q_offset = data["q_offset"]
    gripper = data["gripper"]
    init_state = data["init_state"].copy()

    # Shift the bowl in init_state
    # Bowl 1 positions: init_state[10]=X, init_state[11]=Y, init_state[12]=Z
    bowl_x_idx = 10
    bowl_pos_idx = bowl_x_idx + axis_idx
    original_bowl = init_state[bowl_pos_idx]
    init_state[bowl_pos_idx] += offset_m
    print(f"Bowl position[{axis}]: {original_bowl:.4f} -> {init_state[bowl_pos_idx]:.4f} "
          f"(shifted by {offset_m*100:.1f}cm)")

    def run_trajectory(q_traj, gripper_cmds, init_st, label="",
                       output_max=0.5, kp=150, substeps=3, record_path=None):
        import cv2
        env = OffScreenRenderEnv(
            bddl_file_name=paths["bddl"],
            controller="JOINT_POSITION",
            camera_heights=256,
            camera_widths=256,
            camera_names=["agentview", "robot0_eye_in_hand"],
        )
        obs = env.reset()
        env.set_init_state(init_st)
        obs = env.env._get_observations()

        robot = env.env.robots[0]
        ctrl = robot.controller
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
            frames_this_step = []
            for sub in range(substeps):
                q_target = q_traj[t, :7]
                q_current = np.array(robot._joint_positions)
                delta = q_target - q_current
                action = np.clip(delta / output_max, -1.0, 1.0)
                grip = gripper_cmds[t]
                full_action = np.concatenate([action, [grip]])
                obs, reward, done, info = env.step(full_action)
                if record_path and sub == 0:
                    av = cv2.flip(cv2.cvtColor(obs['agentview_image'].astype(np.uint8), cv2.COLOR_RGB2BGR), 0)
                    eih = cv2.flip(cv2.cvtColor(obs.get('robot0_eye_in_hand_image', obs['agentview_image']).astype(np.uint8), cv2.COLOR_RGB2BGR), 0)
                    div = np.full((256, 4, 3), 60, dtype=np.uint8)
                    frame = np.concatenate([av, div, eih], axis=1)
                    cv2.putText(frame, label, (6, 20), cv2.FONT_HERSHEY_DUPLEX, 0.48, (50,200,50), 1)
                    frames_this_step.append(frame)

            if record_path and frames_this_step:
                if not hasattr(run_trajectory, '_frames'):
                    run_trajectory._frames = []
                run_trajectory._frames.extend(frames_this_step)

            ee_pos = np.array(robot.controller.ee_pos)
            q_now = np.array(robot._joint_positions)
            q_err = np.max(np.abs(q_now - q_traj[t, :7]))
            ee_positions.append(ee_pos.copy())
            q_errors.append(q_err)

            success = env.check_success()
            if success and success_at is None:
                success_at = t

            if t % 20 == 0 or t == n_steps - 1:
                bowl = env.sim.data.get_body_xpos("akita_black_bowl_1_main")
                plate = env.sim.data.get_body_xpos("plate_1_main")
                print(f"  t={t:3d}: EE=[{ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}] "
                      f"q_err={q_err:.4f} "
                      f"bowl=[{bowl[0]:.3f},{bowl[1]:.3f},{bowl[2]:.3f}] "
                      f"plate=[{plate[0]:.3f},{plate[1]:.3f},{plate[2]:.3f}] "
                      f"{'SUCCESS' if success else ''}")

        # Settling
        print("  --- Settling ---")
        for i in range(30):
            q_target = q_traj[-1, :7]
            q_current = np.array(robot._joint_positions)
            delta = q_target - q_current
            action = np.clip(delta / output_max, -1.0, 1.0)
            full_action = np.concatenate([action, [gripper_cmds[-1]]])
            obs, reward, done, info = env.step(full_action)
            success = env.check_success()
            if success and success_at is None:
                success_at = n_steps + i

        final_success = env.check_success()
        ee_final = np.array(robot.controller.ee_pos)
        bowl_final = env.sim.data.get_body_xpos("akita_black_bowl_1_main").copy()
        plate_final = env.sim.data.get_body_xpos("plate_1_main").copy()

        bowl_plate_dist = np.linalg.norm(bowl_final[:2] - plate_final[:2]) * 100

        print(f"  RESULT: success={final_success}")
        print(f"  Bowl: {bowl_final}, Plate: {plate_final}")
        print(f"  Bowl-plate 2D dist: {bowl_plate_dist:.1f}cm")
        if success_at is not None:
            print(f"  First success at step {success_at}")

        # Save video if recording
        if record_path and hasattr(run_trajectory, '_frames') and run_trajectory._frames:
            if final_success:
                # Add SUCCESS badge to last frames
                for i in range(min(30, len(run_trajectory._frames))):
                    idx = len(run_trajectory._frames) - 1 - i
                    cv2.putText(run_trajectory._frames[idx], 'TASK SUCCESS',
                                (80, 240), cv2.FONT_HERSHEY_DUPLEX, 0.65, (50,200,50), 2)
                # Hold final frame
                for _ in range(30):
                    run_trajectory._frames.append(run_trajectory._frames[-1])
            h, w = run_trajectory._frames[0].shape[:2]
            tmp = record_path.replace('.mp4', '_raw.mp4')
            vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
            for fr in run_trajectory._frames: vw.write(fr)
            vw.release()
            import subprocess
            subprocess.run(['ffmpeg','-y','-i',tmp,'-c:v','libx264','-crf','17','-pix_fmt','yuv420p',record_path], capture_output=True)
            os.unlink(tmp)
            print(f"  Video: {record_path} ({os.path.getsize(record_path)//1024}KB, {len(run_trajectory._frames)} frames)")
            run_trajectory._frames = []

        env.close()
        return {
            "success": bool(final_success),
            "success_at": success_at,
            "ee_final": ee_final.tolist(),
            "q_errors_max": float(np.max(q_errors)),
            "q_errors_mean": float(np.mean(q_errors)),
            "bowl_final": bowl_final.tolist(),
            "plate_final": plate_final.tolist(),
            "bowl_plate_dist_cm": round(bowl_plate_dist, 2),
        }

    # Run offset trajectory with shifted bowl
    record = os.environ.get("RECORD_VIDEO")
    result = run_trajectory(q_offset, gripper, init_state,
                           f"task{task_idx} {offset_m*100:.1f}cm {axis}",
                           record_path=record)

    # If failed, try higher gains
    if not result["success"]:
        print("\n--- Retrying with higher gains ---")
        # Re-create init_state (it may have been consumed)
        init_state2 = data["init_state"].copy()
        init_state2[bowl_pos_idx] += offset_m
        result2 = run_trajectory(q_offset, gripper, init_state2,
                                f"task{task_idx} {offset_m*100:.1f}cm {axis} (high gains)",
                                output_max=1.0, kp=250, substeps=5)
        if result2["success"]:
            result = result2

    # Print final verdict
    print("\n" + "=" * 70)
    status = "SUCCESS" if result["success"] else "FAILED"
    print(f"VERDICT: {status} | task={task_idx} axis={axis} offset={offset_m*100:.1f}cm")
    print(f"  Max q error: {result['q_errors_max']:.4f} rad")
    print(f"  Bowl-plate dist: {result['bowl_plate_dist_cm']:.2f}cm")
    print("=" * 70)

    # Append to results log
    log_entry = {
        "task_idx": task_idx,
        "task_name": TASKS[task_idx],
        "demo_idx": demo_idx,
        "axis": axis,
        "offset_m": offset_m,
        "offset_cm": round(offset_m * 100, 1),
        **result,
    }

    log_file = "/tmp/offset_search_log.json"
    if os.path.exists(log_file):
        with open(log_file) as f:
            log = json.load(f)
    else:
        log = []
    log.append(log_entry)
    with open(log_file, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Log appended to {log_file}")

    return result["success"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["ik", "sim"], required=True)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--offset-axis", choices=["X", "Y", "Z"], default="Y")
    parser.add_argument("--offset-m", type=float, default=0.03)
    parser.add_argument("--demo", type=int, default=0)
    args = parser.parse_args()

    if args.stage == "ik":
        stage_ik(args.task, args.offset_axis, args.offset_m, args.demo)
    else:
        stage_sim(args.task, args.offset_axis, args.offset_m, args.demo)
