#!/usr/bin/env python
"""Multi-axis offset search for visually striking DHB-XR demo configurations.

Two-stage pipeline:
  Stage 1 (IK): cd ~/Projects/repos/embodik-singularity-recovery && \
    pixi run python ~/Projects/repos/dhb_xr/_workspace/multi_offset_search.py --stage ik --config <name>

  Stage 2 (sim): MUJOCO_GL=egl ~/miniforge3/envs/libero/bin/python \
    ~/Projects/repos/dhb_xr/_workspace/multi_offset_search.py --stage sim --config <name>
"""
import sys
import os
import json
import argparse
import numpy as np

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

# Configurations to test — each is (name, task_idx, demo_idx, {x_offset_m, y_offset_m, z_offset_m})
CONFIGS = {
    # Combined offsets on task 0 — maximize lateral displacement for camera visibility
    "t0_Y3_X2":    (0, 0, {"x": 0.02,  "y": 0.03,  "z": 0.0}),   # 3.6cm total
    "t0_Y3_Xn2":   (0, 0, {"x": -0.02, "y": 0.03,  "z": 0.0}),   # 3.6cm total
    "t0_Yn2_X2":   (0, 0, {"x": 0.02,  "y": -0.02, "z": 0.0}),   # 2.8cm total
    "t0_Y3_Z5":    (0, 0, {"x": 0.0,   "y": 0.03,  "z": -0.05}), # 5.8cm total, Z helps reach
    "t0_X15_Y3_Zn3": (0, 0, {"x": 0.015, "y": 0.03, "z": -0.03}), # all three axes
    "t0_X2_Y3_Zn5": (0, 0, {"x": 0.02, "y": 0.03, "z": -0.05}),  # aggressive combined
    "t0_X15_Y3":   (0, 0, {"x": 0.015, "y": 0.03,  "z": 0.0}),   # known good singles
    "t0_Yn2_X15":  (0, 0, {"x": 0.015, "y": -0.02, "z": 0.0}),   # both successful singles

    # Task 1 (bowl from table center) — might have more room
    "t1_Y2_X2":    (1, 0, {"x": 0.02,  "y": 0.02,  "z": 0.0}),
    "t1_Y3_Zn3":   (1, 0, {"x": 0.0,   "y": 0.03,  "z": -0.03}),

    # Larger single-axis pushing boundaries on task 0
    "t0_Y3_Zn4":   (0, 0, {"x": 0.0,   "y": 0.03,  "z": -0.04}), # Y+3 worked, Z-4 worked
    "t0_X2_Zn5":   (0, 0, {"x": 0.02,  "y": 0.0,   "z": -0.05}), # both worked individually

    # More aggressive: push Y even further since X-2cm doesn't help IK
    "t0_Y3_Xn15":  (0, 0, {"x": -0.015, "y": 0.03, "z": 0.0}),   # X-1.5 + Y+3
    "t0_Y3_Xn1_Zn3": (0, 0, {"x": -0.01, "y": 0.03, "z": -0.03}), # 3-axis gentle
    "t0_Y3_Xn15_Zn4": (0, 0, {"x": -0.015, "y": 0.03, "z": -0.04}), # 3-axis moderate

    # Try other tasks: task 7 = bowl on stove (different layout)
    "t7_Y3":       (7, 0, {"x": 0.0,   "y": 0.03,  "z": 0.0}),
    "t7_Y3_Zn3":   (7, 0, {"x": 0.0,   "y": 0.03,  "z": -0.03}),
    "t7_X2_Y3":    (7, 0, {"x": 0.02,  "y": 0.03,  "z": 0.0}),

    # Task 5 = bowl on cookie box (elevated start)
    "t5_Y3":       (5, 0, {"x": 0.0,   "y": 0.03,  "z": 0.0}),
    "t5_Y3_Zn3":   (5, 0, {"x": 0.0,   "y": 0.03,  "z": -0.03}),
}


def get_paths(task_idx):
    name = TASKS[task_idx]
    return {
        "hdf5": os.path.join(LIBERO_SPATIAL_DIR, f"{name}_demo.hdf5"),
        "bddl": os.path.join(BDDL_DIR, f"{name}.bddl"),
    }


def npz_path(config_name):
    return f"/tmp/multi_offset_{config_name}.npz"


def stage_ik(config_name):
    """Solve IK for original + offset trajectories with multi-axis offset."""
    import embodik
    from robot_descriptions import panda_description
    import h5py

    task_idx, demo_idx, offsets = CONFIGS[config_name]
    paths = get_paths(task_idx)

    ox, oy, oz = offsets["x"], offsets["y"], offsets["z"]
    total = np.sqrt(ox**2 + oy**2 + oz**2)

    print("=" * 70)
    print(f"IK SOLVE: config={config_name}")
    print(f"  Task {task_idx}: {TASKS[task_idx]}")
    print(f"  Offset: X={ox*100:+.1f}cm Y={oy*100:+.1f}cm Z={oz*100:+.1f}cm (total={total*100:.1f}cm)")
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

    # Baseline trajectory (0cm offset)
    print("\n--- Solving IK for 0cm (baseline) ---")
    q_0cm, errors_0cm = solve_trajectory(demo_ee_pos, "0cm")

    # Offset trajectory
    print(f"\n--- Solving IK for offset X={ox*100:+.1f}cm Y={oy*100:+.1f}cm Z={oz*100:+.1f}cm ---")
    ee_offset = demo_ee_pos.copy()
    ee_offset[:, 0] += ox
    ee_offset[:, 1] += oy
    ee_offset[:, 2] += oz
    q_off, errors_off = solve_trajectory(ee_offset, "offset")

    # Gripper commands
    grip_qpos = demo_gripper[:, 0]
    gripper_cmd = np.where(grip_qpos > 0.015, -1.0, 1.0)

    out = npz_path(config_name)
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
        offset_x=ox,
        offset_y=oy,
        offset_z=oz,
    )
    print(f"\nSaved to {out}")

    n_fail = np.sum(np.isnan(errors_off))
    if n_fail > 0:
        print(f"WARNING: {n_fail} IK failures in offset trajectory - may not succeed in sim")
    return n_fail == 0


def stage_sim(config_name, record=False, extract_frames=True):
    """Execute IK trajectory in LIBERO sim, optionally record video and extract comparison frames."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, "/home/andypark/Projects/repos/LIBERO")

    import cv2
    from libero.libero.envs.env_wrapper import OffScreenRenderEnv

    task_idx, demo_idx, offsets = CONFIGS[config_name]
    paths = get_paths(task_idx)
    ox, oy, oz = offsets["x"], offsets["y"], offsets["z"]

    data_path = npz_path(config_name)
    data = np.load(data_path)
    q_0cm = data["q_0cm"]
    q_offset = data["q_offset"]
    gripper = data["gripper"]
    init_state_orig = data["init_state"].copy()

    print("=" * 70)
    print(f"SIM EVAL: config={config_name}")
    print(f"  Task {task_idx}: {TASKS[task_idx]}")
    print(f"  Offset: X={ox*100:+.1f}cm Y={oy*100:+.1f}cm Z={oz*100:+.1f}cm")
    print("=" * 70)

    # Bowl position indices in init_state: qpos[10]=X, qpos[11]=Y, qpos[12]=Z
    BOWL_X_IDX = 10

    def run_one(q_traj, gripper_cmds, init_st, label="",
                output_max=0.5, kp=150, substeps=3):
        """Run a single trajectory, return result dict + collected frames."""
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

        print(f"\n--- {label} ---")

        agentview_frames = []
        success_at = None
        n_steps = len(q_traj)

        for t in range(n_steps):
            for sub in range(substeps):
                q_target = q_traj[t, :7]
                q_current = np.array(robot._joint_positions)
                delta = q_target - q_current
                action = np.clip(delta / output_max, -1.0, 1.0)
                grip = gripper_cmds[t]
                full_action = np.concatenate([action, [grip]])
                obs, reward, done, info = env.step(full_action)

            # Save agentview frame
            av = obs['agentview_image'].astype(np.uint8)
            agentview_frames.append(av.copy())

            success = env.check_success()
            if success and success_at is None:
                success_at = t

            if t % 20 == 0 or t == n_steps - 1:
                ee_pos = np.array(robot.controller.ee_pos)
                bowl = env.sim.data.get_body_xpos("akita_black_bowl_1_main")
                print(f"  t={t:3d}: EE=[{ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}] "
                      f"bowl=[{bowl[0]:.3f},{bowl[1]:.3f},{bowl[2]:.3f}] "
                      f"{'SUCCESS' if success else ''}")

        # Settling
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
        bowl_final = env.sim.data.get_body_xpos("akita_black_bowl_1_main").copy()
        plate_final = env.sim.data.get_body_xpos("plate_1_main").copy()
        bowl_plate_dist = np.linalg.norm(bowl_final[:2] - plate_final[:2]) * 100

        print(f"  RESULT: {'SUCCESS' if final_success else 'FAILED'}")
        print(f"  Bowl-plate 2D dist: {bowl_plate_dist:.1f}cm")

        env.close()
        return {
            "success": bool(final_success),
            "success_at": success_at,
            "bowl_final": bowl_final.tolist(),
            "plate_final": plate_final.tolist(),
            "bowl_plate_dist_cm": round(bowl_plate_dist, 2),
            "agentview_frames": agentview_frames,
        }

    # Run baseline (0cm offset, original init_state)
    print("\n=== BASELINE (0cm) ===")
    result_base = run_one(q_0cm, gripper, init_state_orig.copy(), "baseline 0cm")

    # Run offset
    init_state_offset = init_state_orig.copy()
    init_state_offset[BOWL_X_IDX + 0] += ox
    init_state_offset[BOWL_X_IDX + 1] += oy
    init_state_offset[BOWL_X_IDX + 2] += oz
    print(f"\n=== OFFSET X={ox*100:+.1f}cm Y={oy*100:+.1f}cm Z={oz*100:+.1f}cm ===")
    result_off = run_one(q_offset, gripper, init_state_offset, f"offset {config_name}")

    # If offset failed, retry with higher gains
    if not result_off["success"]:
        print("\n--- Retrying offset with higher gains ---")
        init_state_offset2 = init_state_orig.copy()
        init_state_offset2[BOWL_X_IDX + 0] += ox
        init_state_offset2[BOWL_X_IDX + 1] += oy
        init_state_offset2[BOWL_X_IDX + 2] += oz
        result_off2 = run_one(q_offset, gripper, init_state_offset2,
                              f"offset {config_name} (high gains)",
                              output_max=1.0, kp=250, substeps=5)
        if result_off2["success"]:
            result_off = result_off2

    # Extract comparison frames at key timesteps
    if extract_frames and result_base["success"] and result_off["success"]:
        import cv2
        frames_base = result_base["agentview_frames"]
        frames_off = result_off["agentview_frames"]

        # Key frames: 0 (start), 20 (approach), 40 (mid-pick), 60 (lift), -1 (end)
        key_indices = [0, 20, 40, 60, min(len(frames_base)-1, len(frames_off)-1)]
        out_dir = f"/tmp/offset_visual_{config_name}"
        os.makedirs(out_dir, exist_ok=True)

        for idx in key_indices:
            if idx < len(frames_base) and idx < len(frames_off):
                fb = cv2.cvtColor(cv2.flip(frames_base[idx], 0), cv2.COLOR_RGB2BGR)
                fo = cv2.cvtColor(cv2.flip(frames_off[idx], 0), cv2.COLOR_RGB2BGR)
                # Side by side
                div = np.full((256, 4, 3), 200, dtype=np.uint8)
                combined = np.concatenate([fb, div, fo], axis=1)
                cv2.putText(combined, "Baseline", (5, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(combined, f"Offset", (261, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(combined, f"t={idx}", (240, 250),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                path = os.path.join(out_dir, f"compare_t{idx:03d}.png")
                cv2.imwrite(path, combined)
                print(f"Saved comparison frame: {path}")

        # Compute pixel displacement of bowl region at frame 0
        # Use simple difference image to estimate displacement
        f0_base = frames_base[0].astype(float)
        f0_off = frames_off[0].astype(float)
        diff = np.abs(f0_base - f0_off).mean(axis=2)  # avg across channels
        # Find centroid of diff
        total_diff = diff.sum()
        if total_diff > 0:
            yy, xx = np.mgrid[:256, :256]
            cx = (xx * diff).sum() / total_diff
            cy = (yy * diff).sum() / total_diff
            max_diff = diff.max()
            mean_diff = diff.mean()
            print(f"\nPixel difference at t=0: mean={mean_diff:.1f}, max={max_diff:.1f}, centroid=({cx:.0f},{cy:.0f})")
            # Count pixels with significant difference (>20)
            n_significant = (diff > 20).sum()
            print(f"Pixels with diff>20: {n_significant} ({n_significant/256/256*100:.1f}%)")
        else:
            print("No pixel difference detected!")

    # Summary
    print("\n" + "=" * 70)
    base_ok = "SUCCESS" if result_base["success"] else "FAILED"
    off_ok = "SUCCESS" if result_off["success"] else "FAILED"
    print(f"Config: {config_name}")
    print(f"  Baseline: {base_ok}")
    print(f"  Offset:   {off_ok}")
    if result_base["success"] and result_off["success"]:
        print(f"  BOTH SUCCEEDED - good candidate for demo video!")
    print("=" * 70)

    # Log result
    log_entry = {
        "config": config_name,
        "task_idx": task_idx,
        "offset_x_cm": round(ox*100, 1),
        "offset_y_cm": round(oy*100, 1),
        "offset_z_cm": round(oz*100, 1),
        "total_offset_cm": round(np.sqrt(ox**2 + oy**2 + oz**2)*100, 1),
        "baseline_success": result_base["success"],
        "offset_success": result_off["success"],
        "both_success": result_base["success"] and result_off["success"],
        "baseline_bowl_plate_cm": result_base["bowl_plate_dist_cm"],
        "offset_bowl_plate_cm": result_off["bowl_plate_dist_cm"],
    }
    log_file = "/tmp/multi_offset_log.json"
    if os.path.exists(log_file):
        with open(log_file) as f:
            log = json.load(f)
    else:
        log = []
    log.append(log_entry)
    with open(log_file, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Log: {log_file}")

    return result_base["success"] and result_off["success"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["ik", "sim", "list"], required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--all", action="store_true", help="Run all configs")
    args = parser.parse_args()

    if args.stage == "list":
        print("Available configs:")
        for name, (task_idx, demo_idx, offsets) in sorted(CONFIGS.items()):
            ox, oy, oz = offsets["x"], offsets["y"], offsets["z"]
            total = np.sqrt(ox**2 + oy**2 + oz**2)
            print(f"  {name:25s} task={task_idx} X={ox*100:+.1f}cm Y={oy*100:+.1f}cm Z={oz*100:+.1f}cm (total={total*100:.1f}cm)")
        sys.exit(0)

    if args.all:
        configs = sorted(CONFIGS.keys())
    elif args.config:
        configs = [args.config]
    else:
        print("Specify --config <name> or --all")
        sys.exit(1)

    for config_name in configs:
        if config_name not in CONFIGS:
            print(f"Unknown config: {config_name}")
            continue
        if args.stage == "ik":
            ok = stage_ik(config_name)
            print(f"  -> IK {'OK' if ok else 'FAILED'}")
        else:
            ok = stage_sim(config_name)
            print(f"  -> SIM {'BOTH OK' if ok else 'incomplete'}")
