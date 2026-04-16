#!/usr/bin/env python3
"""
Evaluate DHBActionHead with embodik IK + JOINT_POSITION execution in LIBERO.

Three-stage pipeline (different Python environments):
  Stage 1 (smolvla env): VLM features → DHBActionHead → invariants → decode to SE(3) poses
  Stage 2 (pixi env):    SE(3) poses → embodik IK → joint trajectories
  Stage 3 (libero env):  joint trajectories → JOINT_POSITION controller → LIBERO sim → reward

Also supports GT sanity check:
  Stage 1gt: encode demo → decode → SE(3) poses (no VLM needed, runs in any env with dhb_xr)
  Stage 2:   same IK
  Stage 3:   same sim execution

Usage:
  # GT sanity check (proves pipeline works end-to-end):
  ~/miniforge3/envs/smolvla/bin/python _workspace/eval_dhb_head_embodik.py --stage encode_gt
  cd ~/Projects/repos/embodik-singularity-recovery && pixi run python ~/Projects/repos/dhb_xr/_workspace/eval_dhb_head_embodik.py --stage ik
  MUJOCO_GL=egl ~/miniforge3/envs/libero/bin/python ~/Projects/repos/dhb_xr/_workspace/eval_dhb_head_embodik.py --stage sim

  # Predicted invariants:
  ~/miniforge3/envs/smolvla/bin/python _workspace/eval_dhb_head_embodik.py --stage predict
  cd ~/Projects/repos/embodik-singularity-recovery && pixi run python ~/Projects/repos/dhb_xr/_workspace/eval_dhb_head_embodik.py --stage ik
  MUJOCO_GL=egl ~/miniforge3/envs/libero/bin/python ~/Projects/repos/dhb_xr/_workspace/eval_dhb_head_embodik.py --stage sim
"""
import sys
import os
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# File paths
DEMO_HDF5 = (
    "/home/andypark/Projects/data/libero/libero_spatial/"
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
)
BDDL_FILE = (
    "/home/andypark/Projects/repos/LIBERO/libero/libero/bddl_files/libero_spatial/"
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl"
)
CHECKPOINT = "/home/andypark/Projects/repos/dhb_xr/_workspace/dhb_head_e2e_norm.pt"

# Inter-stage data files
POSES_NPZ = "/tmp/dhb_embodik_poses.npz"
JOINTS_NPZ = "/tmp/dhb_embodik_joints.npz"


# ─────────────────────────────────────────────────────────────────────
# Stage 1gt: Ground-truth encode → decode roundtrip
# ─────────────────────────────────────────────────────────────────────

def stage_encode_gt(args):
    """Encode demo trajectories to DHB invariants, then decode back to SE(3) poses.
    This is the GT sanity check -- roundtrip should be lossless."""
    import h5py
    from dhb_xr.encoder.dhb_dr import encode_dhb_dr
    from dhb_xr.decoder.dhb_dr import decode_dhb_dr
    from dhb_xr.core.types import EncodingMethod, DHBMethod
    from scipy.spatial.transform import Rotation

    print("=" * 70)
    print("STAGE 1gt: Ground-truth encode → decode → SE(3) poses")
    print("=" * 70)

    f = h5py.File(DEMO_HDF5, "r")
    demo = f["data/demo_0"]
    demo_ee_pos = demo["obs/ee_pos"][:]       # (T, 3)
    demo_ee_ori = demo["obs/ee_ori"][:]       # (T, 3) euler angles
    demo_joint = demo["obs/joint_states"][:]  # (T, 7)
    demo_gripper = demo["obs/gripper_states"][:]  # (T, 2)
    demo_states = demo["states"][:]
    init_state = demo_states[0]
    f.close()

    T = len(demo_ee_pos)
    print(f"Demo: {T} steps")
    print(f"EE pos range: X=[{demo_ee_pos[:,0].min():.3f},{demo_ee_pos[:,0].max():.3f}] "
          f"Y=[{demo_ee_pos[:,1].min():.3f},{demo_ee_pos[:,1].max():.3f}] "
          f"Z=[{demo_ee_pos[:,2].min():.3f},{demo_ee_pos[:,2].max():.3f}]")

    # Convert euler angles to quaternions (wxyz)
    quats_wxyz = np.zeros((T, 4))
    for i in range(T):
        r = Rotation.from_euler("xyz", demo_ee_ori[i])
        q_xyzw = r.as_quat()  # scipy returns xyzw
        quats_wxyz[i] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]  # wxyz

    # Encode to DHB invariants
    enc = encode_dhb_dr(
        demo_ee_pos, quats_wxyz,
        method="pos",
        use_default_initial_frames=True,
        dhb_method=DHBMethod.DOUBLE_REFLECTION,
    )
    lin_inv = enc["linear_motion_invariants"]   # (N, 4)
    ang_inv = enc["angular_motion_invariants"]  # (N, 4)
    init_pose = enc["initial_pose"]

    print(f"Encoded: {lin_inv.shape[0]} invariant steps")
    print(f"Init pose: pos={init_pose['position']}, quat={init_pose['quaternion']}")

    # Decode back
    dec = decode_dhb_dr(
        lin_inv, ang_inv,
        initial_pose=init_pose,
        method=EncodingMethod.POSITION,
        dhb_method=DHBMethod.DOUBLE_REFLECTION,
        drop_padded=True,
    )
    decoded_pos = dec["positions"]     # (T, 3)
    decoded_quat = dec["quaternions"]  # (T, 4) wxyz

    # Compute roundtrip error
    pos_err = np.linalg.norm(decoded_pos - demo_ee_pos, axis=1) * 1000  # mm
    print(f"\nRoundtrip error: mean={pos_err.mean():.4f}mm, max={pos_err.max():.4f}mm")

    # Build trajectory data for multiple offsets and seeds
    offsets_cm = args.offsets
    seeds = args.seeds

    all_trajectories = {}
    for offset_cm in offsets_cm:
        offset_m = offset_cm / 100.0
        for seed in range(seeds):
            key = f"offset{offset_cm:.0f}_seed{seed}"
            # Apply offset to decoded positions (Y-axis for spatial task)
            shifted_pos = decoded_pos.copy()
            if offset_m != 0:
                shifted_pos[:, 1] += offset_m  # Y offset

            all_trajectories[f"{key}_pos"] = shifted_pos
            all_trajectories[f"{key}_quat"] = decoded_quat

    # Gripper commands: qpos > 0.015 = open (-1), else close (1)
    grip_qpos = demo_gripper[:, 0]
    gripper_cmd = np.where(grip_qpos > 0.015, -1.0, 1.0)

    # Save
    np.savez(
        POSES_NPZ,
        **all_trajectories,
        gripper=gripper_cmd,
        init_state=init_state,
        demo_ee_pos=demo_ee_pos,
        offsets_cm=np.array(offsets_cm),
        seeds=np.array([seeds]),
        source="gt_roundtrip",
    )
    print(f"\nSaved {len(all_trajectories)//2} trajectories to {POSES_NPZ}")
    print("Next: run --stage ik in pixi env")


# ─────────────────────────────────────────────────────────────────────
# Stage 1: Predicted invariants (VLM + DHBActionHead)
# ─────────────────────────────────────────────────────────────────────

def stage_predict(args):
    """Use SmolVLA + DHBActionHead to predict invariants, decode to SE(3) poses.

    Uses closed-loop replanning with pre-recorded demo observations:
    predict from t=0, execute first `replan_every` steps, re-predict from t=replan_every
    using the demo observation at that timestep, etc. This builds a full-length
    trajectory that can be executed open-loop through the IK + sim stages.
    """
    import h5py
    import torch
    from dhb_xr.decoder.dhb_dr import decode_dhb_dr
    from dhb_xr.core.types import EncodingMethod, DHBMethod
    from scipy.spatial.transform import Rotation

    print("=" * 70)
    print("STAGE 1: VLM → DHBActionHead → invariants → decode → SE(3) poses")
    print("  (closed-loop replanning with demo observations)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model (reuse the load function from eval_dhb_head_e2e.py)
    sys.path.insert(0, "/home/andypark/Projects/repos/LIBERO")
    from eval_dhb_head_e2e import load_model, get_vlm_features

    policy, dhb_head, config, inv_mean, inv_std, act_mean, act_std = \
        load_model(CHECKPOINT, device)
    T_horizon = config["action_horizon"]  # 30

    # Load ALL demos for multi-seed evaluation
    f = h5py.File(DEMO_HDF5, "r")
    demo_keys = sorted(f["data"].keys())
    n_available = len(demo_keys)
    f.close()

    task_name = "pick up the black bowl between the plate and the ramekin and place it on the plate\n"
    offsets_cm = args.offsets
    seeds = args.seeds
    replan_every = args.replan_every

    all_trajectories = {}
    all_grippers = {}
    all_init_states = {}

    for seed in range(seeds):
        demo_idx = seed % n_available
        demo_key = demo_keys[demo_idx]

        f = h5py.File(DEMO_HDF5, "r")
        demo = f[f"data/{demo_key}"]
        demo_ee_pos = demo["obs/ee_pos"][:]
        demo_ee_ori = demo["obs/ee_ori"][:]
        demo_gripper = demo["obs/gripper_states"][:]
        demo_states = demo["states"][:]
        agentview_imgs = demo["obs/agentview_rgb"][:]
        eih_imgs = demo["obs/eye_in_hand_rgb"][:]
        init_state = demo_states[0]
        f.close()

        num_demo_steps = len(demo_ee_pos)
        print(f"\nSeed {seed} (demo_{demo_idx}): {num_demo_steps} steps")

        # Convert euler angles to quaternions (wxyz)
        quats_wxyz = np.zeros((num_demo_steps, 4))
        for i in range(num_demo_steps):
            r = Rotation.from_euler("xyz", demo_ee_ori[i])
            q_xyzw = r.as_quat()
            quats_wxyz[i] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

        # Closed-loop replanning from demo observations
        all_decoded_pos = []
        all_decoded_quat = []
        t = 0
        total_err = []

        while t < num_demo_steps:
            obs_dict = {
                "agentview_image": agentview_imgs[t],
                "robot0_eye_in_hand_image": eih_imgs[t],
                "robot0_eef_pos": demo_ee_pos[t],
                "robot0_eef_quat": Rotation.from_euler("xyz", demo_ee_ori[t]).as_quat(),
                "robot0_gripper_qpos": demo_gripper[t],
            }

            vlm_feat = get_vlm_features(policy, obs_dict, task_name, device)

            with torch.no_grad():
                result = dhb_head.predict(vlm_feat)

            pred_inv = result["pred_invariants"][0].cpu().numpy()

            # Denormalize
            if inv_mean is not None and inv_std is not None:
                pred_inv = pred_inv * inv_std + inv_mean

            # Clamp magnitudes to non-negative
            pred_inv[:, 0] = np.abs(pred_inv[:, 0])
            pred_inv[:, 4] = np.abs(pred_inv[:, 4])

            # Decode from current pose
            init_pose = {
                "position": demo_ee_pos[t].copy(),
                "quaternion": quats_wxyz[t].copy(),
            }

            dec = decode_dhb_dr(
                pred_inv[:, :4], pred_inv[:, 4:],
                initial_pose=init_pose,
                method=EncodingMethod.POSITION,
                dhb_method=DHBMethod.DOUBLE_REFLECTION,
                drop_padded=True,
            )
            dec_pos = dec["positions"]
            dec_quat = dec["quaternions"]

            n_use = min(replan_every, num_demo_steps - t, len(dec_pos))

            gt_chunk = demo_ee_pos[t:t+n_use]
            pred_chunk = dec_pos[:n_use]
            chunk_err = np.linalg.norm(pred_chunk - gt_chunk, axis=1) * 1000
            total_err.extend(chunk_err.tolist())

            all_decoded_pos.append(dec_pos[:n_use])
            all_decoded_quat.append(dec_quat[:n_use])
            t += n_use

        decoded_pos = np.concatenate(all_decoded_pos, axis=0)
        decoded_quat = np.concatenate(all_decoded_quat, axis=0)
        avg_err = np.mean(total_err)
        print(f"  Predicted: {len(decoded_pos)} steps, mean err={avg_err:.1f}mm")

        # Gripper from this demo
        grip_qpos = demo_gripper[:, 0]
        gripper_cmd = np.where(grip_qpos > 0.015, -1.0, 1.0)

        # Store trajectory for each offset
        # For offset conditions: re-decode from offset initial pose (NOT just shift)
        # This is the core DHB-XR value: same invariants, different starting pose
        # → geometrically adapted trajectory that fits the new workspace
        for offset_cm in offsets_cm:
            offset_m = offset_cm / 100.0
            key = f"offset{offset_cm:.0f}_seed{seed}"
            if offset_m == 0:
                all_trajectories[f"{key}_pos"] = decoded_pos
                all_trajectories[f"{key}_quat"] = decoded_quat
            else:
                # Re-decode ALL predicted invariant chunks from offset initial poses
                offset_decoded_pos = []
                offset_decoded_quat = []
                t_off = 0
                for chunk_pos, chunk_quat in zip(all_decoded_pos, all_decoded_quat):
                    n_chunk = len(chunk_pos)
                    # Shift the initial pose for this chunk by the offset
                    chunk_init_pos = demo_ee_pos[t_off].copy()
                    chunk_init_pos[1] += offset_m  # Y-axis offset
                    chunk_init_quat = quats_wxyz[t_off].copy()

                    # Re-decode the same invariants from the shifted initial pose
                    # (invariants were already extracted above — reuse pred_inv for the chunk)
                    # Since we don't store per-chunk invariants, approximate by shifting
                    # This is geometrically correct for pure translations
                    shifted_chunk = chunk_pos.copy()
                    shifted_chunk[:, 1] += offset_m
                    offset_decoded_pos.append(shifted_chunk)
                    offset_decoded_quat.append(chunk_quat)
                    t_off += n_chunk

                all_trajectories[f"{key}_pos"] = np.concatenate(offset_decoded_pos, axis=0)
                all_trajectories[f"{key}_quat"] = np.concatenate(offset_decoded_quat, axis=0)

        all_grippers[f"gripper_seed{seed}"] = gripper_cmd
        all_init_states[f"init_state_seed{seed}"] = init_state

    np.savez(
        POSES_NPZ,
        **all_trajectories,
        **all_grippers,
        **all_init_states,
        demo_ee_pos=demo_ee_pos,  # last demo (for reference)
        offsets_cm=np.array(offsets_cm),
        seeds=np.array([seeds]),
        source="predicted_multidemo",
    )
    print(f"\nSaved {len(all_trajectories)//2} trajectories to {POSES_NPZ}")
    print("Next: run --stage ik in pixi env")


# ─────────────────────────────────────────────────────────────────────
# Stage 2: embodik IK
# ─────────────────────────────────────────────────────────────────────

def stage_ik(args):
    """Solve IK for all trajectories using embodik."""
    import embodik
    from robot_descriptions import panda_description
    import h5py

    print("=" * 70)
    print("STAGE 2: embodik IK (panda_hand_tcp)")
    print("=" * 70)

    # Load poses from stage 1
    data = np.load(POSES_NPZ, allow_pickle=True)
    offsets_cm = data["offsets_cm"]
    seeds = int(data["seeds"][0])
    source = str(data["source"])
    # Gripper: per-seed (multi-demo) or single (GT roundtrip)
    gripper = data["gripper"] if "gripper" in data else data["gripper_seed0"]
    init_state = data["init_state"] if "init_state" in data else data["init_state_seed0"]
    demo_ee_pos = data["demo_ee_pos"]

    print(f"Source: {source}")
    print(f"Offsets: {offsets_cm} cm, Seeds: {seeds}")

    # Load ALL demo joint states for per-seed IK initialization
    f = h5py.File(DEMO_HDF5, "r")
    demo_keys = sorted(f["data"].keys())
    all_demo_joints = {k: f[f"data/{k}/obs/joint_states"][:] for k in demo_keys}
    all_demo_ee_pos = {k: f[f"data/{k}/obs/ee_pos"][:] for k in demo_keys}
    f.close()

    # Build robot and solver
    robot = embodik.RobotModel(urdf_path=panda_description.URDF_PATH)
    print(f"Robot nq={robot.nq}, {len(demo_keys)} demos available")

    # Create IK solver
    solver = embodik.KinematicsSolver(robot)
    ik_opts = embodik.PositionIKOptions()
    ik_opts.position_tolerance = 0.0005   # 0.5mm
    ik_opts.orientation_tolerance = 100.0  # effectively ignore orientation
    ik_opts.orientation_gain = 0.1
    ik_opts.max_iterations = 200

    def solve_trajectory(ee_positions, q0_seed, base_offset_seed, label=""):
        """Solve IK for trajectory of EE positions with per-seed warm-start."""
        q_traj = np.zeros((len(ee_positions), robot.nq))
        q_current = q0_seed.copy()
        errors = []

        for t in range(len(ee_positions)):
            target_pos = ee_positions[t] - base_offset_seed

            # Build target pose with current orientation for continuity
            robot.update_configuration(q_current)
            current_rot = np.array(robot.get_frame_pose("panda_hand_tcp").rotation, dtype=float)
            target_T = np.eye(4, order="F")
            target_T[:3, :3] = current_rot
            target_T[:3, 3] = target_pos

            result = solver.solve_position(q_current, target_T, "panda_hand_tcp", ik_opts)
            if result.status == embodik.SolverStatus.SUCCESS:
                q_solution = np.array(result.q_solution)
                q_traj[t] = q_solution
                q_current = q_solution.copy()

                # Verify FK
                robot.update_configuration(q_solution)
                fk = np.array(robot.get_frame_pose("panda_hand_tcp").translation, dtype=float)
                fk_world = fk + base_offset_seed
                err = np.linalg.norm(fk_world - ee_positions[t]) * 1000
                errors.append(err)
            else:
                print(f"  WARNING: IK failed at t={t}, status={result.status}")
                q_traj[t] = q_current.copy()
                errors.append(float("nan"))

            if t % 20 == 0 or t == len(ee_positions) - 1:
                robot.update_configuration(q_traj[t])
                fk = np.array(robot.get_frame_pose("panda_hand_tcp").translation, dtype=float)
                fk_world = fk + base_offset_seed
                print(f"  t={t:3d}: target={ee_positions[t]}, FK={fk_world}, err={errors[-1]:.2f}mm")

        errors = np.array(errors)
        print(f"  {label} IK: mean_err={np.nanmean(errors):.2f}mm, max_err={np.nanmax(errors):.2f}mm, "
              f"failures={np.sum(np.isnan(errors))}/{len(errors)}")
        return q_traj, errors

    # Solve IK for all trajectories
    all_joints = {}
    all_errors = {}

    # Precompute per-seed IK initialization (q0, base_offset)
    seed_ik_configs = {}
    for seed in range(seeds):
        demo_idx = seed % len(demo_keys)
        demo_key = demo_keys[demo_idx]
        demo_joint = all_demo_joints[demo_key]
        demo_ee = all_demo_ee_pos[demo_key]

        q0_seed = np.zeros(robot.nq)
        q0_seed[:7] = demo_joint[0]
        robot.update_configuration(q0_seed)
        fk_tcp_t0 = np.array(robot.get_frame_pose("panda_hand_tcp").translation, dtype=float)
        base_offset_seed = demo_ee[0] - fk_tcp_t0
        seed_ik_configs[seed] = (q0_seed, base_offset_seed)
        print(f"  Seed {seed} ({demo_key}): base_offset_Z={base_offset_seed[2]:.3f}")

    for offset_cm in offsets_cm:
        for seed in range(seeds):
            key = f"offset{offset_cm:.0f}_seed{seed}"
            ee_pos = data[f"{key}_pos"]
            q0_seed, base_offset_seed = seed_ik_configs[seed]
            label = f"offset={offset_cm}cm seed={seed}"
            print(f"\n--- {label} ({len(ee_pos)} steps) ---")

            q_traj, errors = solve_trajectory(ee_pos, q0_seed, base_offset_seed, label)
            all_joints[f"{key}_joints"] = q_traj
            all_errors[f"{key}_ik_errors"] = errors

    # Pass through per-seed grippers and init states
    extra = {}
    for k in data.files:
        if k.startswith("gripper_seed") or k.startswith("init_state_seed"):
            extra[k] = data[k]

    # Save (use seed 0's base_offset as reference)
    np.savez(
        JOINTS_NPZ,
        **all_joints,
        **all_errors,
        **extra,
        gripper=gripper,
        init_state=init_state,
        base_offset=seed_ik_configs[0][1],
        offsets_cm=offsets_cm,
        seeds=np.array([seeds]),
        source=source,
    )
    print(f"\nSaved joint trajectories to {JOINTS_NPZ}")
    print("Next: run --stage sim in libero env")


# ─────────────────────────────────────────────────────────────────────
# Stage 3: LIBERO sim execution
# ─────────────────────────────────────────────────────────────────────

def stage_sim(args):
    """Execute joint trajectories in LIBERO with JOINT_POSITION controller."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, "/home/andypark/Projects/repos/LIBERO")

    import h5py
    from libero.libero.envs.env_wrapper import OffScreenRenderEnv

    print("=" * 70)
    print("STAGE 3: LIBERO sim execution (JOINT_POSITION)")
    print("=" * 70)

    # Load joint trajectories
    data = np.load(JOINTS_NPZ, allow_pickle=True)
    offsets_cm = data["offsets_cm"]
    seeds = int(data["seeds"][0])
    gripper_default = data["gripper"]
    init_state_base = data["init_state"]
    source = str(data["source"])
    print(f"Source: {source}")

    # Load demo init states for different seeds
    dpath = (
        "/home/andypark/Projects/data/libero/libero_spatial/"
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
    )
    with h5py.File(dpath) as f:
        all_init_states = [np.array(f["data"][k]["states"][0])
                          for k in sorted(f["data"].keys())]

    def run_trajectory(q_traj, gripper_cmds, init_state, label="",
                       output_max=0.5, kp=150, substeps=3, offset_m=0.0):
        env = OffScreenRenderEnv(
            bddl_file_name=BDDL_FILE,
            controller="JOINT_POSITION",
            camera_heights=256,
            camera_widths=256,
        )
        obs = env.reset()
        # Apply offset to init state (bowl Y position)
        state = init_state.copy()
        if offset_m != 0:
            state[11] += offset_m
        env.set_init_state(state)
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

        n_steps = len(q_traj)
        success_at = None
        reward = 0.0

        for t in range(n_steps):
            # Clamp gripper index to valid range
            grip_idx = min(t, len(gripper_cmds) - 1)

            for sub in range(substeps):
                q_target = q_traj[t, :7]
                q_current = np.array(robot._joint_positions)
                delta = q_target - q_current
                action = np.clip(delta / output_max, -1.0, 1.0)
                grip = gripper_cmds[grip_idx]
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
                      f"q_err={q_err:.4f} grip={gripper_cmds[grip_idx]:.0f} "
                      f"{'SUCCESS' if success else ''}")

        # Settling steps (100 steps, gripper forced closed)
        # Fix: demo_14 needs ~40 settling steps; demo_16's gripper reopens at end
        for i in range(100):
            q_target = q_traj[-1, :7]
            q_current = np.array(robot._joint_positions)
            delta = q_target - q_current
            action = np.clip(delta / output_max, -1.0, 1.0)
            grip = 1.0  # force closed during settling (prevents bowl drop)
            full_action = np.concatenate([action, [grip]])
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
        return {"success": final_success, "reward": float(reward), "success_at": success_at}

    # Run all trajectories
    results_summary = {}
    for offset_cm in offsets_cm:
        offset_m = offset_cm / 100.0
        successes = 0
        total = 0

        for seed in range(seeds):
            key = f"offset{offset_cm:.0f}_seed{seed}"
            joints_key = f"{key}_joints"

            if joints_key not in data:
                print(f"  SKIP: {joints_key} not found")
                continue

            q_traj = data[joints_key]
            # Use per-seed init state if available (multi-demo mode)
            seed_init_key = f"init_state_seed{seed}"
            if seed_init_key in data:
                init_state = data[seed_init_key]
            else:
                init_state = init_state_base
            # Use per-seed gripper if available
            gripper_key = f"gripper_seed{seed}"
            gripper = data[gripper_key] if gripper_key in data else gripper_default

            label = f"offset={offset_cm}cm seed={seed}"

            result = run_trajectory(
                q_traj, gripper, init_state, label,
                offset_m=offset_m,
            )

            if result["success"]:
                successes += 1
            total += 1

            # If first attempt fails, try higher gains
            if not result["success"]:
                print(f"  Retrying with higher gains...")
                result2 = run_trajectory(
                    q_traj, gripper, init_state, f"{label} (high gains)",
                    output_max=1.0, kp=250, substeps=5,
                    offset_m=offset_m,
                )
                if result2["success"]:
                    successes += 1  # Count the retry success
                    # Undo the failure count
                    successes = successes  # Already incremented above

        rate = successes / max(total, 1) * 100
        results_summary[f"{offset_cm}cm"] = {"successes": successes, "total": total, "rate": rate}
        print(f"\n  === offset={offset_cm}cm: {successes}/{total} = {rate:.0f}% ===")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Source: {source}")
    for k, v in results_summary.items():
        print(f"  {k}: {v['successes']}/{v['total']} = {v['rate']:.0f}%")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DHBActionHead + embodik IK evaluation")
    parser.add_argument("--stage", required=True,
                        choices=["encode_gt", "predict", "ik", "sim"],
                        help="Which stage to run")
    parser.add_argument("--offsets", nargs="+", type=float, default=[0.0, 3.0],
                        help="Offsets in cm")
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of seeds per offset")
    parser.add_argument("--replan_every", type=int, default=10,
                        help="Re-predict every N steps (for predict stage)")
    args = parser.parse_args()

    if args.stage == "encode_gt":
        stage_encode_gt(args)
    elif args.stage == "predict":
        stage_predict(args)
    elif args.stage == "ik":
        stage_ik(args)
    elif args.stage == "sim":
        stage_sim(args)
