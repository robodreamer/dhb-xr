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
import json
import re
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

LIBERO_GRIPPER_OPEN_QPOS_THRESHOLD = 0.015

LIBERO_IMPORT_CANDIDATES = [
    "/home/andypark/Projects/repos/LIBERO",
    "/home/andypark/Projects/repos/LIBERO-PRO",
]
LIBERO_BDDL_ROOT_CANDIDATES = [
    "/home/andypark/Projects/repos/LIBERO/libero/libero/bddl_files",
    "/home/andypark/Projects/repos/LIBERO-PRO/libero/libero/bddl_files",
]
for _candidate in LIBERO_IMPORT_CANDIDATES:
    if Path(_candidate).exists():
        sys.path.insert(0, _candidate)
        break

# File paths
DEMO_HDF5 = (
    "/home/andypark/Projects/data/libero/libero_spatial/"
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
)
for _candidate in LIBERO_BDDL_ROOT_CANDIDATES:
    if Path(_candidate).exists():
        _bddl_root = _candidate
        break
else:
    _bddl_root = LIBERO_BDDL_ROOT_CANDIDATES[0]

BDDL_FILE = str(
    Path(_bddl_root)
    / "libero_spatial"
    / "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl"
)
DEFAULT_CHECKPOINT = str(Path(__file__).with_name("dhb_head_e2e_fix_smoke.pt"))

# Inter-stage data files
POSES_NPZ = "/tmp/dhb_embodik_poses.npz"
JOINTS_NPZ = "/tmp/dhb_embodik_joints.npz"


def poses_npz_path(args):
    return getattr(args, "poses_npz", POSES_NPZ)


def joints_npz_path(args):
    return getattr(args, "joints_npz", JOINTS_NPZ)


def demo_key_sort_key(key):
    match = re.fullmatch(r"demo_(\d+)", str(key))
    if match:
        return (0, int(match.group(1)))
    return (1, str(key))


def sorted_demo_keys(hdf5_group):
    return sorted(hdf5_group.keys(), key=demo_key_sort_key)


def demo_key_for_episode(demo_keys, episode_idx):
    candidate = f"demo_{int(episode_idx)}"
    if candidate in demo_keys:
        return candidate
    return demo_keys[int(episode_idx) % len(demo_keys)]


def build_stage_summary_path(args, stage_name):
    if not args.debug_json:
        return None

    base = Path(args.debug_json)
    if base.suffix:
        return str(base.with_name(f"{base.stem}_{stage_name}{base.suffix}"))
    return str(base / f"{stage_name}.json")


def write_debug_summary(path, payload):
    if not path:
        return

    def _to_jsonable(value):
        if isinstance(value, dict):
            return {k: _to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_to_jsonable(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(_to_jsonable(payload), f, indent=2)
    print(f"Debug summary: {out_path}")


def make_video_path(video_dir, label, suffix="mp4"):
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_")
    return str(Path(video_dir) / f"{safe}.{suffix}")


def quaternion_step_angles_deg(quaternions_wxyz):
    if len(quaternions_wxyz) < 2:
        return np.zeros((0,), dtype=np.float32)

    quats = np.asarray(quaternions_wxyz, dtype=np.float64)
    step_dots = np.sum(quats[:-1] * quats[1:], axis=1)
    step_dots = np.clip(np.abs(step_dots), -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(step_dots)).astype(np.float32)


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

    # Gripper commands: qpos > threshold = open (-1), else close (1)
    grip_qpos = demo_gripper[:, 0]
    gripper_cmd = np.where(grip_qpos > LIBERO_GRIPPER_OPEN_QPOS_THRESHOLD, -1.0, 1.0)

    # Save
    poses_path = poses_npz_path(args)
    np.savez(
        poses_path,
        **all_trajectories,
        gripper=gripper_cmd,
        init_state=init_state,
        demo_ee_pos=demo_ee_pos,
        offsets_cm=np.array(offsets_cm),
        seeds=np.array([seeds]),
        source="gt_roundtrip",
    )
    print(f"\nSaved {len(all_trajectories)//2} trajectories to {poses_path}")
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
    from dhb_xr.integration.vla.smolvla_adapter import SmolVLADHBAdapter
    from scipy.spatial.transform import Rotation

    print("=" * 70)
    print("STAGE 1: VLM → DHBActionHead → invariants → decode → SE(3) poses")
    print("  (closed-loop replanning with demo observations)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model (reuse the load function from eval_dhb_head_e2e.py)
    sys.path.insert(0, "/home/andypark/Projects/repos/LIBERO")
    from eval_dhb_head_e2e import load_model, get_vlm_features

    if not args.checkpoint:
        raise ValueError("--checkpoint is required for --stage predict")

    policy, dhb_head, config, inv_mean, inv_std, act_mean, act_std = \
        load_model(args.checkpoint, device)
    adapter = SmolVLADHBAdapter.from_dhb_head(
        dhb_head, vlm_dim=config["vlm_dim"], action_horizon=config["action_horizon"]
    )
    T_horizon = config["action_horizon"]  # 30

    # Load ALL demos for multi-seed evaluation
    f = h5py.File(DEMO_HDF5, "r")
    demo_keys = sorted_demo_keys(f["data"])
    n_available = len(demo_keys)
    f.close()

    task_name = "pick up the black bowl between the plate and the ramekin and place it on the plate\n"
    offsets_cm = args.offsets
    seeds = args.seeds
    replan_every = args.replan_every

    all_trajectories = {}
    all_grippers = {}
    all_init_states = {}
    debug_rows = []

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
                result = {
                    "pred_invariants": adapter.predict_invariants_from_observation(vlm_feat, obs_dict)
                }

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
            rot_step_deg = quaternion_step_angles_deg(dec_quat[:n_use])
            total_err.extend(chunk_err.tolist())
            debug_rows.append({
                "seed": seed,
                "demo_key": demo_key,
                "t_start": int(t),
                "n_use": int(n_use),
                "chunk_mean_err_mm": float(np.mean(chunk_err)),
                "chunk_max_err_mm": float(np.max(chunk_err)),
                "decoded_y_min": float(np.min(dec_pos[:n_use, 1])),
                "decoded_y_max": float(np.max(dec_pos[:n_use, 1])),
                "invariant_norm": float(np.linalg.norm(pred_inv)),
                "current_ee_pos": demo_ee_pos[t].tolist(),
                "first_decoded_pos": dec_pos[0].tolist(),
                "mean_step_rot_deg": float(rot_step_deg.mean()) if len(rot_step_deg) else 0.0,
                "max_step_rot_deg": float(rot_step_deg.max()) if len(rot_step_deg) else 0.0,
                "total_rot_deg": float(rot_step_deg.sum()) if len(rot_step_deg) else 0.0,
            })

            all_decoded_pos.append(dec_pos[:n_use])
            all_decoded_quat.append(dec_quat[:n_use])
            t += n_use

        decoded_pos = np.concatenate(all_decoded_pos, axis=0)
        decoded_quat = np.concatenate(all_decoded_quat, axis=0)
        avg_err = np.mean(total_err)
        print(f"  Predicted: {len(decoded_pos)} steps, mean err={avg_err:.1f}mm")

        # Gripper from this demo
        grip_qpos = demo_gripper[:, 0]
        gripper_cmd = np.where(grip_qpos > LIBERO_GRIPPER_OPEN_QPOS_THRESHOLD, -1.0, 1.0)

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

    poses_path = poses_npz_path(args)
    np.savez(
        poses_path,
        **all_trajectories,
        **all_grippers,
        **all_init_states,
        demo_ee_pos=demo_ee_pos,  # last demo (for reference)
        offsets_cm=np.array(offsets_cm),
        seeds=np.array([seeds]),
        source="predicted_multidemo",
    )
    print(f"\nSaved {len(all_trajectories)//2} trajectories to {poses_path}")
    write_debug_summary(build_stage_summary_path(args, "predict"), {
        "stage": "predict",
        "checkpoint": args.checkpoint,
        "offsets_cm": list(map(float, offsets_cm)),
        "seeds": int(seeds),
        "replan_every": int(replan_every),
        "source": "predicted_multidemo",
        "chunks": debug_rows,
    })
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
    data = np.load(poses_npz_path(args), allow_pickle=True)
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
        primary_failures = 0
        fallback_repairs = 0
        fallback_alphas = []

        robot.update_configuration(q_current)
        prev_reachable_local = np.array(robot.get_frame_pose("panda_hand_tcp").translation, dtype=float)

        def solve_one(start_q, target_world):
            target_pos = target_world - base_offset_seed

            # Build target pose with current orientation for continuity.
            robot.update_configuration(start_q)
            current_rot = np.array(robot.get_frame_pose("panda_hand_tcp").rotation, dtype=float)
            target_T = np.eye(4, order="F")
            target_T[:3, :3] = current_rot
            target_T[:3, 3] = target_pos
            return solver.solve_position(start_q, target_T, "panda_hand_tcp", ik_opts)

        for t in range(len(ee_positions)):
            desired_world = ee_positions[t]
            result = solve_one(q_current, desired_world)
            if result.status == embodik.SolverStatus.SUCCESS:
                q_solution = np.array(result.q_solution)
                q_traj[t] = q_solution
                q_current = q_solution.copy()

                # Verify FK
                robot.update_configuration(q_solution)
                fk = np.array(robot.get_frame_pose("panda_hand_tcp").translation, dtype=float)
                prev_reachable_local = fk.copy()
                fk_world = fk + base_offset_seed
                err = np.linalg.norm(fk_world - ee_positions[t]) * 1000
                errors.append(err)
            else:
                primary_failures += 1
                repaired = False
                if args.ik_line_search_fallback:
                    prev_world = prev_reachable_local + base_offset_seed
                    for alpha in np.linspace(
                        1.0,
                        max(0.0, float(args.ik_line_search_min_alpha)),
                        max(2, int(args.ik_line_search_steps)),
                    )[1:]:
                        candidate_world = prev_world + float(alpha) * (desired_world - prev_world)
                        fallback_result = solve_one(q_current, candidate_world)
                        if fallback_result.status == embodik.SolverStatus.SUCCESS:
                            q_solution = np.array(fallback_result.q_solution)
                            q_traj[t] = q_solution
                            q_current = q_solution.copy()
                            robot.update_configuration(q_solution)
                            fk = np.array(robot.get_frame_pose("panda_hand_tcp").translation, dtype=float)
                            prev_reachable_local = fk.copy()
                            fk_world = fk + base_offset_seed
                            err = np.linalg.norm(fk_world - desired_world) * 1000
                            errors.append(err)
                            fallback_repairs += 1
                            fallback_alphas.append(float(alpha))
                            repaired = True
                            break
                if not repaired:
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
              f"failures={np.sum(np.isnan(errors))}/{len(errors)}, "
              f"primary_failures={primary_failures}, fallback_repairs={fallback_repairs}")
        return q_traj, errors, {
            "ik_primary_failures": int(primary_failures),
            "ik_fallback_repairs": int(fallback_repairs),
            "ik_fallback_alpha_mean": (
                float(np.mean(fallback_alphas)) if fallback_alphas else None
            ),
            "ik_fallback_alpha_min": (
                float(np.min(fallback_alphas)) if fallback_alphas else None
            ),
        }

    # Solve IK for all trajectories
    all_joints = {}
    all_errors = {}
    debug_rows = []

    # Precompute per-seed IK initialization (q0, base_offset)
    seed_ik_configs = {}
    for seed in range(seeds):
        episode_key = f"episode_idx_seed{seed}"
        if episode_key in data:
            episode_idx = int(np.asarray(data[episode_key]).reshape(-1)[0])
        else:
            episode_idx = seed
        demo_key = demo_key_for_episode(demo_keys, episode_idx)
        demo_joint = all_demo_joints[demo_key]
        demo_ee = all_demo_ee_pos[demo_key]

        q0_seed = np.zeros(robot.nq)
        q0_seed[:7] = demo_joint[0]
        robot.update_configuration(q0_seed)
        fk_tcp_t0 = np.array(robot.get_frame_pose("panda_hand_tcp").translation, dtype=float)
        base_offset_seed = demo_ee[0] - fk_tcp_t0
        seed_ik_configs[seed] = (q0_seed, base_offset_seed)
        print(
            f"  Seed {seed} ({demo_key}, episode_idx={episode_idx}): "
            f"base_offset_Z={base_offset_seed[2]:.3f}"
        )

    for offset_cm in offsets_cm:
        for seed in range(seeds):
            key = f"offset{offset_cm:.0f}_seed{seed}"
            ee_pos = data[f"{key}_pos"]
            q0_seed, base_offset_seed = seed_ik_configs[seed]
            label = f"offset={offset_cm}cm seed={seed}"
            print(f"\n--- {label} ({len(ee_pos)} steps) ---")

            q_traj, errors, repair_stats = solve_trajectory(ee_pos, q0_seed, base_offset_seed, label)
            all_joints[f"{key}_joints"] = q_traj
            all_errors[f"{key}_ik_errors"] = errors
            debug_rows.append({
                "offset_cm": float(offset_cm),
                "seed": seed,
                "steps": int(len(ee_pos)),
                "ik_mean_err_mm": float(np.nanmean(errors)),
                "ik_max_err_mm": float(np.nanmax(errors)),
                "ik_failures": int(np.sum(np.isnan(errors))),
                **repair_stats,
            })

    # Pass through per-seed grippers and init states
    extra = {}
    for k in data.files:
        if k.startswith("gripper_seed") or k.startswith("init_state_seed"):
            extra[k] = data[k]

    # Save (use seed 0's base_offset as reference)
    joints_path = joints_npz_path(args)
    np.savez(
        joints_path,
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
    print(f"\nSaved joint trajectories to {joints_path}")
    write_debug_summary(build_stage_summary_path(args, "ik"), {
        "stage": "ik",
        "offsets_cm": list(map(float, offsets_cm)),
        "seeds": int(seeds),
        "source": source,
        "runs": debug_rows,
    })
    print("Next: run --stage sim in libero env")


# ─────────────────────────────────────────────────────────────────────
# Stage 3: LIBERO sim execution
# ─────────────────────────────────────────────────────────────────────

def stage_sim(args):
    """Execute joint trajectories in LIBERO with JOINT_POSITION controller."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, "/home/andypark/Projects/repos/LIBERO")

    import h5py
    import imageio.v2 as imageio
    from libero.libero.envs.env_wrapper import OffScreenRenderEnv
    from dhb_xr.integration.vla.chunk_rollout import compute_motion_metrics

    print("=" * 70)
    print("STAGE 3: LIBERO sim execution (JOINT_POSITION)")
    print("=" * 70)

    # Load joint trajectories
    data = np.load(joints_npz_path(args), allow_pickle=True)
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
        demo_keys = sorted_demo_keys(f["data"])
        all_init_states = [np.array(f["data"][k]["states"][0])
                          for k in demo_keys]

    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    def find_named_position(sim, body_tokens=(), site_tokens=(), geom_tokens=()):
        """Best-effort lookup for an object's world position by fuzzy name match."""
        def _matches(name, tokens):
            name = name.lower()
            return all(tok in name for tok in tokens)

        model = sim.model
        data = sim.data

        for body_id in range(model.nbody):
            body_name = model.body_id2name(body_id)
            if body_name and _matches(body_name, body_tokens):
                return np.array(data.body_xpos[body_id]), f"body:{body_name}"

        for site_id in range(model.nsite):
            site_name = model.site_id2name(site_id)
            if site_name and _matches(site_name, site_tokens):
                return np.array(data.site_xpos[site_id]), f"site:{site_name}"

        for geom_id in range(model.ngeom):
            geom_name = model.geom_id2name(geom_id)
            if geom_name and _matches(geom_name, geom_tokens):
                return np.array(data.geom_xpos[geom_id]), f"geom:{geom_name}"

        return None, None

    def capture_scene_refs(env):
        sim = env.env.sim
        bowl_pos, bowl_ref = find_named_position(
            sim,
            body_tokens=("bowl",),
            site_tokens=("bowl",),
            geom_tokens=("bowl",),
        )
        plate_pos, plate_ref = find_named_position(
            sim,
            body_tokens=("plate",),
            site_tokens=("plate",),
            geom_tokens=("plate",),
        )
        return {
            "bowl_pos": bowl_pos,
            "bowl_ref": bowl_ref,
            "plate_pos": plate_pos,
            "plate_ref": plate_ref,
        }

    def relative_snapshot(ee_pos, refs):
        snap = {"ee_pos": np.array(ee_pos).tolist()}
        bowl_pos = refs.get("bowl_pos")
        plate_pos = refs.get("plate_pos")
        if bowl_pos is not None:
            snap["ee_minus_bowl"] = (np.array(ee_pos) - bowl_pos).tolist()
        if plate_pos is not None:
            snap["ee_minus_plate"] = (np.array(ee_pos) - plate_pos).tolist()
        return snap

    def run_trajectory(q_traj, gripper_cmds, init_state, label="",
                       output_max=0.5, kp=150, substeps=3, offset_m=0.0,
                       close_gate_xy_m=0.05, close_gate_z_m=0.05,
                       release_gate_xy_m=0.04, release_gate_z_m=0.06,
                       disable_gripper_gates=False,
                       reach_only=False, reach_threshold_xy_m=0.05,
                       reach_threshold_z_m=0.08, reach_stop_on_success=False,
                       placement_tail=False, tail_plate_xy_m=0.05,
                       tail_max_release_z_m=0.08, tail_move_steps=12,
                       tail_open_steps=12, tail_lift_steps=12):
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
        scene_refs = capture_scene_refs(env)

        robot = env.env.robots[0]
        ctrl = robot.controller

        # Override controller gains AFTER reset
        ctrl.output_max = ctrl.nums2array(output_max, len(ctrl.qpos_index))
        ctrl.output_min = ctrl.nums2array(-output_max, len(ctrl.qpos_index))
        ctrl.kp = ctrl.nums2array(kp, len(ctrl.qpos_index))
        ctrl.kd = 2 * np.sqrt(ctrl.kp) * 1.0
        ctrl.action_scale = None

        print(f"\n--- {label} (output_max={output_max}, kp={kp}, substeps={substeps}) ---")
        video_path = make_video_path(video_dir, label)
        writer = imageio.get_writer(video_path, fps=10)

        def append_frame(observation):
            frame = observation.get("agentview_image")
            if frame is None:
                return
            frame = np.flipud(np.asarray(frame, dtype=np.uint8))
            writer.append_data(frame)

        n_steps = len(q_traj)
        success_at = None
        reward = 0.0
        close_step = None
        reopen_step = None
        close_snapshot = None
        reopen_snapshot = None
        ee_trace = [np.array(robot.controller.ee_pos, dtype=np.float32).copy()]
        obs_gripper_qpos = obs.get("robot0_gripper_qpos")
        if obs_gripper_qpos is not None:
            obs_gripper_qpos = float(np.asarray(obs_gripper_qpos).reshape(-1)[0])
            last_effective_grip = (
                -1.0 if obs_gripper_qpos > LIBERO_GRIPPER_OPEN_QPOS_THRESHOLD else 1.0
            )
        else:
            last_effective_grip = -1.0
        executed_trace = []
        append_frame(obs)

        def reach_ready(ee_pos_xyz):
            rel = relative_snapshot(ee_pos_xyz, scene_refs)
            if "ee_minus_bowl" not in rel:
                return False
            ee_minus_bowl = np.asarray(rel["ee_minus_bowl"], dtype=np.float32)
            return (
                float(np.linalg.norm(ee_minus_bowl[:2])) <= reach_threshold_xy_m
                and float(ee_minus_bowl[2]) <= reach_threshold_z_m
            )

        def step_joint_target(q_target, grip_cmd, repeats):
            nonlocal reward
            for _ in range(repeats):
                q_current = np.array(robot._joint_positions)
                delta = q_target - q_current
                action = np.clip(delta / output_max, -1.0, 1.0)
                full_action = np.concatenate([action, [grip_cmd]])
                step_obs, r, done, info = env.step(full_action)
                if r > reward:
                    reward = r
                ee_trace.append(np.array(robot.controller.ee_pos, dtype=np.float32).copy())
                append_frame(step_obs)
            return step_obs

        for t in range(n_steps):
            # Clamp gripper index to valid range
            grip_idx = min(t, len(gripper_cmds) - 1)
            desired_grip = -1.0 if reach_only else gripper_cmds[grip_idx]
            ee_pos_pre = np.array(robot.controller.ee_pos)
            effective_grip = desired_grip

            rel_pre = relative_snapshot(ee_pos_pre, scene_refs)
            ee_minus_bowl = np.array(rel_pre.get("ee_minus_bowl", [np.inf, np.inf, np.inf]))
            ee_minus_plate = np.array(rel_pre.get("ee_minus_plate", [np.inf, np.inf, np.inf]))
            close_ready = (
                abs(float(ee_minus_bowl[0])) <= close_gate_xy_m
                and abs(float(ee_minus_bowl[1])) <= close_gate_xy_m
                and float(ee_minus_bowl[2]) <= close_gate_z_m
            )
            release_ready = (
                abs(float(ee_minus_plate[0])) <= release_gate_xy_m
                and abs(float(ee_minus_plate[1])) <= release_gate_xy_m
                and float(ee_minus_plate[2]) <= release_gate_z_m
            )

            # Geometry vetoes are useful for replayed demo timing, but they
            # should be disable-able for learned gripper policies whose labels
            # already come from offline geometry-conditioned supervision.
            if not disable_gripper_gates:
                if desired_grip > 0 and last_effective_grip <= 0:
                    if not close_ready:
                        effective_grip = -1.0
                if desired_grip <= 0 and last_effective_grip > 0:
                    if not release_ready:
                        effective_grip = 1.0

            q_target = q_traj[t, :7]
            obs = step_joint_target(q_target, effective_grip, substeps)

            prev_grip = last_effective_grip
            cur_grip = effective_grip
            last_effective_grip = effective_grip
            ee_pos = np.array(robot.controller.ee_pos)
            rel_post = relative_snapshot(ee_pos, scene_refs)
            executed_trace.append(
                {
                    "t": int(t),
                    "q_target": q_target.copy(),
                    "ee_pos": ee_pos.copy(),
                    "ee_minus_plate": np.array(rel_post.get("ee_minus_plate", [np.inf, np.inf, np.inf])),
                    "effective_grip": float(effective_grip),
                }
            )
            if close_step is None and cur_grip > 0 and prev_grip <= 0:
                close_step = t
                close_snapshot = relative_snapshot(ee_pos, scene_refs)
            if reopen_step is None and cur_grip <= 0 and prev_grip > 0:
                reopen_step = t
                reopen_snapshot = relative_snapshot(ee_pos, scene_refs)

            success = env.check_success()
            if success and success_at is None:
                success_at = t
            if reach_only and success_at is None and reach_ready(ee_pos):
                success_at = t
            if reach_only and reach_stop_on_success and success_at is not None:
                break

            if t % 20 == 0 or t == n_steps - 1:
                q_now = np.array(robot._joint_positions)
                q_err = np.max(np.abs(q_now - q_traj[t, :7]))
                print(f"  t={t:3d}: EE=[{ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}] "
                      f"q_err={q_err:.4f} grip={effective_grip:.0f} "
                      f"{'SUCCESS' if success else ''}")

        if reach_only and reach_stop_on_success and success_at is not None:
            tail_applied = False
        else:
            tail_applied = False

        tail_descend_source_step = None
        if (not reach_only) and placement_tail and close_step is not None and reopen_step is None:
            ee_pos_tail_start = np.array(robot.controller.ee_pos)
            rel_tail_start = relative_snapshot(ee_pos_tail_start, scene_refs)
            ee_minus_plate = np.array(rel_tail_start.get("ee_minus_plate", [np.inf, np.inf, np.inf]))
            tail_ready = (
                abs(float(ee_minus_plate[0])) <= tail_plate_xy_m
                and abs(float(ee_minus_plate[1])) <= tail_plate_xy_m
                and float(ee_minus_plate[2]) <= tail_max_release_z_m
            )
            plate_candidates = [
                item for item in executed_trace
                if item["effective_grip"] > 0
                and abs(float(item["ee_minus_plate"][0])) <= tail_plate_xy_m
                and abs(float(item["ee_minus_plate"][1])) <= tail_plate_xy_m
                and float(item["ee_minus_plate"][2]) < float(ee_minus_plate[2]) - 1e-4
            ]
            if tail_ready:
                if plate_candidates:
                    descend_target = min(plate_candidates, key=lambda item: float(item["ee_minus_plate"][2]))
                    tail_descend_source_step = int(descend_target["t"])
                    tail_q_target = descend_target["q_target"]
                else:
                    tail_q_target = q_traj[-1, :7]

                obs = step_joint_target(tail_q_target, 1.0, tail_move_steps)
                ee_pos = np.array(robot.controller.ee_pos)
                if success_at is None and env.check_success():
                    success_at = n_steps

                prev_grip = last_effective_grip
                obs = step_joint_target(tail_q_target, -1.0, tail_open_steps)
                last_effective_grip = -1.0
                ee_pos = np.array(robot.controller.ee_pos)
                if reopen_step is None and prev_grip > 0:
                    reopen_step = n_steps + tail_move_steps
                    reopen_snapshot = relative_snapshot(ee_pos, scene_refs)
                if success_at is None and env.check_success():
                    success_at = n_steps + tail_move_steps

                obs = step_joint_target(q_traj[-1, :7], -1.0, tail_lift_steps)
                if success_at is None and env.check_success():
                    success_at = n_steps + tail_move_steps + tail_open_steps
                tail_applied = True

        # Settling steps: hold the final commanded gripper state instead of
        # forcibly reclosing. For placement tasks, force-closing here can undo
        # a late release and masks whether the rollout actually executed the
        # intended open/close sequence.
        if not (reach_only and reach_stop_on_success and success_at is not None):
            settle_grip = -1.0 if reach_only else (last_effective_grip if len(gripper_cmds) else 1.0)
            for i in range(100):
                q_target = q_traj[-1, :7]
                grip = settle_grip
                obs = step_joint_target(q_target, grip, 1)
                success = env.check_success()
                if success and success_at is None:
                    success_at = n_steps + i
                if reach_only and success_at is None and reach_ready(np.array(robot.controller.ee_pos)):
                    success_at = n_steps + i
                append_frame(obs)

        final_success = bool(success_at is not None) if reach_only else env.check_success()
        ee_final = np.array(robot.controller.ee_pos)
        motion_metrics = compute_motion_metrics(np.asarray(ee_trace, dtype=np.float32))
        print(f"  Final: EE={ee_final}, success={final_success}, reward={reward:.2f}")
        if success_at is not None:
            print(f"  First success at step {success_at}")
        if close_step is not None:
            print(f"  First close at step {close_step}: EE={close_snapshot['ee_pos']}")
        if reopen_step is not None:
            print(f"  First reopen at step {reopen_step}: EE={reopen_snapshot['ee_pos']}")

        writer.close()
        env.close()
        return {
            "success": final_success,
            "reward": float(reward),
            "success_at": success_at,
            "close_gate_xy_m": float(close_gate_xy_m),
            "close_gate_z_m": float(close_gate_z_m),
            "release_gate_xy_m": float(release_gate_xy_m),
            "release_gate_z_m": float(release_gate_z_m),
            "disable_gripper_gates": bool(disable_gripper_gates),
            "reach_only": bool(reach_only),
            "reach_threshold_xy_m": float(reach_threshold_xy_m),
            "reach_threshold_z_m": float(reach_threshold_z_m),
            "reach_stop_on_success": bool(reach_stop_on_success),
            "placement_tail": bool(placement_tail),
            "tail_applied": bool(tail_applied),
            "tail_descend_source_step": tail_descend_source_step,
            "close_step": close_step,
            "reopen_step": reopen_step,
            "close_snapshot": close_snapshot,
            "reopen_snapshot": reopen_snapshot,
            "final_snapshot": relative_snapshot(ee_final, scene_refs),
            "scene_refs": {
                "bowl_ref": scene_refs.get("bowl_ref"),
                "plate_ref": scene_refs.get("plate_ref"),
                "bowl_pos": scene_refs.get("bowl_pos").tolist() if scene_refs.get("bowl_pos") is not None else None,
                "plate_pos": scene_refs.get("plate_pos").tolist() if scene_refs.get("plate_pos") is not None else None,
            },
            "motion_metrics": motion_metrics,
            "video_path": video_path,
        }

    # Run all trajectories
    results_summary = {}
    default_results_summary = {}
    retry_assisted_summary = {}
    sim_rows = []
    for offset_cm in offsets_cm:
        offset_m = offset_cm / 100.0
        default_successes = 0
        retry_assisted_successes = 0
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
            episode_key = f"episode_idx_seed{seed}"
            if episode_key in data:
                episode_idx = int(np.asarray(data[episode_key]).reshape(-1)[0])
                demo_key = demo_key_for_episode(demo_keys, episode_idx)
            else:
                demo_key = demo_keys[seed] if seed < len(demo_keys) else f"seed_{seed}"
            # Use per-seed gripper if available
            gripper_key = f"gripper_seed{seed}"
            gripper = data[gripper_key] if gripper_key in data else gripper_default

            label = f"offset={offset_cm}cm seed={seed}"

            result = run_trajectory(
                q_traj, gripper, init_state, label,
                offset_m=offset_m,
                close_gate_xy_m=args.close_gate_xy_m,
                close_gate_z_m=args.close_gate_z_m,
                release_gate_xy_m=args.release_gate_xy_m,
                release_gate_z_m=args.release_gate_z_m,
                disable_gripper_gates=args.disable_gripper_gates,
                reach_only=args.reach_only,
                reach_threshold_xy_m=args.reach_threshold_xy_m,
                reach_threshold_z_m=args.reach_threshold_z_m,
                reach_stop_on_success=args.reach_stop_on_success,
                placement_tail=args.placement_tail,
                tail_plate_xy_m=args.tail_plate_xy_m,
                tail_max_release_z_m=args.tail_max_release_z_m,
                tail_move_steps=args.tail_move_steps,
                tail_open_steps=args.tail_open_steps,
                tail_lift_steps=args.tail_lift_steps,
            )
            sim_rows.append({
                "offset_cm": float(offset_cm),
                "seed": seed,
                "demo_key": demo_key,
                "attempt": "default",
                **result,
            })

            if result["success"]:
                default_successes += 1
                retry_assisted_successes += 1
            total += 1

            # If first attempt fails, try higher gains
            if not result["success"]:
                print(f"  Retrying with higher gains...")
                result2 = run_trajectory(
                    q_traj, gripper, init_state, f"{label} (high gains)",
                    output_max=1.0, kp=250, substeps=5,
                    offset_m=offset_m,
                    close_gate_xy_m=args.close_gate_xy_m,
                    close_gate_z_m=args.close_gate_z_m,
                    release_gate_xy_m=args.release_gate_xy_m,
                    release_gate_z_m=args.release_gate_z_m,
                    reach_only=args.reach_only,
                    reach_threshold_xy_m=args.reach_threshold_xy_m,
                    reach_threshold_z_m=args.reach_threshold_z_m,
                    reach_stop_on_success=args.reach_stop_on_success,
                    placement_tail=args.placement_tail,
                    tail_plate_xy_m=args.tail_plate_xy_m,
                    tail_max_release_z_m=args.tail_max_release_z_m,
                    tail_move_steps=args.tail_move_steps,
                    tail_open_steps=args.tail_open_steps,
                    tail_lift_steps=args.tail_lift_steps,
                )
                sim_rows.append({
                    "offset_cm": float(offset_cm),
                    "seed": seed,
                    "demo_key": demo_key,
                    "attempt": "high_gains",
                    **result2,
                })
                if result2["success"]:
                    retry_assisted_successes += 1

        default_rate = default_successes / max(total, 1) * 100
        retry_rate = retry_assisted_successes / max(total, 1) * 100
        key = f"{offset_cm}cm"
        default_results_summary[key] = {
            "successes": default_successes,
            "total": total,
            "rate": default_rate,
        }
        retry_assisted_summary[key] = {
            "successes": retry_assisted_successes,
            "total": total,
            "rate": retry_rate,
        }
        # Keep the legacy field as default-attempt success for headline tables.
        results_summary[key] = default_results_summary[key]
        print(
            f"\n  === offset={offset_cm}cm: default {default_successes}/{total} = "
            f"{default_rate:.0f}%, retry-assisted {retry_assisted_successes}/{total} = "
            f"{retry_rate:.0f}% ==="
        )

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Source: {source}")
    for k, v in results_summary.items():
        print(f"  {k}: {v['successes']}/{v['total']} = {v['rate']:.0f}%")
    write_debug_summary(build_stage_summary_path(args, "sim"), {
        "stage": "sim",
        "source": source,
        "summary": results_summary,
        "default_summary": default_results_summary,
        "retry_assisted_summary": retry_assisted_summary,
        "runs": sim_rows,
    })


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
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Checkpoint produced by train_dhb_head_e2e.py (required for predict stage)")
    parser.add_argument("--debug-json", default=None,
                        help="Base path for per-stage debug JSON summaries")
    parser.add_argument("--video-dir", default="_workspace/validation_videos",
                        help="Directory for saved sim-stage videos")
    parser.add_argument("--poses-npz", default=POSES_NPZ,
                        help="Inter-stage pose NPZ path for encode_gt/predict output and IK input")
    parser.add_argument("--joints-npz", default=JOINTS_NPZ,
                        help="Inter-stage joint NPZ path for IK output and sim input")
    parser.add_argument("--ik-line-search-fallback", action="store_true",
                        help="Repair infeasible IK targets by backing off along the last-reachable to desired target segment.")
    parser.add_argument("--ik-line-search-steps", type=int, default=8,
                        help="Number of alpha samples for IK line-search fallback.")
    parser.add_argument("--ik-line-search-min-alpha", type=float, default=0.0,
                        help="Minimum segment fraction to try for IK line-search fallback.")
    parser.add_argument("--close-gate-xy-m", type=float, default=0.05,
                        help="Only allow first close when EE XY is within this distance of the bowl")
    parser.add_argument("--close-gate-z-m", type=float, default=0.05,
                        help="Only allow first close when EE Z is below this height above the bowl")
    parser.add_argument("--release-gate-xy-m", type=float, default=0.04,
                        help="Only allow first reopen when EE XY is within this distance of the plate")
    parser.add_argument("--release-gate-z-m", type=float, default=0.06,
                        help="Only allow first reopen when EE Z is below this height above the plate")
    parser.add_argument("--disable-gripper-gates", action="store_true",
                        help="Do not veto close / reopen transitions with live bowl / plate geometry gates.")
    parser.add_argument("--reach-only", action="store_true",
                        help="Ignore gripper timing and score smooth reach-to-bowl instead of full task success.")
    parser.add_argument("--reach-threshold-xy-m", type=float, default=0.05,
                        help="XY bowl-distance threshold for reach-only success.")
    parser.add_argument("--reach-threshold-z-m", type=float, default=0.08,
                        help="Maximum EE height above the bowl for reach-only success.")
    parser.add_argument("--reach-stop-on-success", action="store_true",
                        help="Stop executing once the bowl reach criterion is satisfied.")
    parser.add_argument("--placement-tail", action="store_true",
                        help="If still holding near the plate at the end, reuse the lowest recent plate-near joint target before opening")
    parser.add_argument("--tail-plate-xy-m", type=float, default=0.05,
                        help="Plate XY tolerance for activating the placement tail")
    parser.add_argument("--tail-max-release-z-m", type=float, default=0.08,
                        help="Maximum EE height above the plate for the placement tail to activate")
    parser.add_argument("--tail-move-steps", type=int, default=12,
                        help="Control steps to move toward the reused plate-near descend target")
    parser.add_argument("--tail-open-steps", type=int, default=12,
                        help="Control steps to hold the descend target while opening")
    parser.add_argument("--tail-lift-steps", type=int, default=12,
                        help="Control steps to return to the final target after opening")
    args = parser.parse_args()

    if args.stage == "encode_gt":
        stage_encode_gt(args)
    elif args.stage == "predict":
        stage_predict(args)
    elif args.stage == "ik":
        stage_ik(args)
    elif args.stage == "sim":
        stage_sim(args)
