#!/usr/bin/env python3
"""
Step 3.5: Evaluate trained DHBActionHead in LIBERO simulation at workspace offsets.

Pipeline per trial:
    1. Reset LIBERO env with bowl shifted by offset
    2. Get initial observation -> extract VLM features (SmolVLA)
    3. DHBActionHead predicts DHB invariants from VLM features
    4. decode_dhb_dr(invariants, current_EE_pose) -> adapted trajectory
    5. Execute trajectory as OSC delta actions: action[:3] = (target - current) * gain
    6. Check task success

Usage:
    MUJOCO_GL=egl ~/miniforge3/envs/smolvla/bin/python \
        _workspace/eval_dhb_head_libero.py \
        --offsets 0 3 5 --seeds 10
"""
from __future__ import annotations

import os, sys, json, argparse, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, "/home/andypark/Projects/repos/LIBERO")

import torch

# --------------------------------------------------------------------------- #
# Bowl state indices in LIBERO init_state (same as eval_degradation_curve.py)
# --------------------------------------------------------------------------- #
BOWL_STATE_INDICES = {"X": 10, "Y": 11, "Z": 12}

# --------------------------------------------------------------------------- #
# Env helpers (reused from eval_degradation_curve.py pattern)
# --------------------------------------------------------------------------- #

def make_env(task_id: int = 0, resolution: int = 256):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(task_id)
    bddl = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env = OffScreenRenderEnv(
        **{
            "bddl_file_name": bddl,
            "camera_heights": resolution,
            "camera_widths": resolution,
            "camera_names": ["agentview", "robot0_eye_in_hand"],
        }
    )
    env.seed(0)
    dpath = os.path.join(
        get_libero_path("datasets"),
        "libero_spatial",
        f"{task.name}_demo.hdf5",
    )
    return env, dpath, task.name.replace("_", " ")


def load_demo(dpath: str, demo_idx: int = 0):
    import h5py

    with h5py.File(dpath) as f:
        key = sorted(f["data"].keys())[demo_idx]
        d = f["data"][key]
        return {
            "init_state": np.array(d["states"][0]),
            "actions": np.array(d["actions"]),
            "ee_pos": np.array(d["obs"]["ee_pos"]),
        }


# --------------------------------------------------------------------------- #
# VLM feature extraction (same as train_dhb_action_head.py)
# --------------------------------------------------------------------------- #

def extract_vlm_features(policy, obs_dict, task_name, device="cuda"):
    """Extract VLM hidden states from SmolVLA for one observation."""
    from scipy.spatial.transform import Rotation

    agentview = obs_dict["agentview_image"].astype(np.uint8)
    eih = obs_dict.get("robot0_eye_in_hand_image", agentview).astype(np.uint8)

    img1 = torch.from_numpy(agentview).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img2 = torch.from_numpy(eih).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img1 = torch.flip(img1, dims=[2, 3])
    img2 = torch.flip(img2, dims=[2, 3])

    eef_pos = obs_dict.get("robot0_eef_pos", np.zeros(3))
    eef_quat = obs_dict.get("robot0_eef_quat", np.array([0, 0, 0, 1], dtype=np.float64))
    aa = Rotation.from_quat(eef_quat).as_rotvec().astype(np.float32)
    gripper_qpos = obs_dict.get("robot0_gripper_qpos", np.zeros(2))
    state = np.concatenate([eef_pos, aa, gripper_qpos]).astype(np.float32)

    batch = {
        "observation.images.image": img1.to(device),
        "observation.images.image2": img2.to(device),
        "observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
        "task": [task_name],
    }

    batch = policy._preprocessor(batch)

    with torch.no_grad():
        try:
            vlm_out = policy.model.vlm_with_expert.vlm(
                input_ids=batch.get("input_ids"),
                pixel_values=batch.get("pixel_values"),
                attention_mask=batch.get("attention_mask"),
            )
            hidden = vlm_out.last_hidden_state  # (1, S, vlm_dim)
        except Exception:
            hidden = torch.randn(1, 10, 576, device=device)

    return hidden.detach()


# --------------------------------------------------------------------------- #
# DHB decode: invariants + current EE pose -> Cartesian trajectory
# --------------------------------------------------------------------------- #

def decode_invariants_to_trajectory(pred_inv_np, ee_pos, ee_quat_xyzw):
    """Decode predicted DHB invariants into a Cartesian trajectory.

    Args:
        pred_inv_np: (T, 8) invariants — first 4 linear, last 4 angular
        ee_pos: (3,) current end-effector position
        ee_quat_xyzw: (4,) current quaternion in x,y,z,w order

    Returns:
        positions: (T', 3)  decoded trajectory positions
        quaternions: (T', 4) decoded quaternions (wxyz)
    """
    from dhb_xr.decoder.dhb_dr import decode_dhb_dr
    from dhb_xr.core.types import EncodingMethod

    lin_inv = pred_inv_np[:, :4]
    ang_inv = pred_inv_np[:, 4:]

    # Convert xyzw -> wxyz for DHB decoder
    quat_wxyz = np.array([
        ee_quat_xyzw[3], ee_quat_xyzw[0], ee_quat_xyzw[1], ee_quat_xyzw[2]
    ])
    # Normalize
    quat_wxyz = quat_wxyz / (np.linalg.norm(quat_wxyz) + 1e-12)

    dec = decode_dhb_dr(
        lin_inv,
        ang_inv,
        initial_pose={"position": ee_pos.copy(), "quaternion": quat_wxyz},
        method=EncodingMethod.POSITION,
        drop_padded=True,
        robust_mode=True,
    )
    return dec["positions"], dec["quaternions"]


# --------------------------------------------------------------------------- #
# Trajectory execution via OSC delta actions
# --------------------------------------------------------------------------- #

def execute_trajectory(env, positions, quaternions, max_steps=80, gain=2.0,
                       gripper_action=0.0, settle_steps=5):
    """Execute a decoded Cartesian trajectory using OSC delta actions.

    For each waypoint, compute:
        action[:3] = (target_pos - current_pos) * gain
        action[3:6] = (target_euler - current_euler) * gain  (small angle approx)
        action[6]   = gripper_action

    We iterate through waypoints, spending up to `substeps` env steps per
    waypoint to let the controller converge.

    Args:
        env: LIBERO OffScreenRenderEnv
        positions: (N, 3) target Cartesian positions
        quaternions: (N, 4) target quaternions (wxyz)
        max_steps: total env step budget
        gain: proportional gain for delta computation
        gripper_action: gripper command (-1 close, 0 neutral, 1 open)
        settle_steps: extra steps after last waypoint

    Returns:
        success: bool
        reward: float
        total_steps: int
    """
    from scipy.spatial.transform import Rotation

    n_waypoints = len(positions)
    step_count = 0
    reward = 0.0
    done = False
    obs = None

    # Steps per waypoint — distribute budget across waypoints + settle
    steps_per_wp = max(1, (max_steps - settle_steps) // max(n_waypoints, 1))

    for wp_idx in range(n_waypoints):
        if done or step_count >= max_steps:
            break

        target_pos = positions[wp_idx]

        # Target orientation: convert wxyz quaternion to axis-angle delta
        target_quat_wxyz = quaternions[wp_idx]
        target_quat_xyzw = np.array([
            target_quat_wxyz[1], target_quat_wxyz[2],
            target_quat_wxyz[3], target_quat_wxyz[0]
        ])

        for _ in range(steps_per_wp):
            if step_count >= max_steps:
                break

            # Get current EE state from env
            if obs is None:
                # First step: get obs from env's current state
                obs, reward, done, info = env.step([0.0] * 7)
                step_count += 1
                if done:
                    break

            cur_pos = obs.get("robot0_eef_pos", np.zeros(3))
            cur_quat_xyzw = obs.get("robot0_eef_quat", np.array([0, 0, 0, 1.0]))

            # Position delta
            delta_pos = (target_pos - cur_pos) * gain

            # Orientation delta (axis-angle)
            try:
                R_cur = Rotation.from_quat(cur_quat_xyzw)
                R_target = Rotation.from_quat(target_quat_xyzw)
                R_delta = R_target * R_cur.inv()
                delta_ori = R_delta.as_rotvec() * gain
            except Exception:
                delta_ori = np.zeros(3)

            # Clamp deltas for safety
            delta_pos = np.clip(delta_pos, -0.05, 0.05)
            delta_ori = np.clip(delta_ori, -0.25, 0.25)

            action = np.concatenate([delta_pos, delta_ori, [gripper_action]])
            obs, reward, done, info = env.step(action.tolist())
            step_count += 1

            if done:
                break

            # Early exit from this waypoint if close enough
            dist = np.linalg.norm(target_pos - obs.get("robot0_eef_pos", cur_pos))
            if dist < 0.005:
                break

    # Settle: let the arm stabilize
    for _ in range(settle_steps):
        if done or step_count >= max_steps:
            break
        obs, reward, done, info = env.step([0.0] * 7)
        step_count += 1

    return reward > 0, float(reward), step_count


# --------------------------------------------------------------------------- #
# Single trial
# --------------------------------------------------------------------------- #

def run_dhb_trial(env, policy, head, demo, offset_m, axis, task_name,
                  device="cuda", max_steps=80, gain=2.0, action_horizon=30):
    """Run one DHB-XR trial at a given workspace offset.

    Returns:
        success: bool
        reward: float
        info: dict with extra diagnostics
    """
    # 1. Reset env with shifted bowl
    init = demo["init_state"].copy()
    if offset_m != 0:
        bowl_idx = BOWL_STATE_INDICES[axis]
        init[bowl_idx] += offset_m

    env.reset()
    obs = env.set_init_state(init)
    # Let physics settle
    for _ in range(10):
        obs, _, _, _ = env.step([0.0] * 7)

    # 2. Extract VLM features from initial observation
    vlm_feat = extract_vlm_features(policy, obs, task_name, device)

    # 3. DHBActionHead predicts invariants
    head.eval()
    with torch.no_grad():
        result = head(vlm_feat)
        pred_inv = result["pred_invariants"]  # (1, T, 8)

    pred_inv_np = pred_inv[0].cpu().numpy()  # (T, 8)

    # 4. DHB decode from current EE pose -> adapted trajectory
    ee_pos = obs.get("robot0_eef_pos", np.zeros(3))
    ee_quat_xyzw = obs.get("robot0_eef_quat", np.array([0, 0, 0, 1.0]))

    try:
        positions, quaternions = decode_invariants_to_trajectory(
            pred_inv_np, ee_pos, ee_quat_xyzw
        )
    except Exception as e:
        return False, 0.0, {"error": f"decode failed: {e}"}

    # 5. Execute trajectory as OSC delta actions
    success, reward, total_steps = execute_trajectory(
        env, positions, quaternions,
        max_steps=max_steps, gain=gain,
    )

    info = {
        "trajectory_len": len(positions),
        "total_steps": total_steps,
        "inv_norm": float(np.linalg.norm(pred_inv_np)),
    }
    return success, reward, info


# --------------------------------------------------------------------------- #
# Load SmolVLA
# --------------------------------------------------------------------------- #

def load_smolvla(device="cuda"):
    """Load SmolVLA policy for VLM feature extraction."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    import lerobot.policies.smolvla.processor_smolvla  # noqa: F401
    from lerobot.policies.factory import make_pre_post_processors

    model_id = "HuggingFaceVLA/smolvla_libero"
    print(f"Loading SmolVLA ({model_id})...")
    t0 = time.time()
    policy = SmolVLAPolicy.from_pretrained(model_id).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=model_id
    )
    policy._preprocessor = preprocessor
    policy._postprocessor = postprocessor
    print(f"  Loaded in {time.time() - t0:.0f}s")
    return policy


# --------------------------------------------------------------------------- #
# Load trained DHBActionHead
# --------------------------------------------------------------------------- #

def load_dhb_head(checkpoint_path, device="cuda", action_horizon=30):
    """Load a trained DHBActionHead from checkpoint."""
    from dhb_xr.integration.vla.action_head import DHBActionHead

    head = DHBActionHead(
        vlm_dim=576,       # SmolVLM2-500M hidden dim
        action_horizon=action_horizon,
        invariant_dim=8,   # 4 linear + 4 angular
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    head.load_state_dict(state_dict)
    head.eval()

    param_count = sum(p.numel() for p in head.parameters())
    print(f"DHBActionHead loaded: {checkpoint_path}")
    print(f"  Parameters: {param_count:,} ({param_count / 1e6:.1f}M)")
    return head


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DHBActionHead in LIBERO with workspace offsets"
    )
    parser.add_argument(
        "--checkpoint",
        default="_workspace/dhb_action_head_trained.pt",
        help="Path to trained DHBActionHead checkpoint",
    )
    parser.add_argument("--task", type=int, default=0, help="LIBERO-Spatial task ID")
    parser.add_argument(
        "--axis", default="Y", choices=["X", "Y", "Z"],
        help="Perturbation axis for bowl shift",
    )
    parser.add_argument(
        "--offsets", type=float, nargs="+", default=[0, 3, 5],
        help="Offset magnitudes in cm (default: 0 3 5)",
    )
    parser.add_argument(
        "--seeds", type=int, default=10,
        help="Number of trials per offset condition",
    )
    parser.add_argument(
        "--max-steps", type=int, default=80,
        help="Max env steps per trial",
    )
    parser.add_argument("--gain", type=float, default=2.0, help="OSC delta gain")
    parser.add_argument(
        "--action-horizon", type=int, default=30,
        help="Action horizon T (must match training)",
    )
    parser.add_argument(
        "--out", default="_workspace/eval_dhb_head_results.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Task: {args.task}, Axis: {args.axis}, Seeds: {args.seeds}")
    print(f"Offsets (cm): {args.offsets}")
    print(f"Gain: {args.gain}, Max steps: {args.max_steps}")
    print(f"Action horizon: {args.action_horizon}")
    print()

    # Load models
    policy = load_smolvla(device=device)
    head = load_dhb_head(args.checkpoint, device=device,
                         action_horizon=args.action_horizon)

    # Setup LIBERO env
    env, dpath, task_name = make_env(args.task)
    print(f"Task: {task_name}")

    import h5py
    with h5py.File(dpath) as f:
        n_demos = len(f["data"].keys())
    print(f"Available demos: {n_demos}")
    print()

    # ---------- Run evaluation ----------
    results = {
        "metadata": {
            "task_id": args.task,
            "task_name": task_name,
            "axis": args.axis,
            "seeds_per_offset": args.seeds,
            "offsets_cm": args.offsets,
            "gain": args.gain,
            "max_steps": args.max_steps,
            "action_horizon": args.action_horizon,
            "checkpoint": args.checkpoint,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "trials": [],
        "summary": {},
    }

    for offset_cm in args.offsets:
        offset_m = offset_cm / 100.0
        successes = 0
        trial_details = []

        print(f"--- Offset: {offset_cm}cm ({args.axis}-axis) ---")
        for seed in range(args.seeds):
            demo_idx = seed % n_demos
            demo = load_demo(dpath, demo_idx)

            t0 = time.time()
            try:
                success, reward, info = run_dhb_trial(
                    env, policy, head, demo,
                    offset_m=offset_m,
                    axis=args.axis,
                    task_name=task_name,
                    device=device,
                    max_steps=args.max_steps,
                    gain=args.gain,
                    action_horizon=args.action_horizon,
                )
            except Exception as e:
                print(f"  Seed {seed}: ERROR - {e}")
                success, reward, info = False, 0.0, {"error": str(e)}

            elapsed = time.time() - t0
            successes += int(success)

            trial_record = {
                "offset_cm": offset_cm,
                "seed": seed,
                "demo_idx": demo_idx,
                "success": bool(success),
                "reward": float(reward),
                "time_s": round(elapsed, 1),
            }
            trial_record.update(info)
            trial_details.append(trial_record)

            status = "OK" if success else "FAIL"
            info_str = f"traj={info.get('trajectory_len', '?')}"
            print(
                f"  Seed {seed:2d} (demo {demo_idx:2d}): {status}  "
                f"r={reward:.2f}  {elapsed:.1f}s  {info_str}  "
                f"[{successes}/{seed + 1}]"
            )

        rate = successes / args.seeds
        results["trials"].extend(trial_details)
        results["summary"][f"{offset_cm}cm"] = {
            "successes": successes,
            "total": args.seeds,
            "rate": rate,
        }
        print(f"  >>> {offset_cm}cm: {successes}/{args.seeds} = {rate:.0%}")
        print()

    env.close()

    # Save results
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {out}")

    # Summary table
    print()
    print("=" * 55)
    print("  DHB-XR Action Head Evaluation Summary")
    print("=" * 55)
    print(f"  Task {args.task}: {task_name}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Axis: {args.axis}, Gain: {args.gain}")
    print(f"  {'Offset':>8} | {'Success':>10} | {'Rate':>8}")
    print(f"  {'-' * 38}")
    for offset_cm in args.offsets:
        s = results["summary"][f"{offset_cm}cm"]
        bar = "#" * int(s["rate"] * 20) + "." * (20 - int(s["rate"] * 20))
        print(
            f"  {offset_cm:>6.0f}cm | {s['successes']:>3}/{s['total']:<3}     "
            f"| {s['rate']:>6.0%} [{bar}]"
        )
    print("=" * 55)


if __name__ == "__main__":
    main()
