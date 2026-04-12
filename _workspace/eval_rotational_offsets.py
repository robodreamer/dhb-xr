#!/usr/bin/env python3
"""
Step 1.6: SmolVLA degradation under rotational workspace perturbation.

Tests whether the base SmolVLA degrades under rotational changes too.
Rotates the bowl position around the robot base by various angles.

Usage:
    MUJOCO_GL=egl ~/miniforge3/envs/smolvla/bin/python \
        _workspace/eval_rotational_offsets.py --seeds 10
"""
from __future__ import annotations
import os, sys, json, argparse, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, "/home/andypark/Projects/repos/LIBERO")

BOWL_STATE_INDICES = {'X': 10, 'Y': 11, 'Z': 12}


def make_env(task_id=0, resolution=256):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(**{
        "bddl_file_name": bddl,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_names": ["agentview", "robot0_eye_in_hand"],
    })
    env.seed(0)
    dpath = os.path.join(get_libero_path("datasets"), "libero_spatial",
                         f"{task.name}_demo.hdf5")
    return env, dpath, task.name.replace("_", " ")


def load_demo(dpath, demo_idx=0):
    import h5py
    with h5py.File(dpath) as f:
        key = sorted(f["data"].keys())[demo_idx]
        d = f["data"][key]
        return {
            "init_state": np.array(d["states"][0]),
            "actions": np.array(d["actions"]),
        }


def quat_to_axis_angle(quat):
    import torch
    q = torch.from_numpy(np.array(quat, dtype=np.float32))
    w = q[3].clamp(-1.0, 1.0)
    den = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0))
    if den > 1e-10:
        angle = 2.0 * torch.acos(w)
        axis = q[:3] / den
        return (axis * angle).numpy()
    return np.zeros(3, dtype=np.float32)


def load_smolvla(device="cuda"):
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    import lerobot.policies.smolvla.processor_smolvla
    from lerobot.policies.factory import make_pre_post_processors

    model_id = "HuggingFaceVLA/smolvla_libero"
    policy = SmolVLAPolicy.from_pretrained(model_id)
    policy = policy.to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=model_id
    )
    policy._preprocessor = preprocessor
    policy._postprocessor = postprocessor
    return policy


def apply_rotation_to_bowl(init_state, angle_deg):
    """Rotate bowl position around the robot base (Z-axis rotation)."""
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    # Get bowl XY position
    bx = init_state[BOWL_STATE_INDICES['X']]
    by = init_state[BOWL_STATE_INDICES['Y']]

    # Rotate around origin (robot base)
    new_bx = cos_a * bx - sin_a * by
    new_by = sin_a * bx + cos_a * by

    rotated = init_state.copy()
    rotated[BOWL_STATE_INDICES['X']] = new_bx
    rotated[BOWL_STATE_INDICES['Y']] = new_by
    return rotated


def run_trial(env, policy, demo, angle_deg, task_name, max_steps=80, device="cuda"):
    import torch

    init = apply_rotation_to_bowl(demo["init_state"], angle_deg)
    env.reset()
    obs = env.set_init_state(init)
    for _ in range(10):
        obs, _, _, _ = env.step([0.0] * 7)

    policy.reset()
    reward = 0.0

    for step in range(max_steps):
        with torch.no_grad():
            agentview = obs["agentview_image"].astype(np.uint8)
            eih = obs.get("robot0_eye_in_hand_image", agentview).astype(np.uint8)
            img1 = torch.from_numpy(agentview).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            img2 = torch.from_numpy(eih).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            img1 = torch.flip(img1, dims=[2, 3])
            img2 = torch.flip(img2, dims=[2, 3])

            eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
            eef_quat = obs.get("robot0_eef_quat", np.array([0, 0, 0, 1], dtype=np.float64))
            eef_aa = quat_to_axis_angle(eef_quat)
            gripper_qpos = obs.get("robot0_gripper_qpos", np.zeros(2))
            state_vec = np.concatenate([eef_pos, eef_aa, gripper_qpos]).astype(np.float32)
            state_t = torch.from_numpy(state_vec).unsqueeze(0)

            batch = {
                "observation.images.image": img1,
                "observation.images.image2": img2,
                "observation.state": state_t,
                "task": [task_name],
            }
            batch = policy._preprocessor(batch)
            action = policy.select_action(batch)
            action = policy._postprocessor(action)

        if isinstance(action, torch.Tensor):
            action_np = action[0].cpu().numpy() if action.ndim == 2 else action.cpu().numpy()
        elif isinstance(action, dict):
            action_np = action["action"]
            if isinstance(action_np, torch.Tensor):
                action_np = action_np[0].cpu().numpy() if action_np.ndim == 2 else action_np.cpu().numpy()
        else:
            action_np = np.array(action)

        obs, r, done, _ = env.step(action_np.tolist())
        reward = r
        if done:
            break

    return reward > 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--angles", type=float, nargs="+",
                        default=[0, 5, 10, 15, 20, 30])
    parser.add_argument("--out", default="_workspace/rotational_degradation.json")
    args = parser.parse_args()

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Angles (deg): {args.angles}")
    print(f"Seeds: {args.seeds}")

    policy = load_smolvla(device=device)
    env, dpath, task_name = make_env()
    print(f"Task: {task_name}")

    import h5py
    with h5py.File(dpath) as f:
        n_demos = len(f["data"].keys())

    results = {"angles_deg": args.angles, "seeds": args.seeds, "summary": {}}

    for angle in args.angles:
        successes = 0
        for seed in range(args.seeds):
            demo = load_demo(dpath, seed % n_demos)
            try:
                ok = run_trial(env, policy, demo, angle, task_name, device=device)
                successes += int(ok)
            except Exception as e:
                print(f"  Seed {seed}: ERROR - {e}")

            status = "OK" if ok else "FAIL"
            print(f"  {angle:3.0f}deg seed {seed:2d}: {status} [{successes}/{seed+1}]")

        rate = successes / args.seeds
        results["summary"][f"{angle}deg"] = {
            "successes": successes, "total": args.seeds, "rate": rate
        }
        print(f"  >>> {angle}deg: {successes}/{args.seeds} = {rate:.0%}\n")

    env.close()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {args.out}")

    print("\n=== Rotational Degradation ===")
    for angle in args.angles:
        s = results["summary"][f"{angle}deg"]
        print(f"  {angle:5.0f}deg: {s['successes']}/{s['total']} = {s['rate']:.0%}")


if __name__ == "__main__":
    main()
