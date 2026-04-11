#!/usr/bin/env python3
"""
Step 2.4: Build LeRobot-format augmented dataset for SmolVLA fine-tuning.

For each original LIBERO demo + each DHB-augmented offset:
1. Reset env with bowl shifted to offset position
2. Replay the augmented actions while capturing images
3. Record frames in LeRobot format

This ensures images match the offset workspace, not just actions.

Usage:
    MUJOCO_GL=egl ~/miniforge3/envs/smolvla/bin/python \
        _workspace/build_augmented_dataset.py \
        --mode dhb --task 0

Three runs needed:
    --mode original  (50 demos, 0cm offset)
    --mode dhb       (450 demos at random offsets)
    --mode naive     (450 demos at same random offsets)
"""
from __future__ import annotations
import os, sys, argparse, time, json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, "/home/andypark/Projects/repos/LIBERO")

BOWL_STATE_INDICES = {'X': 10, 'Y': 11, 'Z': 12}
FPS = 10  # LIBERO control rate


def make_env(task_id: int = 0, resolution: int = 256):
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


def load_demos_and_augments(dpath, aug_path, mode):
    """Load original demos and augmented actions."""
    import h5py

    # Load original demos
    demos = []
    with h5py.File(dpath) as f:
        for key in sorted(f["data"].keys()):
            d = f["data"][key]
            demos.append({
                "init_state": np.array(d["states"][0]),
                "actions": np.array(d["actions"]),
            })

    if mode == "original":
        # Just use original demos with zero offset
        return [(d, np.zeros(3), d["actions"]) for d in demos]

    # Load augmented data
    aug = np.load(aug_path)
    n_aug = int(aug["n_augmented_per_demo"])
    offsets = aug["offsets_used"]

    if mode == "dhb":
        lengths = aug["dhb_augmented_lengths"]
        all_actions = aug["dhb_augmented_actions"]
    elif mode == "naive":
        lengths = aug["naive_augmented_lengths"]
        all_actions = aug["naive_augmented_actions"]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Split actions by demo
    aug_actions = []
    idx = 0
    for l in lengths:
        aug_actions.append(all_actions[idx:idx+l])
        idx += l

    # Build (demo, offset, actions) tuples
    result = []
    for i, demo in enumerate(demos):
        for j in range(n_aug):
            aug_idx = i * n_aug + j
            offset = offsets[aug_idx]
            actions = aug_actions[aug_idx]
            result.append((demo, offset, actions))

    return result


def quat_to_axis_angle(quat):
    """Convert quaternion (x,y,z,w) to axis-angle (3,)."""
    import torch
    q = torch.from_numpy(np.array(quat, dtype=np.float32))
    w = q[3].clamp(-1.0, 1.0)
    den = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0))
    if den > 1e-10:
        angle = 2.0 * torch.acos(w)
        axis = q[:3] / den
        return (axis * angle).numpy()
    return np.zeros(3, dtype=np.float32)


def replay_and_record(env, demo, offset, actions, task_name):
    """Replay actions in env at offset, recording observations.

    Returns list of frames, each a dict with images, state, action.
    """
    import torch

    init = demo["init_state"].copy()
    # Apply offset to bowl position
    for ax_idx, ax in enumerate(['X', 'Y', 'Z']):
        init[BOWL_STATE_INDICES[ax]] += offset[ax_idx]

    env.reset()
    obs = env.set_init_state(init)
    # Stabilize
    for _ in range(5):
        obs, _, _, _ = env.step([0.0] * 7)

    frames = []
    for t in range(len(actions)):
        # Capture observation BEFORE taking action
        agentview = obs["agentview_image"].astype(np.uint8)
        eih = obs.get("robot0_eye_in_hand_image", agentview).astype(np.uint8)
        eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
        eef_quat = obs.get("robot0_eef_quat", np.array([0, 0, 0, 1], dtype=np.float64))
        eef_aa = quat_to_axis_angle(eef_quat)
        gripper_qpos = obs.get("robot0_gripper_qpos", np.zeros(2))
        state = np.concatenate([eef_pos, eef_aa, gripper_qpos]).astype(np.float32)

        frame = {
            "observation.images.image": agentview,       # (H,W,3) uint8
            "observation.images.image2": eih,             # (H,W,3) uint8
            "observation.state": state,                   # (8,) float32
            "action": actions[t].astype(np.float32),      # (7,) float32
            "task": task_name,
        }
        frames.append(frame)

        # Take action
        obs, r, done, _ = env.step(actions[t].tolist())
        if done:
            break

    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["original", "dhb", "naive"], required=True)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--aug-file", default="_workspace/augmented_demos_task0.npz")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Limit number of episodes (for testing)")
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = f"_workspace/lerobot_dataset_{args.mode}_task{args.task}"

    env, dpath, task_name = make_env(args.task)
    episodes = load_demos_and_augments(dpath, args.aug_file, args.mode)

    if args.max_episodes:
        episodes = episodes[:args.max_episodes]

    print(f"Mode: {args.mode}")
    print(f"Task: {task_name}")
    print(f"Episodes: {len(episodes)}")
    print(f"Output: {args.out_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_episodes = []
    t0 = time.time()

    for ep_idx, (demo, offset, actions) in enumerate(episodes):
        frames = replay_and_record(env, demo, offset, actions, task_name)

        # Save episode as npz
        ep_data = {
            "images": np.array([f["observation.images.image"] for f in frames]),
            "images2": np.array([f["observation.images.image2"] for f in frames]),
            "states": np.array([f["observation.state"] for f in frames]),
            "actions": np.array([f["action"] for f in frames]),
            "task": task_name,
            "offset": offset,
        }
        ep_path = out_dir / f"episode_{ep_idx:04d}.npz"
        np.savez_compressed(ep_path, **ep_data)

        all_episodes.append({
            "episode_idx": ep_idx,
            "n_frames": len(frames),
            "offset_cm": [round(o * 100, 1) for o in offset],
        })

        if (ep_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (ep_idx + 1) / elapsed
            eta = (len(episodes) - ep_idx - 1) / rate
            print(f"  Episode {ep_idx+1}/{len(episodes)} "
                  f"({elapsed:.0f}s elapsed, {eta:.0f}s remaining)")

    env.close()
    elapsed = time.time() - t0

    # Save metadata
    meta = {
        "mode": args.mode,
        "task_id": args.task,
        "task_name": task_name,
        "n_episodes": len(all_episodes),
        "fps": FPS,
        "episodes": all_episodes,
        "total_time_s": round(elapsed, 1),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    total_frames = sum(e["n_frames"] for e in all_episodes)
    size_mb = sum(f.stat().st_size for f in out_dir.glob("*.npz")) / 1e6
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Episodes: {len(all_episodes)}")
    print(f"  Total frames: {total_frames}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Saved to: {out_dir}")


if __name__ == "__main__":
    main()
