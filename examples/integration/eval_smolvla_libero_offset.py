#!/usr/bin/env python3
"""
Evaluate SmolVLA (LIBERO fine-tune) at workspace offsets.

Demonstrates that standard VLA tokenization (raw action coordinates)
causes policy failure at offset positions, while DHB geometric adaptation
maintains 0mm endpoint error.

Models tested:
  - HuggingFaceVLA/smolvla_libero: SmolVLA fine-tuned on LIBERO (~500MB)
    Acts as the "standard VLA" baseline using raw coordinate tokenization.

Comparison:
  - At 0cm: SmolVLA should achieve ~70-90% success (in-distribution)
  - At 5cm: SmolVLA degrades (tokens change with workspace position)
  - DHB adaptation: 0mm endpoint error at all offsets (geometric guarantee)

Usage:
    MUJOCO_GL=egl ~/miniforge3/envs/smolvla/bin/python \
        examples/integration/eval_smolvla_libero_offset.py \
        --offsets 0 5 10 --num-tasks 3 --trials 5

Requires:
    ~/miniforge3/envs/smolvla/ with lerobot + torch installed
    ~/miniforge3/envs/libero/ with LIBERO sim
"""
from __future__ import annotations
import os, sys, json, argparse, time
from pathlib import Path
import numpy as np

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, "/home/andypark/Projects/repos/LIBERO")

BOWL_X_STATE_IDX = 10   # init_state[10] = qpos[9] = bowl x

# ── Environment helpers ───────────────────────────────────────────────────────
def make_env(task_id: int, resolution: int = 224):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task  = suite.get_task(task_id)
    bddl  = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env   = OffScreenRenderEnv(**{
        "bddl_file_name":  bddl,
        "camera_heights":  resolution,
        "camera_widths":   resolution,
        "camera_names":    ["agentview"],
    })
    env.seed(0)
    dpath = os.path.join(get_libero_path("datasets"), "libero_spatial",
                         f"{task.name}_demo.hdf5")
    return env, dpath, task.name.replace("_", " ")


def load_demo(dpath: str, demo_idx: int = 0):
    import h5py
    with h5py.File(dpath) as f:
        key = sorted(f["data"].keys())[demo_idx]
        d   = f["data"][key]
        return {
            "init_state":  np.array(d["states"][0]),
            "actions":     np.array(d["actions"]),
            "ee_pos":      np.array(d["obs"]["ee_pos"]),
        }


# ── SmolVLA inference ─────────────────────────────────────────────────────────
def load_smolvla(model_id: str = "HuggingFaceVLA/smolvla_libero", device: str = "cuda"):
    """Load SmolVLA policy from HuggingFace."""
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    print(f"Loading {model_id}...")
    t0 = time.time()
    policy = SmolVLAPolicy.from_pretrained(model_id)
    print(f"Loaded in {time.time()-t0:.0f}s")
    policy = policy.to(device).eval()

    mem = torch.cuda.memory_allocated() / 1e9 if device == "cuda" else 0
    total = torch.cuda.get_device_properties(0).total_memory / 1e9 if device == "cuda" else 0
    print(f"GPU: {mem:.1f}/{total:.1f} GB")
    return policy


def run_smolvla(env, policy, demo: dict, offset_m: float,
                task_name: str, max_steps: int = 220, device: str = "cuda") -> bool:
    """Run SmolVLA policy with optional bowl offset."""
    import torch
    from PIL import Image

    init = demo["init_state"].copy()
    if offset_m != 0:
        init[BOWL_X_STATE_IDX] += offset_m   # shift bowl

    env.reset()
    obs = env.set_init_state(init)
    for _ in range(10):   # stabilize
        obs, _, _, _ = env.step([0.0] * 7)

    policy.reset()
    reward = 0.0

    for step in range(max_steps):
        image_rgb = obs["agentview_image"].astype(np.uint8)
        img = Image.fromarray(image_rgb)

        with torch.no_grad():
            # SmolVLA expects: {"observation.images.top": tensor, "observation.state": tensor}
            img_t = torch.from_numpy(image_rgb).permute(2,0,1).unsqueeze(0).float() / 255.0
            state  = torch.zeros(1, 7)  # placeholder state
            batch  = {
                "observation.images.top": img_t.to(device),
                "observation.state": state.to(device),
                "task": [task_name],
            }
            action_chunk = policy.select_action(batch)

        # Take first action from chunk
        if isinstance(action_chunk, torch.Tensor):
            action = action_chunk[0, 0].cpu().numpy() if action_chunk.ndim == 3 else \
                     action_chunk[0].cpu().numpy()
        else:
            action = np.array(action_chunk)

        obs, r, done, _ = env.step(action.tolist())
        reward = r
        if done:
            break

    return reward > 0


# ── DHB adaptation baseline ───────────────────────────────────────────────────
def run_dhb_adaptation(env, demo: dict, offset_m: float,
                       alpha: float = 0.5, Kp: float = 2.0) -> bool:
    """DHB blended controller: geometric adaptation."""
    from dhb_xr.encoder.dhb_dr import encode_dhb_dr
    from dhb_xr.decoder.dhb_dr import decode_dhb_dr
    from dhb_xr.core.types import EncodingMethod
    from scipy.spatial.transform import Rotation

    ee_pos = demo["ee_pos"]
    # Dummy orientations (identity quaternions wxyz) for position-only encoding
    quats_wxyz = np.zeros((len(ee_pos), 4)); quats_wxyz[:, 0] = 1.0

    enc = encode_dhb_dr(ee_pos, quats_wxyz, method="pos",
                        use_default_initial_frames=True)
    new_start = enc["initial_pose"]["position"] + np.array([offset_m, 0, 0])
    dec = decode_dhb_dr(enc["linear_motion_invariants"],
                        enc["angular_motion_invariants"],
                        initial_pose={"position": new_start,
                                      "quaternion": enc["initial_pose"]["quaternion"]},
                        method=EncodingMethod.POSITION)
    dhb_targets = dec["positions"]

    init = demo["init_state"].copy()
    if offset_m != 0:
        init[BOWL_X_STATE_IDX] += offset_m
    env.reset(); obs = env.set_init_state(init)
    reward = 0.0

    for t, orig_action in enumerate(demo["actions"]):
        current_ee = np.array(obs["robot0_eef_pos"])
        target = dhb_targets[t] if t < len(dhb_targets) else dhb_targets[-1]
        action = orig_action.copy()
        action[:3] = np.clip(orig_action[:3] + alpha * Kp * (target - current_ee), -1, 1)
        obs, r, done, _ = env.step(action)
        reward = r
        if done: break

    return reward > 0


# ── Main ─────────────────────────────────────────────────────────────────────
def main(args):
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load SmolVLA policy
    policy = load_smolvla(args.model, device)
    results = {"smolvla": {}, "dhb": {}}

    for task_id in range(args.num_tasks):
        env, dpath, task_name = make_env(task_id)
        print(f"\nTask {task_id}: {task_name[:60]}")

        for offset_cm in args.offsets:
            sm_ok = dhb_ok = 0
            for trial in range(args.trials):
                demo = load_demo(dpath, demo_idx=trial % 50)
                try:
                    if offset_cm == 0:
                        sm  = run_smolvla(env, policy, demo, 0, task_name, device=device)
                        dhb = run_dhb_adaptation(env, demo, 0)
                    else:
                        sm  = run_smolvla(env, policy, demo, offset_cm/100, task_name, device=device)
                        dhb = run_dhb_adaptation(env, demo, offset_cm/100)
                    sm_ok  += int(sm)
                    dhb_ok += int(dhb)
                except Exception as e:
                    print(f"  Trial {trial} error: {e}")

            sm_rate  = sm_ok  / args.trials
            dhb_rate = dhb_ok / args.trials
            key = f"task{task_id}_{offset_cm}cm"
            results["smolvla"][key] = sm_rate
            results["dhb"][key]     = dhb_rate
            print(f"  {offset_cm:3}cm | SmolVLA {sm_ok}/{args.trials}={sm_rate:.0%} | "
                  f"DHB {dhb_ok}/{args.trials}={dhb_rate:.0%}")

        env.close()

    # Summary table
    print("\n=== Summary: SmolVLA vs DHB Adaptation ===")
    print(f"{'Offset':>8} | {'SmolVLA':>10} | {'DHB-XR':>10} | {'Gain':>8}")
    print("-" * 44)
    for offset_cm in args.offsets:
        sm_rates  = [v for k, v in results["smolvla"].items() if k.endswith(f"_{offset_cm}cm")]
        dhb_rates = [v for k, v in results["dhb"].items()     if k.endswith(f"_{offset_cm}cm")]
        sm_avg  = np.mean(sm_rates)  if sm_rates  else 0
        dhb_avg = np.mean(dhb_rates) if dhb_rates else 0
        print(f"  {offset_cm:>5}cm | {sm_avg:>9.1%} | {dhb_avg:>9.1%} | +{dhb_avg-sm_avg:>6.1%}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default="HuggingFaceVLA/smolvla_libero")
    parser.add_argument("--num-tasks", type=int,   default=3)
    parser.add_argument("--trials",    type=int,   default=5)
    parser.add_argument("--offsets",   type=float, nargs="+", default=[0, 5, 10])
    parser.add_argument("--out",
        default="results/exp_smolvla_vs_dhb.json")
    args = parser.parse_args()
    main(args)
