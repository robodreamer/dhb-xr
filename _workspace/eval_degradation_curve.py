#!/usr/bin/env python3
"""
Step 1: SmolVLA Degradation Curve under Spatial Perturbation.

Runs SmolVLA on LIBERO-Spatial task 0 at systematic offsets to quantify
where VLA inference breaks under workspace shifts. This is the paper's
motivating figure: success rate vs offset magnitude.

Usage:
    MUJOCO_GL=egl ~/miniforge3/envs/libero/bin/python \
        _workspace/eval_degradation_curve.py \
        --axis Y --seeds 20

Output:
    _workspace/degradation_curve_results.json
    _workspace/degradation_curve.png
"""
from __future__ import annotations
import os, sys, json, argparse, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, "/home/andypark/Projects/repos/LIBERO")
sys.path.insert(0, "/home/andypark/Projects/repos/dhb_xr_paper/examples/integration")

# Bowl state indices in LIBERO init_state
BOWL_STATE_INDICES = {'X': 10, 'Y': 11, 'Z': 12}

OFFSETS_CM = [0, 1, 2, 3, 5, 7, 10]


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


def load_demo(dpath: str, demo_idx: int = 0):
    import h5py
    with h5py.File(dpath) as f:
        key = sorted(f["data"].keys())[demo_idx]
        d = f["data"][key]
        return {
            "init_state": np.array(d["states"][0]),
            "actions": np.array(d["actions"]),
        }


def quat_to_axis_angle(quat):
    """Convert quaternion (x,y,z,w) to axis-angle (3,). Matches lerobot LiberoProcessorStep."""
    import torch
    q = torch.from_numpy(np.array(quat, dtype=np.float32))
    w = q[3].clamp(-1.0, 1.0)
    den = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0))
    if den > 1e-10:
        angle = 2.0 * torch.acos(w)
        axis = q[:3] / den
        return (axis * angle).numpy()
    return np.zeros(3, dtype=np.float32)


def load_smolvla(model_id: str = "HuggingFaceVLA/smolvla_libero", device: str = "cuda"):
    """Load SmolVLA policy + preprocessor/postprocessor pipelines."""
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    import lerobot.policies.smolvla.processor_smolvla  # noqa: F401
    from lerobot.policies.factory import make_pre_post_processors

    print(f"Loading {model_id}...")
    t0 = time.time()
    policy = SmolVLAPolicy.from_pretrained(model_id)
    print(f"Loaded in {time.time()-t0:.0f}s")
    policy = policy.to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=model_id
    )
    policy._preprocessor = preprocessor
    policy._postprocessor = postprocessor
    return policy


def run_smolvla_trial(env, policy, demo, offset_m, axis, task_name,
                      max_steps=80, device="cuda"):
    """Run one SmolVLA closed-loop trial at given offset. Returns (success, reward)."""
    import torch

    init = demo["init_state"].copy()
    if offset_m != 0:
        bowl_idx = BOWL_STATE_INDICES[axis]
        init[bowl_idx] += offset_m

    env.reset()
    obs = env.set_init_state(init)
    for _ in range(10):
        obs, _, _, _ = env.step([0.0] * 7)

    policy.reset()
    reward = 0.0

    for step in range(max_steps):
        with torch.no_grad():
            agentview = obs["agentview_image"].astype(np.uint8)
            eih = obs.get("robot0_eye_in_hand_image",
                          obs.get("agentview_image")).astype(np.uint8)
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

    return reward > 0, reward


def main():
    parser = argparse.ArgumentParser(description="SmolVLA degradation curve experiment")
    parser.add_argument("--task", type=int, default=0, help="LIBERO-Spatial task ID")
    parser.add_argument("--axis", default="Y", choices=["X", "Y", "Z"],
                        help="Perturbation axis")
    parser.add_argument("--seeds", type=int, default=20,
                        help="Number of trials per offset condition")
    parser.add_argument("--offsets", type=float, nargs="+", default=None,
                        help="Override offset list in cm (default: 0,1,2,3,5,7,10)")
    parser.add_argument("--out", default="_workspace/degradation_curve_results.json")
    args = parser.parse_args()

    offsets_cm = args.offsets if args.offsets else OFFSETS_CM
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Task: {args.task}, Axis: {args.axis}, Seeds: {args.seeds}")
    print(f"Offsets (cm): {offsets_cm}")

    policy = load_smolvla(device=device)
    env, dpath, task_name = make_env(args.task)
    print(f"Task name: {task_name}")

    # Count available demos
    import h5py
    with h5py.File(dpath) as f:
        n_demos = len(f["data"].keys())
    print(f"Available demos: {n_demos}")

    results = {
        "metadata": {
            "task_id": args.task,
            "task_name": task_name,
            "axis": args.axis,
            "seeds_per_offset": args.seeds,
            "offsets_cm": offsets_cm,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "trials": [],
        "summary": {},
    }

    for offset_cm in offsets_cm:
        offset_m = offset_cm / 100.0
        successes = 0
        trial_details = []

        print(f"\n--- Offset: {offset_cm}cm ({args.axis}-axis) ---")
        for seed in range(args.seeds):
            demo_idx = seed % n_demos
            demo = load_demo(dpath, demo_idx)

            t0 = time.time()
            try:
                success, reward = run_smolvla_trial(
                    env, policy, demo, offset_m, args.axis, task_name, device=device
                )
            except Exception as e:
                print(f"  Seed {seed}: ERROR - {e}")
                success, reward = False, 0.0

            elapsed = time.time() - t0
            successes += int(success)

            trial_details.append({
                "offset_cm": offset_cm,
                "seed": seed,
                "demo_idx": demo_idx,
                "success": success,
                "reward": float(reward),
                "time_s": round(elapsed, 1),
            })

            status = "OK" if success else "FAIL"
            print(f"  Seed {seed:2d} (demo {demo_idx:2d}): {status}  "
                  f"r={reward:.2f}  {elapsed:.1f}s  "
                  f"[{successes}/{seed+1}]")

        rate = successes / args.seeds
        results["trials"].extend(trial_details)
        results["summary"][f"{offset_cm}cm"] = {
            "successes": successes,
            "total": args.seeds,
            "rate": rate,
        }
        print(f"  >>> {offset_cm}cm: {successes}/{args.seeds} = {rate:.0%}")

    env.close()

    # Save results
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out}")

    # Print summary table
    print("\n=== Degradation Curve Summary ===")
    print(f"Task {args.task}: {task_name}")
    print(f"{'Offset':>8} | {'Success':>10} | {'Rate':>8}")
    print("-" * 35)
    for offset_cm in offsets_cm:
        s = results["summary"][f"{offset_cm}cm"]
        bar = "#" * int(s["rate"] * 20) + "." * (20 - int(s["rate"] * 20))
        print(f"  {offset_cm:>5}cm | {s['successes']:>3}/{s['total']:<3}     | {s['rate']:>6.0%} [{bar}]")


if __name__ == "__main__":
    main()
