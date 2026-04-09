"""LIBERO simulation with image/video capture for paper figures.

Runs naive replay vs DHB-XR trajectory adaptation at multiple perturbation levels.
Saves:
  - output/videos/  — MP4 comparison videos (naive vs DHB, side-by-side)
  - output/frames/  — Key frames: initial, mid-task, success/failure for each condition

Usage:
    ~/miniforge3/envs/libero/bin/python examples/integration/libero_sim_with_media.py \\
        --perturbations 0 5 10 15 \\
        --task-idx 0 \\
        --n-trials 3

Outputs (per perturbation level):
    output/videos/libero_naive_Xcm_v{N}.mp4
    output/videos/libero_dhb_Xcm_v{N}.mp4
    output/videos/libero_comparison_Xcm.mp4  (side-by-side)
    output/frames/libero_{condition}_{Xcm}_frame{t}.png
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# Paths
_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path("/home/andypark/Projects/repos/LIBERO")))
sys.path.insert(0, str(Path("/home/andypark/Projects/repos/LIBERO-PRO")))

from dhb_xr.encoder.dhb_dr import encode_dhb_dr
from dhb_xr.decoder.dhb_dr import decode_dhb_dr

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

import h5py
import glob
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv


def run_episode(
    env: OffScreenRenderEnv,
    actions: np.ndarray,
    init_state: np.ndarray,
    capture_frames: bool = True,
    every_n: int = 5,
) -> dict:
    """Run one episode and capture frames."""
    env.reset()
    env.set_init_state(init_state)

    frames, rewards = [], []
    max_reward = 0.0

    for i, action in enumerate(actions):
        obs, reward, done, info = env.step(action)
        max_reward = max(max_reward, reward)
        rewards.append(float(reward))

        if capture_frames and (i % every_n == 0 or i == 0 or done):
            # Render agentview RGB (128x128x3)
            try:
                frame = obs.get("agentview_rgb", None)
                if frame is None:
                    frame = env.sim.render(
                        camera_name="agentview", height=256, width=256, depth=False
                    )
                    frame = frame[::-1]  # flip y axis
                frames.append(frame.copy())
            except Exception:
                pass

        if done:
            break

    success = max_reward > 0
    return {"success": success, "max_reward": max_reward, "frames": frames, "rewards": rewards}


def make_comparison_video(
    frames_naive: list,
    frames_dhb: list,
    output_path: Path,
    fps: int = 10,
    label_naive: str = "Naive Replay",
    label_dhb: str = "DHB-XR Adaptation",
) -> None:
    """Create side-by-side comparison video."""
    if not HAS_IMAGEIO or not frames_naive or not frames_dhb:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Pad shorter sequence
    max_len = max(len(frames_naive), len(frames_dhb))
    if len(frames_naive) < max_len:
        frames_naive = frames_naive + [frames_naive[-1]] * (max_len - len(frames_naive))
    if len(frames_dhb) < max_len:
        frames_dhb = frames_dhb + [frames_dhb[-1]] * (max_len - len(frames_dhb))

    combined = []
    for fn, fd in zip(frames_naive, frames_dhb):
        h = max(fn.shape[0], fd.shape[0])
        fn = np.pad(fn, ((0, h-fn.shape[0]), (0, 0), (0, 0)))
        fd = np.pad(fd, ((0, h-fd.shape[0]), (0, 0), (0, 0)))
        # Add divider line
        div = np.ones((h, 4, 3), dtype=np.uint8) * 128
        row = np.concatenate([fn, div, fd], axis=1)
        combined.append(row.astype(np.uint8))

    imageio.mimwrite(str(output_path), combined, fps=fps, codec="libx264")


def save_key_frames(
    frames: list,
    output_dir: Path,
    prefix: str,
    indices: list | None = None,
) -> None:
    """Save key frames as PNG images."""
    if not frames:
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    if indices is None:
        # Save first, middle, and last frames
        idxs = [0, len(frames) // 2, len(frames) - 1]
    else:
        idxs = indices

    for i, idx in enumerate(idxs):
        if 0 <= idx < len(frames):
            frame = frames[idx]
            path = output_dir / f"{prefix}_frame{i:02d}.png"
            if HAS_PIL:
                Image.fromarray(frame.astype(np.uint8)).save(str(path))
            elif HAS_IMAGEIO:
                imageio.imwrite(str(path), frame.astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description="LIBERO simulation with media capture")
    parser.add_argument("--perturbations", nargs="+", type=float, default=[0, 5, 10, 15],
                        help="Perturbation magnitudes in cm")
    parser.add_argument("--task-idx", type=int, default=0, help="LIBERO task index")
    parser.add_argument("--n-trials", type=int, default=3, help="Trials per condition")
    parser.add_argument("--output-dir", type=str,
                        default=str(_ROOT / "output"),
                        help="Output directory for videos and frames")
    parser.add_argument("--no-video", action="store_true", help="Skip video rendering")
    parser.add_argument("--fps", type=int, default=10, help="Video FPS")
    args = parser.parse_args()

    mags = [m / 100 for m in args.perturbations]  # cm → m
    output_dir = Path(args.output_dir)
    video_dir = output_dir / "videos"
    frame_dir = output_dir / "frames"

    # Load demo
    demo_files = sorted(glob.glob(str(_ROOT / "../../data/libero/libero_spatial/*.hdf5")))
    if not demo_files:
        demo_files = sorted(glob.glob("/home/andypark/Projects/data/libero/**/*.hdf5", recursive=True))
    demo_file = demo_files[args.task_idx % len(demo_files)]
    task_name = Path(demo_file).stem.replace("_demo", "")

    print(f"Task: {task_name}")
    print(f"Perturbations: {[int(m*100) for m in mags]} cm")
    print(f"Trials: {args.n_trials}")
    print(f"Output: {output_dir}")

    with h5py.File(demo_file) as f:
        dk = sorted(k for k in f["data"].keys() if "demo" in k)[0]
        pos = np.array(f["data"][dk]["obs"]["ee_pos"])
        euler = np.array(f["data"][dk]["obs"]["ee_ori"])
        xyzw = Rotation.from_euler("xyz", euler).as_quat()
        q_wxyz = np.column_stack([xyzw[:,3], xyzw[:,0], xyzw[:,1], xyzw[:,2]])
        actions = np.array(f["data"][dk]["actions"])
        init_state = np.array(f["data"][dk]["states"][0])

    enc = encode_dhb_dr(pos, q_wxyz)

    # Set up environment
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    bddl = suite.get_task_bddl_file_path(args.task_idx % suite.get_num_tasks())
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)

    all_results = {}

    for mag in mags:
        mag_cm = int(mag * 100)
        print(f"\n--- {mag_cm} cm perturbation ---")
        offset = np.array([1, 1, 0]) / np.sqrt(2) * mag

        naive_results, dhb_results = [], []
        best_naive_frames, best_dhb_frames = [], []
        best_naive_success, best_dhb_success = False, False

        for trial in range(args.n_trials):
            # Naive replay
            r_naive = run_episode(env, actions, init_state,
                                  capture_frames=not args.no_video)
            naive_results.append(r_naive["success"])
            if r_naive["success"] or not best_naive_frames:
                best_naive_frames = r_naive["frames"]
                best_naive_success = r_naive["success"]

            # DHB adaptation
            new_ip = copy.deepcopy(enc["initial_pose"])
            new_ip["position"] = pos[0] + offset
            dec = decode_dhb_dr(enc["linear_motion_invariants"],
                                enc["angular_motion_invariants"], new_ip)
            apts = actions.copy()
            for i in range(min(len(dec["positions"])-1, len(apts))):
                apts[i, :3] = dec["positions"][i+1] - dec["positions"][i]
            r_dhb = run_episode(env, apts, init_state,
                                capture_frames=not args.no_video)
            dhb_results.append(r_dhb["success"])
            if r_dhb["success"] or not best_dhb_frames:
                best_dhb_frames = r_dhb["frames"]
                best_dhb_success = r_dhb["success"]

            status_n = "✓" if r_naive["success"] else "✗"
            status_d = "✓" if r_dhb["success"] else "✗"
            print(f"  Trial {trial+1}: naive={status_n}  DHB={status_d}")

        naive_rate = np.mean(naive_results)
        dhb_rate = np.mean(dhb_results)
        print(f"  Success: naive={naive_rate:.0%}  DHB={dhb_rate:.0%}")

        all_results[f"{mag_cm}cm"] = {
            "naive_pct": int(round(naive_rate * 100)),
            "dhb_pct": int(round(dhb_rate * 100)),
            "n": args.n_trials,
        }

        # Save media
        if not args.no_video and HAS_IMAGEIO:
            # Individual videos (best trial)
            naive_label = f"Naive Replay ({mag_cm}cm) — {'SUCCESS' if best_naive_success else 'FAIL'}"
            dhb_label = f"DHB-XR Adapt ({mag_cm}cm) — {'SUCCESS' if best_dhb_success else 'FAIL'}"

            naive_vid = video_dir / f"{task_name}_naive_{mag_cm}cm.mp4"
            dhb_vid = video_dir / f"{task_name}_dhb_{mag_cm}cm.mp4"
            comp_vid = video_dir / f"{task_name}_comparison_{mag_cm}cm.mp4"

            if best_naive_frames:
                video_dir.mkdir(parents=True, exist_ok=True)
                imageio.mimwrite(str(naive_vid), [f.astype(np.uint8) for f in best_naive_frames], fps=args.fps)
                print(f"  Saved: {naive_vid.name}")
            if best_dhb_frames:
                imageio.mimwrite(str(dhb_vid), [f.astype(np.uint8) for f in best_dhb_frames], fps=args.fps)
                print(f"  Saved: {dhb_vid.name}")

            # Side-by-side comparison
            make_comparison_video(best_naive_frames, best_dhb_frames, comp_vid,
                                  fps=args.fps, label_naive=naive_label, label_dhb=dhb_label)
            if comp_vid.exists():
                print(f"  Saved: {comp_vid.name}")

        # Save key frames for paper figures
        save_key_frames(best_naive_frames, frame_dir,
                        f"{task_name}_naive_{mag_cm}cm")
        save_key_frames(best_dhb_frames, frame_dir,
                        f"{task_name}_dhb_{mag_cm}cm")

    env.close()

    # Write results
    results_path = _ROOT / "results" / "exp1_sim_success_rates.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== FINAL RESULTS ===")
    print(f"{'':25}" + "".join(f"{int(m*100):>6d}cm" for m in mags))
    for meth, lbl in [("naive_pct", "Naive replay"), ("dhb_pct", "DHB-XR adapt")]:
        row = f"{lbl:<25}" + "".join(f"{all_results[f'{int(m*100)}cm'][meth]:>5d}%" for m in mags)
        print(row)
    print(f"\nResults: {results_path}")
    if not args.no_video:
        print(f"Videos:  {video_dir}")
        print(f"Frames:  {frame_dir}")


if __name__ == "__main__":
    main()
