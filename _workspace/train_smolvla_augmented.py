#!/usr/bin/env python3
"""
Step 2.4: Fine-tune SmolVLA on original / DHB-augmented / naive-augmented data.

Converts our npz episode files to a LeRobot dataset, then trains SmolVLA
with LoRA on each variant.

Usage:
    ~/miniforge3/envs/smolvla/bin/python _workspace/train_smolvla_augmented.py \
        --mode original --steps 5000

Three runs:
    --mode original   (50 demos baseline)
    --mode dhb        (50 orig + 450 DHB-augmented)
    --mode naive      (50 orig + 450 naive-augmented)
"""
from __future__ import annotations
import os, sys, argparse, json, shutil, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def build_lerobot_dataset(mode: str, task_id: int = 0, root: str = None):
    """Convert our npz episodes to a LeRobot v3.0 dataset."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import torch
    from PIL import Image

    repo_id = f"local/libero_spatial_t{task_id}_{mode}"
    if root is None:
        root = f"_workspace/lerobot_v3_{mode}_task{task_id}"

    # Clean previous
    root_path = Path(root)
    if root_path.exists():
        shutil.rmtree(root_path)

    # Load metadata from npz dataset
    npz_dir = Path(f"_workspace/lerobot_dataset_{mode}_task{task_id}")
    if mode in ("dhb", "naive"):
        # Include original demos + augmented
        orig_dir = Path(f"_workspace/lerobot_dataset_original_task{task_id}")
        all_dirs = [orig_dir, npz_dir]
    else:
        all_dirs = [npz_dir]

    # Collect all episode files
    episode_files = []
    for d in all_dirs:
        eps = sorted(d.glob("episode_*.npz"))
        episode_files.extend(eps)

    print(f"Building LeRobot dataset for mode={mode}")
    print(f"  Episodes: {len(episode_files)}")

    # Read first episode to get shapes
    sample = np.load(episode_files[0])
    img_shape = sample["images"].shape[1:]  # (H, W, 3)
    state_dim = sample["states"].shape[1]
    action_dim = sample["actions"].shape[1]
    task_name = str(sample["task"])

    features = {
        "observation.images.image": {
            "dtype": "image",
            "shape": (3, img_shape[0], img_shape[1]),
            "names": ["channels", "height", "width"],
        },
        "observation.images.image2": {
            "dtype": "image",
            "shape": (3, img_shape[0], img_shape[1]),
            "names": ["channels", "height", "width"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["action"],
        },
        "task": {
            "dtype": "string",
            "shape": (1,),
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=10,
        features=features,
        root=root,
        robot_type="panda",
        use_videos=False,  # Store as images, not videos
    )

    for ep_idx, ep_file in enumerate(episode_files):
        data = np.load(ep_file)
        images = data["images"]      # (T, H, W, 3)
        images2 = data["images2"]    # (T, H, W, 3)
        states = data["states"]      # (T, state_dim)
        actions = data["actions"]    # (T, action_dim)

        for t in range(len(actions)):
            frame = {
                "observation.images.image": Image.fromarray(images[t]),
                "observation.images.image2": Image.fromarray(images2[t]),
                "observation.state": torch.from_numpy(states[t]),
                "action": torch.from_numpy(actions[t]),
                "task": task_name,
            }
            dataset.add_frame(frame)

        dataset.save_episode()

        if (ep_idx + 1) % 50 == 0:
            print(f"  Added {ep_idx + 1}/{len(episode_files)} episodes")

    dataset.finalize()
    print(f"  Dataset finalized: {root}")
    print(f"  Total frames: {dataset.num_frames}")
    return root, repo_id


def train(mode: str, dataset_root: str, steps: int = 5000, batch_size: int = 4):
    """Fine-tune SmolVLA using lerobot-train CLI."""
    import subprocess

    output_dir = f"_workspace/smolvla_finetuned_{mode}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        f"--policy.path=HuggingFaceVLA/smolvla_libero",
        f"--dataset.repo_id=local/placeholder",
        f"--dataset.root={dataset_root}",
        f"--env.type=libero",
        f"--output_dir={output_dir}",
        f"--batch_size={batch_size}",
        f"--steps={steps}",
        f"--eval_freq={steps}",  # Only eval at end
        f"--save_freq={steps}",
        f"--log_freq=100",
        "--peft.method_type=LORA",
        "--peft.r=16",
        "--use_policy_training_preset=True",
    ]

    print(f"\nTraining SmolVLA ({mode})...")
    print(f"  Steps: {steps}")
    print(f"  Output: {output_dir}")
    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["original", "dhb", "naive"], required=True)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--skip-dataset", action="store_true",
                        help="Skip dataset building (use existing)")
    args = parser.parse_args()

    if not args.skip_dataset:
        dataset_root, repo_id = build_lerobot_dataset(args.mode, args.task)
    else:
        dataset_root = f"_workspace/lerobot_v3_{args.mode}_task{args.task}"

    train(args.mode, dataset_root, args.steps, args.batch_size)


if __name__ == "__main__":
    main()
