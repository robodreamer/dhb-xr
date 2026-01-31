#!/usr/bin/env python3
"""
Download UMI-XDOF sample episodes from S3 for the tutorial notebook.

Syncs 5 episodes per task from s3://hmnd-partner-ingest/xdof/date=2025_05_23/
excluding video files (.mp4) to keep download size small.

Usage:
    python scripts/download_xdof_samples.py [--out-dir DIR] [--episodes-per-task N]

Requires: aws CLI configured (e.g. aws sso login).
"""

import argparse
import subprocess
import sys
from pathlib import Path

S3_BUCKET = "s3://hmnd-partner-ingest/xdof/date=2025_05_23"
TASKS = [
    "pick_place_beverage",
    "pick_place_candy_bag",
    "pick_place_chips_bag",
    "pick_place_cleaning_sponge",
    "pick_place_dish_detergent",
    "pick_place_paper_towel",
    "pick_place_shaving_razor",
    "pick_place_soup_can",
    "pick_place_toothpaste",
    "pick_place_yoghurt",
]
DEFAULT_OUT_DIR = "/home/andypark/Projects/data/xdof-umi/xdof_samples"
EPISODES_PER_TASK = 5


def list_episode_prefixes(task: str) -> list[str]:
    """List episode folder names (prefixes) for a task. Returns first N available."""
    result = subprocess.run(
        ["aws", "s3", "ls", f"{S3_BUCKET}/{task}/"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Warning: failed to list {task}: {result.stderr}", file=sys.stderr)
        return []
    # Lines look like "                           PRE episode_xxx/"
    prefixes = []
    for line in result.stdout.strip().splitlines():
        if "PRE" in line:
            name = line.split()[-1].rstrip("/")
            if name.startswith("episode_"):
                prefixes.append(name)
    return prefixes


def sync_episode(task: str, episode_prefix: str, out_dir: Path) -> bool:
    """Sync one episode (exclude .mp4). Returns True on success."""
    s3_prefix = f"{S3_BUCKET}/{task}/{episode_prefix}/"
    local_dir = out_dir / task / episode_prefix
    local_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "aws", "s3", "sync", s3_prefix, str(local_dir),
            "--exclude", "*.mp4",
            "--exclude", "*annotation*",
            "--exclude", "write_complete.flag",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Warning: sync failed {s3_prefix}: {result.stderr}", file=sys.stderr)
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Download UMI-XDOF samples from S3")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(DEFAULT_OUT_DIR),
        help="Output directory for xdof_samples",
    )
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=EPISODES_PER_TASK,
        help="Number of episodes to download per task",
    )
    args = parser.parse_args()
    out_dir = args.out_dir
    n_per_task = args.episodes_per_task

    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for task in TASKS:
        prefixes = list_episode_prefixes(task)
        to_sync = prefixes[:n_per_task]
        if not to_sync:
            print(f"  {task}: no episodes found, skip")
            continue
        print(f"  {task}: syncing {len(to_sync)} episodes ...")
        for ep in to_sync:
            if sync_episode(task, ep, out_dir):
                total += 1
    print(f"Done. Downloaded {total} episodes to {out_dir}")


if __name__ == "__main__":
    main()
