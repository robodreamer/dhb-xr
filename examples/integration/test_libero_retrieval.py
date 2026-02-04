#!/usr/bin/env python
"""Test motion retrieval with real LIBERO-Spatial dataset.

This demonstrates:
1. Loading LIBERO episodes using LiberoAdapter
2. Encoding trajectories with DHB-DR
3. Building a motion database for retrieval
4. Querying similar motions across different tasks

Usage:
    pixi run python examples/integration/test_libero_retrieval.py
"""
import os
import sys
from pathlib import Path

# Add src to path when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dhb_xr.integration.vla.libero import LiberoAdapter
from dhb_xr.database.motion_db import MotionDatabase

LIBERO_DIR = Path(os.environ.get("LIBERO_DATA_DIR", "/home/andypark/Projects/data/libero/libero_spatial"))

print("=" * 60)
print("LIBERO-Spatial Motion Retrieval Demo")
print("=" * 60)
print()

# Load episodes from multiple tasks
adapter = LiberoAdapter()
all_episodes = []
task_names = []

hdf5_files = sorted(LIBERO_DIR.glob("*.hdf5"))[:5]  # First 5 tasks
print(f"Loading episodes from {len(hdf5_files)} tasks...\n")

for hdf5_file in hdf5_files:
    task_name = hdf5_file.stem.replace("_demo", "")
    episodes = list(adapter.load_dataset(str(hdf5_file)))[:10]  # 10 episodes per task

    for ep in episodes:
        ep["metadata"]["task_name"] = task_name
        ep["metadata"]["file"] = hdf5_file.name

    all_episodes.extend(episodes)
    task_names.append(task_name)
    print(f"  {task_name[:60]}...")
    print(f"    Loaded {len(episodes)} episodes")

print(f"\nTotal: {len(all_episodes)} episodes from {len(task_names)} tasks")
print()

# Build motion database (encoding happens inside db.add())
print("Building motion database...")
db = MotionDatabase(dhb_method="double_reflection")

for i, ep in enumerate(all_episodes):
    db.add(
        positions=ep["positions"],
        quaternions=ep["quaternions"],
        metadata={**ep["metadata"], "idx": i},
    )

print(f"  Database contains {len(db.invariants_list)} trajectories")
print()

# Query similar motions
print("=" * 60)
print("Motion Retrieval Results")
print("=" * 60)
print()

# Use first episode from each task as query
for task_idx in range(min(3, len(task_names))):
    query_idx = task_idx * 10  # First episode of each task
    query_ep = all_episodes[query_idx]
    query_task = query_ep["metadata"]["task_name"]

    print(f"Query: {query_task[:50]}...")
    print(f"  Demo: {query_ep['metadata']['demo_id']}, Frames: {len(query_ep['positions'])}")

    # Find top-6 similar motions (including self)
    # Use DTW for variable-length trajectories
    results = db.retrieve(
        query_positions=query_ep["positions"],
        query_quaternions=query_ep["quaternions"],
        k=6,
        use_dtw=True,
    )

    print(f"\n  Top-5 Similar Motions:")
    for j, (invariants, meta, distance) in enumerate(results[1:6]):  # Skip self
        match_task = meta.get("task_name", "unknown")
        same_task = "SAME" if match_task == query_task else "DIFF"

        print(f"    {j+1}. [{same_task}] dist={distance:.4f}")
        print(f"       Task: {match_task[:50]}...")
        print(f"       Demo: {meta.get('demo_id', 'unknown')}")

    print()

print("=" * 60)
print("SUCCESS: Motion retrieval working with LIBERO data!")
print("=" * 60)
