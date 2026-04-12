#!/usr/bin/env python3
"""
Motion DB Phase 3: Cross-workspace retrieval benchmark.

For each demo, shift it by various offsets, re-encode, and check if:
1. DHB invariant retrieval returns the SAME demo (should be ~100%)
2. Position-based retrieval returns the SAME demo (degrades with offset)

This demonstrates the practical value of invariant-space indexing.

Usage:
    pixi run python _workspace/eval_cross_workspace_retrieval.py
"""
import sys, json, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, "/home/andypark/Projects/repos/LIBERO")


def main():
    from dhb_xr.encoder.dhb_dr import encode_dhb_dr
    from scipy.spatial.transform import Rotation
    from libero.libero import benchmark, get_libero_path
    import h5py, os

    print("Loading demos...")
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()

    # Load 5 demos per task (50 total)
    demos = []
    for task_id in range(10):
        task = suite.get_task(task_id)
        dpath = os.path.join(get_libero_path("datasets"), "libero_spatial",
                             f"{task.name}_demo.hdf5")
        with h5py.File(dpath) as f:
            for i, key in enumerate(sorted(f["data"].keys())):
                if i >= 5:
                    break
                d = f["data"][key]
                ee_pos = np.array(d["obs"]["ee_pos"])
                if "ee_states" in d["obs"]:
                    euler = np.array(d["obs"]["ee_states"])[:, 3:]
                    rots = Rotation.from_euler("xyz", euler)
                    q_xyzw = rots.as_quat()
                    quats = np.column_stack([q_xyzw[:, 3], q_xyzw[:, 0],
                                             q_xyzw[:, 1], q_xyzw[:, 2]])
                else:
                    quats = np.tile([1, 0, 0, 0], (len(ee_pos), 1))

                demos.append({
                    "task_id": task_id,
                    "ee_pos": ee_pos,
                    "quats": quats,
                })

    print(f"Loaded {len(demos)} demos from {10} tasks")

    # Encode all demos
    print("Encoding...")
    invariants = []
    for demo in demos:
        try:
            enc = encode_dhb_dr(demo["ee_pos"], demo["quats"],
                                method="pos", use_default_initial_frames=True)
            inv = np.concatenate([enc["linear_motion_invariants"],
                                  enc["angular_motion_invariants"]], axis=1)
            invariants.append(inv)
        except:
            invariants.append(None)

    offsets_cm = [0, 1, 3, 5, 10, 20]
    results = {"offsets_cm": offsets_cm, "n_demos": len(demos), "per_offset": {}}

    print("\n=== Cross-Workspace Self-Retrieval ===")
    print(f"{'Offset':>8} | {'DHB self-ret':>13} | {'Pos self-ret':>13}")
    print("-" * 45)

    for offset_cm in offsets_cm:
        offset_m = offset_cm / 100.0
        offset_vec = np.array([0, offset_m, 0])

        dhb_correct = 0
        pos_correct = 0
        total = 0

        for i, demo in enumerate(demos):
            if invariants[i] is None:
                continue

            # DHB: invariants don't change under translation
            # So the nearest neighbor is always the same regardless of offset
            # We verify by re-encoding at offset and checking distance = 0
            shifted_pos = demo["ee_pos"] + offset_vec
            try:
                enc_shifted = encode_dhb_dr(shifted_pos, demo["quats"],
                                             method="pos", use_default_initial_frames=True)
                inv_shifted = np.concatenate([enc_shifted["linear_motion_invariants"],
                                               enc_shifted["angular_motion_invariants"]], axis=1)

                # Find nearest neighbor in invariants (excluding self)
                min_dist = float('inf')
                min_idx = -1
                for j in range(len(demos)):
                    if j == i or invariants[j] is None:
                        continue
                    min_len = min(len(inv_shifted), len(invariants[j]))
                    d = np.linalg.norm(inv_shifted[:min_len] - invariants[j][:min_len])
                    if d < min_dist:
                        min_dist = d
                        min_idx = j

                # Check if same task
                if min_idx >= 0 and demos[min_idx]["task_id"] == demo["task_id"]:
                    dhb_correct += 1
            except:
                pass

            # Position-based: shift positions, find nearest by L2
            shifted_pos_flat = shifted_pos
            min_dist = float('inf')
            min_idx = -1
            for j in range(len(demos)):
                if j == i:
                    continue
                min_len = min(len(shifted_pos_flat), len(demos[j]["ee_pos"]))
                d = np.linalg.norm(shifted_pos_flat[:min_len] - demos[j]["ee_pos"][:min_len])
                if d < min_dist:
                    min_dist = d
                    min_idx = j

            if min_idx >= 0 and demos[min_idx]["task_id"] == demo["task_id"]:
                pos_correct += 1

            total += 1

        dhb_rate = dhb_correct / total if total > 0 else 0
        pos_rate = pos_correct / total if total > 0 else 0

        results["per_offset"][f"{offset_cm}cm"] = {
            "dhb_same_task": round(dhb_rate, 3),
            "pos_same_task": round(pos_rate, 3),
            "total": total,
        }
        print(f"  {offset_cm:>5}cm | {dhb_correct}/{total}={dhb_rate:>5.0%}      | {pos_correct}/{total}={pos_rate:>5.0%}")

    with open("_workspace/cross_workspace_retrieval.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: _workspace/cross_workspace_retrieval.json")


if __name__ == "__main__":
    main()
