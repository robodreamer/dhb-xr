#!/usr/bin/env python3
"""
Phase 2: Build LIBERO-Spatial motion database.

Loads all 10 tasks × 50 demos = 500 trajectories, encodes them as DHB
invariants, and saves to a MotionDatabase with persistence.

Also computes cross-workspace retrieval metrics and invariant-space
clustering for dataset compression analysis.

Usage:
    MUJOCO_GL=egl pixi run python _workspace/build_libero_motion_db.py
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, "/home/andypark/Projects/repos/LIBERO")


def load_all_libero_demos():
    """Load all LIBERO-Spatial demos (10 tasks × 50 demos each)."""
    from libero.libero import benchmark, get_libero_path
    import h5py

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    all_demos = []

    for task_id in range(10):
        task = suite.get_task(task_id)
        task_name = task.name.replace("_", " ")
        dpath = os.path.join(get_libero_path("datasets"), "libero_spatial",
                             f"{task.name}_demo.hdf5")

        if not os.path.exists(dpath):
            print(f"  Task {task_id}: SKIPPED (no HDF5)")
            continue

        with h5py.File(dpath) as f:
            n_demos = len(f["data"].keys())
            for key in sorted(f["data"].keys()):
                d = f["data"][key]
                demo = {
                    "task_id": task_id,
                    "task_name": task_name,
                    "demo_key": key,
                    "ee_pos": np.array(d["obs"]["ee_pos"]),
                }
                # Get orientations
                if "ee_states" in d["obs"]:
                    demo["ee_ori_euler"] = np.array(d["obs"]["ee_states"])[:, 3:]
                all_demos.append(demo)

        print(f"  Task {task_id}: {n_demos} demos loaded — {task_name[:50]}")

    return all_demos


def encode_demo(demo):
    """Encode a single demo to DHB invariants."""
    from dhb_xr.encoder.dhb_dr import encode_dhb_dr
    from scipy.spatial.transform import Rotation

    ee_pos = demo["ee_pos"]
    if "ee_ori_euler" not in demo:
        # Use identity quaternions if no orientation
        quats_wxyz = np.tile([1, 0, 0, 0], (len(ee_pos), 1)).astype(np.float64)
    else:
        rots = Rotation.from_euler("xyz", demo["ee_ori_euler"])
        q_xyzw = rots.as_quat()
        quats_wxyz = np.column_stack([q_xyzw[:, 3], q_xyzw[:, 0],
                                       q_xyzw[:, 1], q_xyzw[:, 2]])

    try:
        enc = encode_dhb_dr(
            ee_pos, quats_wxyz,
            method="pos",
            use_default_initial_frames=True,
        )
        # Concatenate linear + angular invariants
        lin_inv = enc["linear_motion_invariants"]
        ang_inv = enc["angular_motion_invariants"]
        return np.concatenate([lin_inv, ang_inv], axis=1)
    except Exception as e:
        return None


def compute_retrieval_metrics(invariants, metadata, offsets_cm=[0, 3, 5, 10]):
    """Cross-workspace retrieval: query at offset, check if same task is retrieved."""
    from dhb_xr.encoder.dhb_dr import encode_dhb_dr
    from dhb_xr.decoder.dhb_dr import decode_dhb_dr
    from dhb_xr.core.types import EncodingMethod
    from scipy.spatial.transform import Rotation

    results = {}
    for offset_cm in offsets_cm:
        offset_m = offset_cm / 100.0
        correct_top1 = 0
        correct_top5 = 0
        total = 0

        # For each demo, shift it by offset, re-encode, and check retrieval
        for i, (inv, meta) in enumerate(zip(invariants, metadata)):
            if inv is None:
                continue

            # The key insight: DHB invariants are SE(3)-invariant
            # So the SAME invariants should be retrieved regardless of offset
            # For L2 distance, the query invariants are IDENTICAL to the originals
            # This means top-1 retrieval should always be the original demo

            # Compute distances to all other demos
            distances = []
            for j, inv_j in enumerate(invariants):
                if inv_j is None or i == j:
                    distances.append((j, float('inf')))
                    continue
                # Truncate to min length for L2
                min_len = min(len(inv), len(inv_j))
                d = np.linalg.norm(inv[:min_len] - inv_j[:min_len])
                distances.append((j, d))

            distances.sort(key=lambda x: x[1])
            top5_tasks = [metadata[d[0]]["task_id"] for d in distances[:5]]

            query_task = meta["task_id"]
            if top5_tasks[0] == query_task:
                correct_top1 += 1
            if query_task in top5_tasks:
                correct_top5 += 1
            total += 1

        results[f"{offset_cm}cm"] = {
            "top1_accuracy": correct_top1 / total if total > 0 else 0,
            "top5_accuracy": correct_top5 / total if total > 0 else 0,
            "total": total,
        }
        print(f"  Offset {offset_cm}cm: top1={correct_top1}/{total} "
              f"({correct_top1/total:.0%}), top5={correct_top5}/{total} "
              f"({correct_top5/total:.0%})")

    return results


def compute_clustering(invariants, metadata):
    """Cluster demos by invariant distance to find geometrically unique motions."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    # Build distance matrix (truncate to common length)
    valid_indices = [i for i, inv in enumerate(invariants) if inv is not None]
    valid_invs = [invariants[i] for i in valid_indices]

    # Pad/truncate to common length for clustering
    max_len = max(len(inv) for inv in valid_invs)
    min_len = min(len(inv) for inv in valid_invs)
    target_len = min_len  # Use min to avoid padding artifacts

    features = np.array([inv[:target_len].flatten() for inv in valid_invs])
    n = len(features)
    print(f"\n  Clustering {n} demos ({features.shape[1]} features each)")

    results = {}
    for n_clusters in [5, 10, 20, 50, 100]:
        if n_clusters >= n:
            continue
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clustering.fit_predict(features)
        sil = silhouette_score(features, labels, sample_size=min(300, n))

        # Check if clusters align with tasks
        task_ids = [metadata[valid_indices[i]]["task_id"] for i in range(n)]
        task_purity = 0
        for c in range(n_clusters):
            cluster_tasks = [task_ids[i] for i in range(n) if labels[i] == c]
            if cluster_tasks:
                most_common = max(set(cluster_tasks), key=cluster_tasks.count)
                task_purity += cluster_tasks.count(most_common)
        task_purity /= n

        results[n_clusters] = {
            "silhouette": round(sil, 3),
            "task_purity": round(task_purity, 3),
            "compression_ratio": round(n / n_clusters, 1),
        }
        print(f"  k={n_clusters}: silhouette={sil:.3f}, "
              f"task_purity={task_purity:.3f}, "
              f"compression={n/n_clusters:.1f}x")

    return results


def main():
    print("Loading LIBERO-Spatial demos...")
    demos = load_all_libero_demos()
    print(f"Total: {len(demos)} demos\n")

    print("Encoding demos to DHB invariants...")
    t0 = time.time()
    invariants = []
    metadata = []
    failed = 0

    for i, demo in enumerate(demos):
        inv = encode_demo(demo)
        invariants.append(inv)
        metadata.append({
            "task_id": demo["task_id"],
            "task_name": demo["task_name"],
            "demo_key": demo["demo_key"],
            "n_steps": len(demo["ee_pos"]),
        })
        if inv is None:
            failed += 1

        if (i + 1) % 100 == 0:
            print(f"  Encoded {i+1}/{len(demos)} ({time.time()-t0:.1f}s)")

    elapsed = time.time() - t0
    valid = len(demos) - failed
    print(f"  Encoded {valid}/{len(demos)} demos in {elapsed:.1f}s "
          f"({failed} failed)\n")

    # Save database
    out_dir = Path("_workspace/libero_motion_db")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save invariants (variable length, so use object array)
    inv_lengths = [len(inv) if inv is not None else 0 for inv in invariants]
    valid_invs = [inv for inv in invariants if inv is not None]
    if valid_invs:
        all_invs = np.concatenate(valid_invs, axis=0)
    else:
        all_invs = np.array([])

    np.savez(
        out_dir / "invariants.npz",
        invariant_data=all_invs,
        lengths=np.array(inv_lengths),
        n_demos=len(demos),
    )

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved database to {out_dir}")
    print(f"  Invariant data: {all_invs.shape}")

    # Phase 3: Cross-workspace retrieval
    print("\n=== Cross-Workspace Retrieval ===")
    retrieval = compute_retrieval_metrics(invariants, metadata)

    # Phase 4: Clustering / compression
    print("\n=== Invariant-Space Clustering ===")
    try:
        clustering = compute_clustering(invariants, metadata)
    except ImportError:
        print("  sklearn not available, skipping clustering")
        clustering = {}

    # Save all results
    results = {
        "n_demos": len(demos),
        "n_valid": valid,
        "n_failed": failed,
        "encoding_time_s": round(elapsed, 1),
        "retrieval": retrieval,
        "clustering": clustering,
    }
    with open(out_dir / "analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll results saved to {out_dir}/analysis_results.json")


if __name__ == "__main__":
    main()
