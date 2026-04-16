#!/usr/bin/env python3
"""
Quick motion DB comparison: DHB-DR vs DHB-TI vs magnitude-only vs position.

Lightweight version that avoids OOM by:
1. Processing demos one at a time (not holding all raw data)
2. Encoding immediately and discarding raw arrays
3. Using subsample of 100 demos for DTW (50 takes too long for all)

Usage:
    MUJOCO_GL=egl pixi run python _workspace/quick_motion_db_comparison.py
"""
from __future__ import annotations
import os, sys, json, time, gc
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, "/home/andypark/Projects/repos/LIBERO")

import builtins
_orig_print = builtins.print
def print(*a, **kw):
    kw.setdefault("flush", True)
    _orig_print(*a, **kw)
builtins.print = print


def get_quats_wxyz(ee_pos, ee_ori_euler=None):
    from scipy.spatial.transform import Rotation
    if ee_ori_euler is None:
        return np.tile([1, 0, 0, 0], (len(ee_pos), 1)).astype(np.float64)
    rots = Rotation.from_euler("xyz", ee_ori_euler)
    q = rots.as_quat()
    return np.column_stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]])


def encode_all_strategies(ee_pos, quats, M=64):
    """Encode a single demo with all strategies. Returns dict of encodings."""
    from dhb_xr.encoder.dhb_dr import encode_dhb_dr
    from dhb_xr.encoder.dhb_ti import encode_dhb_dr_ti

    results = {}

    # DHB-DR full (8D, variable length)
    try:
        enc = encode_dhb_dr(ee_pos, quats, method="pos", use_default_initial_frames=True)
        inv = np.concatenate([enc["linear_motion_invariants"],
                              enc["angular_motion_invariants"]], axis=1)
        results["dhb_dr_full"] = inv  # (T-1, 8)
        results["dhb_dr_mag"] = np.concatenate([
            enc["linear_motion_invariants"][:, 0:1],
            enc["angular_motion_invariants"][:, 0:1]], axis=1)  # (T-1, 2)
    except Exception:
        pass

    # DHB-TI full (8D, fixed length M-1)
    try:
        enc = encode_dhb_dr_ti(ee_pos, quats, M=M, progress_kind="translation",
                               method="pos", use_default_initial_frames=True)
        inv = np.concatenate([enc["linear_motion_invariants"],
                              enc["angular_motion_invariants"]], axis=1)
        results["dhb_ti_full"] = inv  # (M-1, 8)
        results["dhb_ti_mag"] = np.concatenate([
            enc["linear_motion_invariants"][:, 0:1],
            enc["angular_motion_invariants"][:, 0:1]], axis=1)  # (M-1, 2)
    except Exception:
        pass

    # Position fixed-length
    old_t = np.linspace(0, 1, len(ee_pos))
    new_t = np.linspace(0, 1, M)
    pos_fixed = np.empty((M, 3))
    for c in range(3):
        pos_fixed[:, c] = np.interp(new_t, old_t, ee_pos[:, c])
    results["position"] = pos_fixed  # (M, 3)

    return results


def main():
    print("=" * 60)
    print("QUICK MOTION DB: DHB-DR vs DHB-TI vs Position")
    print("=" * 60)

    # Load demos and encode immediately (memory-efficient)
    from libero.libero import benchmark, get_libero_path
    import h5py

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    M = 64

    all_encodings = {k: [] for k in ["dhb_dr_full", "dhb_dr_mag",
                                      "dhb_ti_full", "dhb_ti_mag", "position"]}
    task_ids = []

    print("\n[1/5] Loading & encoding 500 demos...")
    t0 = time.time()
    for task_id in range(10):
        task = suite.get_task(task_id)
        dpath = os.path.join(get_libero_path("datasets"), "libero_spatial",
                             f"{task.name}_demo.hdf5")
        if not os.path.exists(dpath):
            continue

        with h5py.File(dpath, "r") as f:
            for key in sorted(f["data"].keys()):
                d = f["data"][key]
                ee_pos = np.array(d["obs"]["ee_pos"])
                ee_ori = np.array(d["obs"]["ee_states"])[:, 3:] if "ee_states" in d["obs"] else None
                quats = get_quats_wxyz(ee_pos, ee_ori)

                encs = encode_all_strategies(ee_pos, quats, M=M)
                for k in all_encodings:
                    all_encodings[k].append(encs.get(k))
                task_ids.append(task_id)

        print(f"  Task {task_id}: done ({len(task_ids)} total)")
        gc.collect()

    print(f"  Encoded {len(task_ids)} demos in {time.time()-t0:.1f}s")
    task_ids = np.array(task_ids)

    # Count valid per strategy
    for k in all_encodings:
        n_valid = sum(1 for e in all_encodings[k] if e is not None)
        shapes = set(e.shape for e in all_encodings[k] if e is not None)
        print(f"  {k}: {n_valid} valid, shapes: {shapes}")

    # ---------- Build flat matrices ----------
    from scipy.spatial.distance import cdist

    def build_matrix(encodings, target_len=None):
        valid_idx = [i for i, e in enumerate(encodings) if e is not None]
        if not valid_idx:
            return np.array([]), valid_idx, np.array([])
        shapes = set(encodings[i].shape for i in valid_idx)
        if len(shapes) == 1:
            mat = np.array([encodings[i].flatten() for i in valid_idx])
        else:
            if target_len is None:
                target_len = int(np.median([encodings[i].shape[0] for i in valid_idx]))
            k = encodings[valid_idx[0]].shape[1]
            mat = np.empty((len(valid_idx), target_len * k))
            for row, i in enumerate(valid_idx):
                enc = encodings[i]
                if enc.shape[0] == target_len:
                    mat[row] = enc.flatten()
                else:
                    old_t = np.linspace(0, 1, enc.shape[0])
                    new_t = np.linspace(0, 1, target_len)
                    interp = np.empty((target_len, k))
                    for c in range(k):
                        interp[:, c] = np.interp(new_t, old_t, enc[:, c])
                    mat[row] = interp.flatten()
        return mat, valid_idx, task_ids[valid_idx]

    # ---------- Within-task retrieval at 0cm ----------
    print("\n[2/5] WITHIN-TASK RETRIEVAL (L2, 0cm)")
    print(f"  {'Method':<20s} {'top-1':>8s} {'top-5':>8s}")

    l2_results = {}
    for name in all_encodings:
        mat, valid_idx, tids = build_matrix(all_encodings[name])
        if mat.size == 0:
            l2_results[name] = {"top1": 0, "top5": 0}
            continue
        D = cdist(mat, mat, metric="euclidean")
        np.fill_diagonal(D, np.inf)
        sorted_idx = np.argsort(D, axis=1)
        n = len(tids)
        c1 = sum(1 for i in range(n) if tids[sorted_idx[i, 0]] == tids[i])
        c5 = sum(1 for i in range(n) if tids[i] in tids[sorted_idx[i, :5]])
        r = {"top1": round(c1/n, 4), "top5": round(c5/n, 4), "n": n}
        l2_results[name] = r
        print(f"  {name:<20s} {r['top1']:>7.1%} {r['top5']:>7.1%}")
        del D, sorted_idx
        gc.collect()

    # ---------- Self-retrieval under SE(3) offset ----------
    print("\n[3/5] SELF-RETRIEVAL UNDER SE(3) OFFSET")
    offsets_cm = [0, 3, 5, 10]

    # For self-retrieval, we re-encode shifted versions of 50 random demos
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(task_ids), size=50, replace=False)

    # Reload sampled demos (need raw ee_pos for shifting)
    print("  Reloading 50 sample demos for self-retrieval...")
    sampled_demos = []
    demo_counter = 0
    for task_id in range(10):
        task = suite.get_task(task_id)
        dpath = os.path.join(get_libero_path("datasets"), "libero_spatial",
                             f"{task.name}_demo.hdf5")
        if not os.path.exists(dpath):
            continue
        with h5py.File(dpath, "r") as f:
            for key in sorted(f["data"].keys()):
                if demo_counter in sample_idx:
                    d = f["data"][key]
                    ee_pos = np.array(d["obs"]["ee_pos"])
                    ee_ori = np.array(d["obs"]["ee_states"])[:, 3:] if "ee_states" in d["obs"] else None
                    quats = get_quats_wxyz(ee_pos, ee_ori)
                    sampled_demos.append({"ee_pos": ee_pos, "quats": quats,
                                          "task_id": task_id, "idx": demo_counter})
                demo_counter += 1

    self_results = {}
    strategies_for_self = {
        "dhb_dr_full": lambda pos, q: encode_all_strategies(pos, q, M).get("dhb_dr_full"),
        "dhb_dr_mag": lambda pos, q: encode_all_strategies(pos, q, M).get("dhb_dr_mag"),
        "dhb_ti_full": lambda pos, q: encode_all_strategies(pos, q, M).get("dhb_ti_full"),
        "dhb_ti_mag": lambda pos, q: encode_all_strategies(pos, q, M).get("dhb_ti_mag"),
        "position": lambda pos, q: encode_all_strategies(pos, q, M).get("position"),
    }

    for name, enc_fn in strategies_for_self.items():
        self_results[name] = {}
        # Encode originals
        orig_encs = [enc_fn(d["ee_pos"], d["quats"]) for d in sampled_demos]

        for offset_cm in offsets_cm:
            offset_m = offset_cm / 100.0
            shift_encs = []
            for d in sampled_demos:
                shifted_pos = d["ee_pos"].copy()
                shifted_pos[:, 0] += offset_m
                shift_encs.append(enc_fn(shifted_pos, d["quats"]))

            # Match shifted to originals
            valid = [(i, orig_encs[i], shift_encs[i])
                     for i in range(len(sampled_demos))
                     if orig_encs[i] is not None and shift_encs[i] is not None]
            if not valid:
                self_results[name][f"{offset_cm}cm"] = 0.0
                continue

            # Flatten (interpolate if needed)
            target_len = valid[0][1].shape[0]
            k = valid[0][1].shape[1]

            def flatten(enc):
                if enc.shape[0] == target_len:
                    return enc.flatten()
                old_t = np.linspace(0, 1, enc.shape[0])
                new_t = np.linspace(0, 1, target_len)
                interp = np.empty((target_len, k))
                for c in range(k):
                    interp[:, c] = np.interp(new_t, old_t, enc[:, c])
                return interp.flatten()

            orig_mat = np.array([flatten(v[1]) for v in valid])
            shift_mat = np.array([flatten(v[2]) for v in valid])
            D = cdist(shift_mat, orig_mat, metric="euclidean")
            nn = np.argmin(D, axis=1)
            correct = int(np.sum(nn == np.arange(len(valid))))
            self_results[name][f"{offset_cm}cm"] = round(correct / len(valid), 4)

        print(f"  {name}: " + "  ".join(f"{oc}cm={self_results[name][f'{oc}cm']:.0%}"
                                         for oc in offsets_cm))

    # ---------- Within-task retrieval at various offsets ----------
    print("\n[4/5] WITHIN-TASK RETRIEVAL AT VARIOUS OFFSETS")

    # For each offset, re-encode ALL demos from scratch (memory efficient: one task at a time)
    cross_results = {}
    for name in all_encodings:
        cross_results[name] = {}

    for offset_cm in offsets_cm:
        offset_m = offset_cm / 100.0
        shifted_encs = {k: [] for k in all_encodings}
        shifted_tids = []

        for task_id in range(10):
            task = suite.get_task(task_id)
            dpath = os.path.join(get_libero_path("datasets"), "libero_spatial",
                                 f"{task.name}_demo.hdf5")
            if not os.path.exists(dpath):
                continue
            with h5py.File(dpath, "r") as f:
                for key in sorted(f["data"].keys()):
                    d = f["data"][key]
                    ee_pos = np.array(d["obs"]["ee_pos"]) + np.array([offset_m, 0, 0])
                    ee_ori = np.array(d["obs"]["ee_states"])[:, 3:] if "ee_states" in d["obs"] else None
                    quats = get_quats_wxyz(ee_pos, ee_ori)
                    encs = encode_all_strategies(ee_pos, quats, M=M)
                    for k in shifted_encs:
                        shifted_encs[k].append(encs.get(k))
                    shifted_tids.append(task_id)
            gc.collect()

        shifted_tids = np.array(shifted_tids)

        for name in all_encodings:
            mat, valid_idx, tids = build_matrix(shifted_encs[name])
            if mat.size == 0:
                cross_results[name][f"{offset_cm}cm"] = {"top1": 0, "top5": 0}
                continue
            # Use shifted task ids
            tids = shifted_tids[valid_idx]
            D = cdist(mat, mat, metric="euclidean")
            np.fill_diagonal(D, np.inf)
            sorted_idx = np.argsort(D, axis=1)
            n = len(tids)
            c1 = sum(1 for i in range(n) if tids[sorted_idx[i, 0]] == tids[i])
            c5 = sum(1 for i in range(n) if tids[i] in tids[sorted_idx[i, :5]])
            cross_results[name][f"{offset_cm}cm"] = {"top1": round(c1/n, 4), "top5": round(c5/n, 4)}
            del D, sorted_idx
            gc.collect()

        print(f"  offset={offset_cm}cm: " + "  ".join(
            f"{name}={cross_results[name][f'{offset_cm}cm'].get('top1',0):.1%}"
            for name in all_encodings))

    # ---------- Summary ----------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\n--- Within-task retrieval (L2, 0cm) ---")
    print(f"  {'Method':<20s} {'top-1':>8s} {'top-5':>8s}")
    for name in all_encodings:
        r = l2_results.get(name, {})
        print(f"  {name:<20s} {r.get('top1',0):>7.1%} {r.get('top5',0):>7.1%}")

    print("\n--- Self-retrieval under offset ---")
    print(f"  {'Method':<20s}", end="")
    for oc in offsets_cm:
        print(f" {oc}cm".rjust(8), end="")
    print()
    for name in all_encodings:
        r = self_results.get(name, {})
        print(f"  {name:<20s}", end="")
        for oc in offsets_cm:
            print(f" {r.get(f'{oc}cm', 0):>6.0%}", end="")
        print()

    print("\n--- Within-task top-1 at offsets ---")
    print(f"  {'Method':<20s}", end="")
    for oc in offsets_cm:
        print(f" {oc}cm".rjust(8), end="")
    print()
    for name in all_encodings:
        r = cross_results.get(name, {})
        print(f"  {name:<20s}", end="")
        for oc in offsets_cm:
            print(f" {r.get(f'{oc}cm', {}).get('top1', 0):>6.1%}", end="")
        print()

    results = {
        "description": "Quick motion DB comparison: DHB-DR vs DHB-TI vs magnitude-only vs position",
        "n_demos": len(task_ids),
        "M": M,
        "within_task_l2_0cm": l2_results,
        "self_retrieval_under_offset": self_results,
        "within_task_at_offsets": cross_results,
    }
    out = Path("_workspace/quick_motion_db_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
