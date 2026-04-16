#!/usr/bin/env python3
"""
Improved motion database retrieval comparison.

Compares retrieval accuracy across multiple encoding strategies:
  1. DHB-DR (baseline, all 8 invariants)
  2. DHB-DR magnitude-only (2 invariants: linear + angular magnitude)
  3. DHB-TI (arc-length reparameterized, all 8 invariants)
  4. DHB-TI magnitude-only (2 invariants)
  5. Position-based (raw ee_pos, no invariants)

Key optimization: all DHB-TI encodings produce fixed-length output (M samples),
enabling vectorized distance matrix computation instead of O(n^2) Python loops.

Usage:
    MUJOCO_GL=egl pixi run python _workspace/improved_motion_db.py
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, "/home/andypark/Projects/repos/LIBERO")

# Force unbuffered output
import builtins
_original_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _original_print(*args, **kwargs)
builtins.print = print


def load_all_libero_demos():
    """Load all LIBERO-Spatial demos (10 tasks x 50 demos each)."""
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
                if "ee_states" in d["obs"]:
                    demo["ee_ori_euler"] = np.array(d["obs"]["ee_states"])[:, 3:]
                all_demos.append(demo)
        print(f"  Task {task_id}: {n_demos} demos -- {task_name[:50]}")

    return all_demos


def get_quats_wxyz(demo):
    from scipy.spatial.transform import Rotation
    ee_pos = demo["ee_pos"]
    if "ee_ori_euler" not in demo:
        return np.tile([1, 0, 0, 0], (len(ee_pos), 1)).astype(np.float64)
    rots = Rotation.from_euler("xyz", demo["ee_ori_euler"])
    q_xyzw = rots.as_quat()
    return np.column_stack([q_xyzw[:, 3], q_xyzw[:, 0],
                            q_xyzw[:, 1], q_xyzw[:, 2]])


# ---------- Encoding strategies ----------

def encode_dhb_dr_full(demo):
    from dhb_xr.encoder.dhb_dr import encode_dhb_dr
    try:
        enc = encode_dhb_dr(demo["ee_pos"], get_quats_wxyz(demo),
                            method="pos", use_default_initial_frames=True)
        return np.concatenate([enc["linear_motion_invariants"],
                               enc["angular_motion_invariants"]], axis=1)
    except Exception:
        return None


def encode_dhb_dr_mag(demo):
    from dhb_xr.encoder.dhb_dr import encode_dhb_dr
    try:
        enc = encode_dhb_dr(demo["ee_pos"], get_quats_wxyz(demo),
                            method="pos", use_default_initial_frames=True)
        return np.concatenate([enc["linear_motion_invariants"][:, 0:1],
                               enc["angular_motion_invariants"][:, 0:1]], axis=1)
    except Exception:
        return None


def encode_dhb_ti_full(demo, M=64):
    from dhb_xr.encoder.dhb_ti import encode_dhb_dr_ti
    try:
        enc = encode_dhb_dr_ti(demo["ee_pos"], get_quats_wxyz(demo), M=M,
                               progress_kind="translation",
                               method="pos", use_default_initial_frames=True)
        return np.concatenate([enc["linear_motion_invariants"],
                               enc["angular_motion_invariants"]], axis=1)
    except Exception:
        return None


def encode_dhb_ti_mag(demo, M=64):
    from dhb_xr.encoder.dhb_ti import encode_dhb_dr_ti
    try:
        enc = encode_dhb_dr_ti(demo["ee_pos"], get_quats_wxyz(demo), M=M,
                               progress_kind="translation",
                               method="pos", use_default_initial_frames=True)
        return np.concatenate([enc["linear_motion_invariants"][:, 0:1],
                               enc["angular_motion_invariants"][:, 0:1]], axis=1)
    except Exception:
        return None


def encode_position_fixed(demo, M=64):
    """Interpolate raw positions to fixed length M for fair comparison."""
    pos = demo["ee_pos"].astype(np.float64)
    old_t = np.linspace(0, 1, len(pos))
    new_t = np.linspace(0, 1, M)
    result = np.empty((M, 3))
    for c in range(3):
        result[:, c] = np.interp(new_t, old_t, pos[:, c])
    return result


# ---------- Vectorized distance matrix ----------

def build_flat_matrix(encodings, target_len=None):
    """
    Build (N, D) matrix from list of (T, k) arrays.
    For fixed-length encodings (DHB-TI, position_fixed), just flatten.
    For variable-length (DHB-DR), interpolate to target_len then flatten.
    """
    valid_idx = [i for i, e in enumerate(encodings) if e is not None]
    if not valid_idx:
        return np.array([]), valid_idx

    shapes = set(e.shape for e in encodings if e is not None)
    # If all same shape, just flatten
    if len(shapes) == 1:
        mat = np.array([encodings[i].flatten() for i in valid_idx], dtype=np.float64)
        return mat, valid_idx

    # Variable length: interpolate to common length
    if target_len is None:
        lengths = [encodings[i].shape[0] for i in valid_idx]
        target_len = int(np.median(lengths))

    k = encodings[valid_idx[0]].shape[1]
    mat = np.empty((len(valid_idx), target_len * k), dtype=np.float64)
    for row, i in enumerate(valid_idx):
        enc = encodings[i]
        if enc.shape[0] == target_len:
            mat[row] = enc.flatten()
        else:
            # Interpolate
            old_t = np.linspace(0, 1, enc.shape[0])
            new_t = np.linspace(0, 1, target_len)
            interp = np.empty((target_len, k))
            for c in range(k):
                interp[:, c] = np.interp(new_t, old_t, enc[:, c])
            mat[row] = interp.flatten()
    return mat, valid_idx


def retrieval_from_matrix(mat, valid_idx, metadata, exclude_self=True):
    """
    Vectorized retrieval using scipy cdist.
    mat: (N_valid, D)
    """
    from scipy.spatial.distance import cdist
    n = mat.shape[0]
    if n < 2:
        return {"top1": 0, "top5": 0, "total": 0}

    D = cdist(mat, mat, metric="euclidean")

    if exclude_self:
        np.fill_diagonal(D, np.inf)

    task_ids = np.array([metadata[valid_idx[i]]["task_id"] for i in range(n)])
    sorted_idx = np.argsort(D, axis=1)

    correct_top1 = 0
    correct_top5 = 0
    for i in range(n):
        query_task = task_ids[i]
        nn_task = task_ids[sorted_idx[i, 0]]
        if nn_task == query_task:
            correct_top1 += 1
        top5_tasks = task_ids[sorted_idx[i, :5]]
        if query_task in top5_tasks:
            correct_top5 += 1

    return {
        "top1": round(correct_top1 / n, 4),
        "top5": round(correct_top5 / n, 4),
        "total": n,
    }


def dtw_retrieval_subsample(encodings, metadata, subsample_idx):
    """DTW retrieval on a subsample (Python loop, slow but handles variable length)."""
    from dhb_xr.database.similarity import soft_dtw_distance

    sub_encs = [encodings[i] for i in subsample_idx]
    sub_meta = [metadata[i] for i in subsample_idx]
    n = len(sub_encs)

    valid = [(i, sub_encs[i]) for i in range(n) if sub_encs[i] is not None]
    nv = len(valid)
    if nv < 2:
        return {"top1": 0, "top5": 0, "total": 0}

    D = np.full((nv, nv), np.inf)
    for a in range(nv):
        for b in range(a + 1, nv):
            d = soft_dtw_distance(valid[a][1], valid[b][1], gamma=0.5)
            D[a, b] = d
            D[b, a] = d

    task_ids = np.array([sub_meta[valid[i][0]]["task_id"] for i in range(nv)])
    sorted_idx = np.argsort(D, axis=1)

    correct_top1 = 0
    correct_top5 = 0
    for i in range(nv):
        qt = task_ids[i]
        if task_ids[sorted_idx[i, 0]] == qt:
            correct_top1 += 1
        if qt in task_ids[sorted_idx[i, :5]]:
            correct_top5 += 1

    return {
        "top1": round(correct_top1 / nv, 4),
        "top5": round(correct_top5 / nv, 4),
        "total": nv,
    }


# ---------- Apply SE(3) offset ----------

def apply_offset(demo, offset_m):
    new_demo = dict(demo)
    new_demo["ee_pos"] = demo["ee_pos"].copy()
    new_demo["ee_pos"][:, 0] += offset_m
    return new_demo


# ---------- Self-retrieval ----------

def self_retrieval_vectorized(demos, encode_fn, offsets_cm, n_sample=50):
    """
    Self-retrieval: encode subset, shift by offset, re-encode,
    check if shifted version matches the original.
    """
    rng = np.random.RandomState(42)
    indices = rng.choice(len(demos), size=min(n_sample, len(demos)), replace=False)

    originals = [encode_fn(demos[idx]) for idx in indices]

    results = {}
    for offset_cm in offsets_cm:
        offset_m = offset_cm / 100.0
        shifted = [encode_fn(apply_offset(demos[idx], offset_m)) for idx in indices]

        valid_pairs = [(i, originals[i], shifted[i])
                       for i in range(len(indices))
                       if originals[i] is not None and shifted[i] is not None]
        if not valid_pairs:
            results[f"{offset_cm}cm"] = {"accuracy": 0, "correct": 0, "total": 0}
            continue

        # Build matrix for originals
        n = len(valid_pairs)
        # Use interpolation for variable-length
        target_len = valid_pairs[0][1].shape[0]
        k = valid_pairs[0][1].shape[1]

        def flatten_interp(enc):
            if enc.shape[0] == target_len:
                return enc.flatten()
            old_t = np.linspace(0, 1, enc.shape[0])
            new_t = np.linspace(0, 1, target_len)
            interp = np.empty((target_len, k))
            for c in range(k):
                interp[:, c] = np.interp(new_t, old_t, enc[:, c])
            return interp.flatten()

        orig_mat = np.array([flatten_interp(vp[1]) for vp in valid_pairs])
        shift_mat = np.array([flatten_interp(vp[2]) for vp in valid_pairs])

        from scipy.spatial.distance import cdist
        D = cdist(shift_mat, orig_mat, metric="euclidean")
        nn = np.argmin(D, axis=1)
        correct = int(np.sum(nn == np.arange(n)))

        results[f"{offset_cm}cm"] = {
            "accuracy": round(correct / n, 4),
            "correct": correct,
            "total": n,
        }
    return results


# ---------- Main ----------

def main():
    print("=" * 70)
    print("IMPROVED MOTION DB: DHB-TI + Magnitude-Only + DTW Comparison")
    print("=" * 70)

    print("\n[1/6] Loading LIBERO-Spatial demos...")
    demos = load_all_libero_demos()
    n_demos = len(demos)
    metadata = [{"task_id": d["task_id"], "task_name": d["task_name"],
                 "demo_key": d["demo_key"]} for d in demos]
    print(f"  Total: {n_demos} demos\n")

    # ---------- Encode ----------
    strategies = {
        "dhb_dr_full":  encode_dhb_dr_full,
        "dhb_dr_mag":   encode_dhb_dr_mag,
        "dhb_ti_full":  encode_dhb_ti_full,
        "dhb_ti_mag":   encode_dhb_ti_mag,
        "position":     encode_position_fixed,
    }

    all_encodings = {}
    for name, enc_fn in strategies.items():
        print(f"  Encoding {name}...", end=" ")
        t0 = time.time()
        encs = []
        for i, d in enumerate(demos):
            encs.append(enc_fn(d))
            if (i + 1) % 100 == 0:
                print(f"{i+1}", end=" ")
        elapsed = time.time() - t0
        n_valid = sum(1 for e in encs if e is not None)
        shapes = set(e.shape for e in encs if e is not None)
        print(f"  -> {n_valid}/{n_demos} valid, {elapsed:.1f}s, shapes: {shapes}")
        all_encodings[name] = encs

    # ---------- Within-task retrieval (L2) ----------
    print("\n" + "=" * 70)
    print("[3/6] WITHIN-TASK RETRIEVAL (L2, exclude self)")
    print("=" * 70)

    l2_results = {}
    for name in strategies:
        encs = all_encodings[name]
        mat, valid_idx = build_flat_matrix(encs)
        if mat.size == 0:
            l2_results[name] = {"top1": 0, "top5": 0, "total": 0}
            continue
        print(f"  {name}: matrix {mat.shape}...", end=" ")
        t0 = time.time()
        r = retrieval_from_matrix(mat, valid_idx, metadata)
        print(f"top-1={r['top1']:.1%}  top-5={r['top5']:.1%}  ({time.time()-t0:.1f}s)")
        l2_results[name] = r

    # ---------- DTW retrieval (subsample 50) ----------
    print("\n" + "=" * 70)
    print("[4/6] WITHIN-TASK RETRIEVAL (soft-DTW, subsample 50)")
    print("=" * 70)

    rng = np.random.RandomState(42)
    subsample_idx = rng.choice(n_demos, size=min(50, n_demos), replace=False)

    dtw_results = {}
    for name in ["dhb_dr_full", "dhb_dr_mag", "dhb_ti_full", "dhb_ti_mag"]:
        print(f"  {name} (DTW)...", end=" ")
        t0 = time.time()
        r = dtw_retrieval_subsample(all_encodings[name], metadata, subsample_idx)
        elapsed = time.time() - t0
        print(f"top-1={r['top1']:.1%}  top-5={r['top5']:.1%}  ({elapsed:.1f}s)")
        dtw_results[name] = r

    # ---------- Self-retrieval under SE(3) offset ----------
    print("\n" + "=" * 70)
    print("[5/6] SELF-RETRIEVAL UNDER SE(3) OFFSET (50 samples)")
    print("=" * 70)

    offsets_cm = [0, 3, 5, 10]
    self_retrieval_results = {}
    for name, enc_fn in strategies.items():
        print(f"  {name}:")
        r = self_retrieval_vectorized(demos, enc_fn, offsets_cm, n_sample=50)
        self_retrieval_results[name] = r
        for oc in offsets_cm:
            acc = r[f"{oc}cm"]["accuracy"]
            print(f"    offset={oc}cm: {acc:.1%}")

    # ---------- Within-task at various offsets ----------
    print("\n" + "=" * 70)
    print("[6/6] WITHIN-TASK RETRIEVAL AT VARIOUS OFFSETS")
    print("=" * 70)

    cross_offset_results = {}
    for name, enc_fn in strategies.items():
        cross_offset_results[name] = {}
        print(f"  {name}:")
        for offset_cm in offsets_cm:
            offset_m = offset_cm / 100.0
            shifted_encs = [enc_fn(apply_offset(d, offset_m)) for d in demos]
            mat, valid_idx = build_flat_matrix(shifted_encs)
            if mat.size == 0:
                cross_offset_results[name][f"{offset_cm}cm"] = {"top1": 0, "top5": 0, "total": 0}
                continue
            r = retrieval_from_matrix(mat, valid_idx, metadata)
            cross_offset_results[name][f"{offset_cm}cm"] = r
            print(f"    offset={offset_cm}cm: top-1={r['top1']:.1%}  top-5={r['top5']:.1%}")

    # ---------- Summary ----------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    results = {
        "description": "Improved motion DB: DHB-TI + magnitude-only + DTW comparison",
        "n_demos": n_demos,
        "within_task_l2": l2_results,
        "within_task_dtw_subsample50": dtw_results,
        "self_retrieval_under_offset": self_retrieval_results,
        "within_task_at_offsets": cross_offset_results,
    }

    print("\n--- Within-task retrieval (L2, 0cm) ---")
    print(f"  {'Method':<20s} {'top-1':>8s} {'top-5':>8s}")
    for name in strategies:
        r = l2_results.get(name, {})
        print(f"  {name:<20s} {r.get('top1',0):>7.1%} {r.get('top5',0):>7.1%}")

    print("\n--- Within-task retrieval (DTW, subsample 50) ---")
    print(f"  {'Method':<20s} {'top-1':>8s} {'top-5':>8s}")
    for name in dtw_results:
        r = dtw_results[name]
        print(f"  {name:<20s} {r['top1']:>7.1%} {r['top5']:>7.1%}")

    print("\n--- Self-retrieval under offset (L2) ---")
    print(f"  {'Method':<20s}", end="")
    for oc in offsets_cm:
        print(f" {oc}cm".rjust(8), end="")
    print()
    for name in strategies:
        r = self_retrieval_results.get(name, {})
        print(f"  {name:<20s}", end="")
        for oc in offsets_cm:
            acc = r.get(f"{oc}cm", {}).get("accuracy", 0)
            print(f" {acc:>6.1%}", end="")
        print()

    print("\n--- Within-task retrieval at offsets ---")
    print(f"  {'Method':<20s}", end="")
    for oc in offsets_cm:
        print(f" {oc}cm".rjust(8), end="")
    print()
    for name in strategies:
        r = cross_offset_results.get(name, {})
        print(f"  {name:<20s}", end="")
        for oc in offsets_cm:
            acc = r.get(f"{oc}cm", {}).get("top1", 0)
            print(f" {acc:>6.1%}", end="")
        print()

    out_path = Path("_workspace/improved_motion_db_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
