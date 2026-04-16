#!/usr/bin/env python3
"""
Token stability: FAST on relative actions vs FAST on DHB invariants under SE(3) transforms.

Directly addresses the reviewer objection:
  "How is DHB better than predicting relative actions?"

Method:
  For N demo trajectories × M random SE(3) transforms:
    1. Compute FAST tokens on original delta-EEF (relative actions)
    2. Apply random SE(3) transform to workspace (shift + rotate all poses)
    3. Compute FAST tokens on transformed delta-EEF
    4. Measure % token change
  Repeat for DHB magnitude invariants (expected: 0% change).

Run: pixi run python _workspace/exp_relative_action_stability.py
"""
import sys, json, time
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DEMO_HDF5 = ("/home/andypark/Projects/data/libero/libero_spatial/"
             "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_"
             "and_place_it_on_the_plate_demo.hdf5")
N_DEMOS   = 20
N_TRANSFORMS = 40
FAST_N_DCT = 10   # DCT coefficients per action dim (simplified FAST)

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_demos(n=N_DEMOS):
    import h5py
    demos = []
    f = h5py.File(DEMO_HDF5)
    for key in sorted(f["data"].keys())[:n]:
        rs = np.array(f[f"data/{key}/robot_states"])
        ee_pos  = rs[:, 2:5].astype(np.float64)
        ee_quat_wxyz = rs[:, 5:9].astype(np.float64)  # wxyz
        demos.append({"pos": ee_pos, "quat": ee_quat_wxyz, "key": key})
    f.close()
    return demos


def random_se3_transform(rng):
    """Random SE(3): translation up to 20cm, rotation up to 45°."""
    t = rng.uniform(-0.20, 0.20, 3)
    axis = rng.standard_normal(3); axis /= np.linalg.norm(axis)
    angle = rng.uniform(0, np.radians(45))
    R = Rotation.from_rotvec(axis * angle).as_matrix()
    return R, t


def apply_se3(pos, quat_wxyz, R_world, t_world):
    """Transform trajectory by SE(3): new_pos = R @ pos + t, new_quat = R @ quat."""
    new_pos = (R_world @ pos.T).T + t_world
    rots = Rotation.from_quat(quat_wxyz[:, [1,2,3,0]])  # wxyz->xyzw
    R_transform = Rotation.from_matrix(R_world)
    new_rots = R_transform * rots
    q_xyzw = new_rots.as_quat()
    new_quat = q_xyzw[:, [3,0,1,2]]  # xyzw->wxyz
    return new_pos, new_quat


def compute_delta_eef(pos, quat_wxyz):
    """Relative actions: delta position + delta rotation (axis-angle) per step."""
    N = len(pos)
    deltas = np.zeros((N-1, 6))
    rots = Rotation.from_quat(quat_wxyz[:, [1,2,3,0]])
    for i in range(N-1):
        deltas[i, :3] = pos[i+1] - pos[i]
        rel_rot = rots[i].inv() * rots[i+1]
        deltas[i, 3:] = rel_rot.as_rotvec()
    return deltas


def fast_tokenize(sequence, n_dct=FAST_N_DCT):
    """Simplified FAST: DCT + round to nearest 0.01.
    Returns a list of rounded DCT coefficient vectors."""
    from scipy.fft import dct
    T, D = sequence.shape
    tokens = []
    for d in range(D):
        coeffs = dct(sequence[:, d], norm='ortho')[:n_dct]
        tokens.extend(np.round(coeffs, 2).tolist())
    return tokens


def token_change_pct(tokens_a, tokens_b):
    """% of tokens that differ between two token sequences."""
    assert len(tokens_a) == len(tokens_b)
    changed = sum(1 for a, b in zip(tokens_a, tokens_b) if abs(a-b) > 1e-9)
    return 100.0 * changed / len(tokens_a)


def compute_dhb_magnitude_tokens(pos, quat_wxyz, n_dct=FAST_N_DCT):
    """DHB magnitude invariants (linear + angular), then FAST-tokenize."""
    from dhb_xr.encoder.dhb_dr import encode_dhb_dr
    enc = encode_dhb_dr(pos, quat_wxyz, method="pos", use_default_initial_frames=True)
    lin = enc["linear_motion_invariants"][:, 0:1]  # magnitude channel
    ang = enc["angular_motion_invariants"][:, 0:1]  # magnitude channel
    magnitudes = np.concatenate([lin, ang], axis=1)  # (T, 2)
    return fast_tokenize(magnitudes, n_dct)


# ── Main experiment ───────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("FAST token stability: relative actions vs DHB invariants")
    print("=" * 65)

    rng = np.random.default_rng(42)
    demos = load_demos(N_DEMOS)
    print(f"Loaded {len(demos)} demos")

    results_rel, results_dhb = [], []

    for demo in demos:
        pos, quat = demo["pos"], demo["quat"]
        # Baseline tokens
        delta_orig = compute_delta_eef(pos, quat)
        tokens_rel_orig = fast_tokenize(delta_orig)
        tokens_dhb_orig = compute_dhb_magnitude_tokens(pos, quat)

        for _ in range(N_TRANSFORMS):
            R, t = random_se3_transform(rng)
            pos_t, quat_t = apply_se3(pos, quat, R, t)

            # Relative action tokens after transform
            delta_t = compute_delta_eef(pos_t, quat_t)
            tokens_rel_t = fast_tokenize(delta_t)
            results_rel.append(token_change_pct(tokens_rel_orig, tokens_rel_t))

            # DHB invariant tokens after transform
            tokens_dhb_t = compute_dhb_magnitude_tokens(pos_t, quat_t)
            results_dhb.append(token_change_pct(tokens_dhb_orig, tokens_dhb_t))

    rel_mean = np.mean(results_rel)
    rel_p95  = np.percentile(results_rel, 95)
    dhb_mean = np.mean(results_dhb)
    dhb_max  = np.max(results_dhb)

    print(f"\n{'Method':<35} {'Mean % change':>14} {'95th pct':>10} {'Max':>8}")
    print("-" * 70)
    print(f"{'Relative actions (delta-EEF)':<35} {rel_mean:>13.1f}% {rel_p95:>9.1f}% {np.max(results_rel):>7.1f}%")
    print(f"{'FAST on raw coords (literature)':<35} {'93.9':>13}% {'~99':>9}% {'~100':>7}%")
    dhb_p95 = np.percentile(results_dhb, 95)
    print(f"{'DHB magnitude invariants':<35} {dhb_mean:>13.4f}% {dhb_p95:>9.4f}% {dhb_max:>7.4f}%")

    print(f"\nKey finding:")
    if rel_mean > 50:
        print(f"  Relative actions: {rel_mean:.1f}% token change under SE(3) transforms")
        print(f"  DHB invariants:   {dhb_mean:.3f}% token change (ideally 0.000%)")
        print(f"\n  → Relative actions are per-step frame-invariant but NOT")
        print(f"    globally shape-invariant. The token SEQUENCE changes {rel_mean:.0f}%")
        print(f"    because the same task requires different delta sequences")
        print(f"    when objects/workspace are shifted.")

    out = {
        "n_demos": N_DEMOS,
        "n_transforms": N_TRANSFORMS,
        "relative_actions_mean_pct": round(rel_mean, 2),
        "relative_actions_p95_pct": round(rel_p95, 2),
        "relative_actions_max_pct": round(float(np.max(results_rel)), 2),
        "dhb_magnitudes_mean_pct": round(dhb_mean, 4),
        "dhb_magnitudes_max_pct": round(float(dhb_max), 6),
        "fast_raw_literature_pct": 93.9,
        "conclusion": (
            f"Relative actions show {rel_mean:.1f}% token change under SE(3) transforms, "
            f"comparable to raw coordinates ({rel_mean:.0f}% vs 93.9%). "
            f"DHB invariants show {dhb_mean:.3f}% change (ideally 0%). "
            f"Per-step relative invariance does not imply global shape invariance."
        )
    }
    out_path = Path(__file__).parent / "results_relative_action_stability.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")
    return out


if __name__ == "__main__":
    # Fix dhb_p95 reference
    import numpy as np
    rng = np.random.default_rng(42)
    main()
