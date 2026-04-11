#!/usr/bin/env python3
"""
Step 2.1-2.3: Generate DHB-augmented LIBERO training demos.

For each original demo, DHB encode → decode at random offsets to produce
geometrically correct augmented trajectories. Also generates naive-offset
demos (simple translation) as a baseline.

Usage:
    pixi run python _workspace/gen_dhb_augmented_demos.py \
        --task 0 --n-augment 9 --max-offset 0.05

Output:
    _workspace/augmented_demos_task0.npz
"""
from __future__ import annotations
import os, sys, argparse, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, "/home/andypark/Projects/repos/LIBERO")

BOWL_STATE_INDICES = {'X': 10, 'Y': 11, 'Z': 12}


def load_all_demos(task_id: int):
    """Load all 50 demos for a LIBERO-Spatial task."""
    import h5py
    from libero.libero import benchmark, get_libero_path

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(task_id)
    dpath = os.path.join(get_libero_path("datasets"), "libero_spatial",
                         f"{task.name}_demo.hdf5")
    task_name = task.name.replace("_", " ")

    demos = []
    with h5py.File(dpath) as f:
        for key in sorted(f["data"].keys()):
            d = f["data"][key]
            demo = {
                "init_state": np.array(d["states"][0]),
                "actions": np.array(d["actions"]),
                "ee_pos": np.array(d["obs"]["ee_pos"]),
            }
            # Get orientations — try ee_states first, fall back to eef_quat
            if "ee_states" in d["obs"]:
                demo["ee_ori_euler"] = np.array(d["obs"]["ee_states"])[:, 3:]
            demos.append(demo)

    print(f"Loaded {len(demos)} demos for task {task_id}: {task_name}")
    return demos, task_name


def euler_to_quat_wxyz(euler_xyz):
    """Convert Euler XYZ angles to quaternion wxyz."""
    from scipy.spatial.transform import Rotation
    rots = Rotation.from_euler("xyz", euler_xyz)
    q_xyzw = rots.as_quat()
    return np.column_stack([q_xyzw[:, 3], q_xyzw[:, 0], q_xyzw[:, 1], q_xyzw[:, 2]])


def dhb_augment_demo(demo, offset_vec):
    """
    DHB encode → decode from offset start/goal.

    Args:
        demo: dict with ee_pos (N,3), ee_ori_euler (N,3), actions (N,7)
        offset_vec: (3,) offset in meters

    Returns:
        dict with adapted ee_pos, ee_ori, actions, gripper (arc-length invariant)
    """
    from dhb_xr.encoder.dhb_dr import encode_dhb_dr
    from dhb_xr.decoder.dhb_dr import decode_dhb_dr
    from dhb_xr.core.types import EncodingMethod

    ee_pos = demo["ee_pos"]
    quats_wxyz = euler_to_quat_wxyz(demo["ee_ori_euler"])

    # New start and goal with offset
    new_start_pos = ee_pos[0] + offset_vec
    new_goal_pos = ee_pos[-1] + offset_vec
    new_start_quat = quats_wxyz[0]
    new_goal_quat = quats_wxyz[-1]

    # DHB encode
    enc = encode_dhb_dr(
        ee_pos, quats_wxyz,
        method="pos",
        use_default_initial_frames=True,
    )

    # DHB decode from offset start
    dec = decode_dhb_dr(
        enc["linear_motion_invariants"],
        enc["angular_motion_invariants"],
        initial_pose={"position": new_start_pos, "quaternion": new_start_quat},
        method=EncodingMethod.POSITION,
    )

    adapted_pos = dec["positions"]
    adapted_quat = dec["quaternions"]  # wxyz

    # Compute adapted actions (delta EE) from adapted trajectory
    # Original actions are 7D: [dx, dy, dz, drx, dry, drz, gripper]
    orig_actions = demo["actions"]
    N_orig = len(orig_actions)
    N_adapted = len(adapted_pos)

    # Interpolate adapted trajectory to match original action count
    from scipy.interpolate import interp1d
    t_adapted = np.linspace(0, 1, N_adapted)
    t_orig = np.linspace(0, 1, N_orig + 1)  # +1 because positions have one more than actions

    interp_pos = interp1d(t_adapted, adapted_pos, axis=0, fill_value="extrapolate")(t_orig)

    # Compute delta positions (adapted actions)
    adapted_actions = orig_actions.copy()
    for i in range(N_orig):
        # Replace position deltas with adapted ones
        adapted_actions[i, :3] = interp_pos[i + 1] - interp_pos[i]

    # Arc-length invariant gripper
    # Compute arc-length of original and adapted trajectories
    orig_diffs = np.diff(ee_pos, axis=0)
    orig_cumlen = np.concatenate([[0], np.cumsum(np.linalg.norm(orig_diffs, axis=1))])
    orig_total = orig_cumlen[-1] if orig_cumlen[-1] > 0 else 1.0
    orig_arc_frac = orig_cumlen / orig_total

    # Detect gripper transitions and their arc-length positions
    gripper_cmds = orig_actions[:, 6]
    gripper_transitions = []
    for t in range(1, len(gripper_cmds)):
        if (gripper_cmds[t] > 0) != (gripper_cmds[t - 1] > 0):
            gripper_transitions.append((orig_arc_frac[t], gripper_cmds[t]))

    # Map gripper transitions to adapted arc-length
    adapted_diffs = np.diff(interp_pos, axis=0)
    adapted_cumlen = np.concatenate([[0], np.cumsum(np.linalg.norm(adapted_diffs, axis=1))])
    adapted_total = adapted_cumlen[-1] if adapted_cumlen[-1] > 0 else 1.0
    adapted_arc_frac = adapted_cumlen / adapted_total

    current_gripper = gripper_cmds[0]
    trans_idx = 0
    for t in range(N_orig):
        arc = adapted_arc_frac[t] if t < len(adapted_arc_frac) else 1.0
        while trans_idx < len(gripper_transitions):
            frac, val = gripper_transitions[trans_idx]
            if arc >= frac:
                current_gripper = val
                trans_idx += 1
            else:
                break
        adapted_actions[t, 6] = current_gripper

    return {
        "actions": adapted_actions,
        "ee_pos": interp_pos[:-1],  # positions at each step
        "offset": offset_vec,
    }


def naive_augment_demo(demo, offset_vec):
    """
    Naive offset: simply translate all EE positions by offset.
    Does NOT preserve trajectory shape near constraints.
    """
    orig_actions = demo["actions"].copy()
    # For purely translational offset, action deltas don't change
    # (same relative motion, just shifted workspace)
    # But the initial state needs shifting
    return {
        "actions": orig_actions,
        "ee_pos": demo["ee_pos"] + offset_vec,
        "offset": offset_vec,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--n-augment", type=int, default=9,
                        help="Augmented demos per original demo")
    parser.add_argument("--max-offset", type=float, default=0.05,
                        help="Max offset in meters per axis")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.out is None:
        args.out = f"_workspace/augmented_demos_task{args.task}.npz"

    rng = np.random.RandomState(args.seed)
    demos, task_name = load_all_demos(args.task)

    print(f"\nGenerating {args.n_augment} augmented demos per original demo")
    print(f"Max offset: {args.max_offset*100:.1f}cm per axis")
    print(f"Total: {len(demos)} x {args.n_augment} = {len(demos) * args.n_augment} augmented demos")

    dhb_augmented = []
    naive_augmented = []
    offsets_used = []
    failed = 0

    t0 = time.time()
    for i, demo in enumerate(demos):
        if "ee_ori_euler" not in demo:
            print(f"  Demo {i}: skipping (no orientation data)")
            failed += 1
            continue

        for j in range(args.n_augment):
            # Random offset (uniform per axis)
            offset = rng.uniform(-args.max_offset, args.max_offset, size=3)
            offsets_used.append(offset)

            try:
                dhb_aug = dhb_augment_demo(demo, offset)
                dhb_augmented.append(dhb_aug["actions"])
            except Exception as e:
                print(f"  Demo {i}, aug {j}: DHB failed - {e}")
                failed += 1
                dhb_augmented.append(demo["actions"])  # fallback to original

            naive_aug = naive_augment_demo(demo, offset)
            naive_augmented.append(naive_aug["actions"])

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  Processed {i+1}/{len(demos)} demos ({elapsed:.1f}s)")

    elapsed = time.time() - t0

    # Also include original demos
    original_actions = [d["actions"] for d in demos]

    # Save
    # Note: actions have variable length, so we save as object arrays
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_path,
        task_id=args.task,
        task_name=task_name,
        n_original=len(demos),
        n_augmented_per_demo=args.n_augment,
        max_offset_m=args.max_offset,
        offsets_used=np.array(offsets_used),
        # Store actions as flat arrays with lengths
        original_lengths=np.array([len(a) for a in original_actions]),
        original_actions=np.concatenate(original_actions, axis=0),
        dhb_augmented_lengths=np.array([len(a) for a in dhb_augmented]),
        dhb_augmented_actions=np.concatenate(dhb_augmented, axis=0),
        naive_augmented_lengths=np.array([len(a) for a in naive_augmented]),
        naive_augmented_actions=np.concatenate(naive_augmented, axis=0),
    )

    size_mb = out_path.stat().st_size / 1e6
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Original: {len(original_actions)} demos")
    print(f"  DHB augmented: {len(dhb_augmented)} demos")
    print(f"  Naive augmented: {len(naive_augmented)} demos")
    print(f"  Failed: {failed}")
    print(f"  Saved: {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
