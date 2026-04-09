"""Precompute DHB invariants from LIBERO demos for StarVLA training.

Adds 'dhb_invariants' dataset to each LIBERO HDF5 file so WM4A_DHB
can load ground-truth invariants during training without re-encoding.

Usage:
    # Precompute for all libero_spatial demos
    python precompute_dhb_invariants.py --dataset-dir /path/to/libero_spatial

    # Preview only (no write)
    python precompute_dhb_invariants.py --dataset-dir /path/to/libero_spatial --dry-run

After running, each HDF5 file will contain:
    data/demo_N/dhb_invariants/
        linear_invariants   (T-1, 4) float32
        angular_invariants  (T-1, 4) float32
        initial_pose_pos    (3,) float32
        initial_pose_quat   (4,) float32 — wxyz
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dhb_xr.encoder.dhb_dr import encode_dhb_dr


def precompute_for_file(hdf5_path: Path, dry_run: bool = False) -> dict:
    """Precompute DHB invariants for all demos in one HDF5 file."""
    import h5py
    from scipy.spatial.transform import Rotation

    stats = {"demos": 0, "steps": 0, "skipped": 0, "errors": 0}

    with h5py.File(str(hdf5_path), "r+" if not dry_run else "r") as f:
        data = f.get("data", f)
        demo_keys = sorted(k for k in data.keys() if "demo" in k.lower())

        for dk in demo_keys:
            demo = data[dk]
            obs = demo.get("obs", demo)

            # Already computed?
            if "dhb_invariants" in demo and not dry_run:
                stats["skipped"] += 1
                continue

            # Extract ee_pos + ee_quat (wxyz)
            try:
                pos = np.array(obs["ee_pos"])  # (T, 3)

                if "robot_states" in demo:
                    rs = np.array(demo["robot_states"])
                    quat_wxyz = rs[:, 3:7]
                elif "ee_ori" in obs:
                    euler = np.array(obs["ee_ori"])
                    xyzw = Rotation.from_euler("xyz", euler).as_quat()
                    quat_wxyz = np.column_stack([xyzw[:, 3], xyzw[:, 0], xyzw[:, 1], xyzw[:, 2]])
                else:
                    stats["errors"] += 1
                    continue

                # Encode
                enc = encode_dhb_dr(pos, quat_wxyz)
                lin_inv = enc["linear_motion_invariants"].astype(np.float32)   # (T-1, 4)
                ang_inv = enc["angular_motion_invariants"].astype(np.float32)  # (T-1, 4)
                ip = enc["initial_pose"]

                if not dry_run:
                    grp = demo.require_group("dhb_invariants")
                    for key in ["linear_invariants", "angular_invariants",
                                "initial_pose_pos", "initial_pose_quat"]:
                        if key in grp:
                            del grp[key]
                    grp.create_dataset("linear_invariants", data=lin_inv)
                    grp.create_dataset("angular_invariants", data=ang_inv)
                    grp.create_dataset("initial_pose_pos",
                                       data=np.array(ip["position"], dtype=np.float32))
                    grp.create_dataset("initial_pose_quat",
                                       data=np.array(ip["quaternion"], dtype=np.float32))

                stats["demos"] += 1
                stats["steps"] += len(pos)

            except Exception as e:
                print(f"  Error on {hdf5_path.name}/{dk}: {e}")
                stats["errors"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Precompute DHB invariants for LIBERO demos")
    parser.add_argument("--dataset-dir", default="/home/andypark/Projects/data/libero/libero_spatial",
                        help="Directory containing LIBERO HDF5 files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview only — do not write to files")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Limit number of files to process")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    files = sorted(dataset_dir.glob("*.hdf5"))
    if args.max_files:
        files = files[: args.max_files]

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Precomputing DHB invariants")
    print(f"  Dataset: {dataset_dir}")
    print(f"  Files:   {len(files)}")
    print()

    total = {"demos": 0, "steps": 0, "skipped": 0, "errors": 0}
    for fi, fpath in enumerate(files):
        stats = precompute_for_file(fpath, dry_run=args.dry_run)
        total = {k: total[k] + stats[k] for k in total}
        print(
            f"  [{fi+1}/{len(files)}] {fpath.name[:50]:50s}  "
            f"demos={stats['demos']}  steps={stats['steps']}  "
            f"skip={stats['skipped']}  err={stats['errors']}"
        )

    print()
    print(f"Done. Total: {total['demos']} demos, {total['steps']} steps, "
          f"{total['skipped']} skipped, {total['errors']} errors")

    if not args.dry_run:
        print()
        print("DHB invariants written. Training config:")
        print("  gt_invariants key: 'dhb_invariants/linear_invariants' + 'angular_invariants'")
        print("  Shape: (T-1, 4) each — use chunk_len to slice")


if __name__ == "__main__":
    main()
