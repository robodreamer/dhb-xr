from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_POSES_NPZ = "/tmp/dhb_embodik_poses.npz"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-npz", required=True)
    parser.add_argument("--output-npz", default=DEFAULT_POSES_NPZ)
    parser.add_argument("--offset-cm", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-json", default=None)
    args = parser.parse_args()

    data = np.load(args.stage1_npz, allow_pickle=True)
    positions = np.asarray(data["positions"], dtype=np.float32)
    quaternions = np.asarray(data["quaternions"], dtype=np.float32)
    gripper = np.asarray(data["gripper"], dtype=np.float32)
    init_state = np.asarray(data["init_state"])

    key = f"offset{args.offset_cm:.0f}_seed{args.seed}"
    payload = {
        f"{key}_pos": positions,
        f"{key}_quat": quaternions,
        "gripper": gripper,
        "init_state": init_state,
        "demo_ee_pos": positions.copy(),
        "offsets_cm": np.array([args.offset_cm], dtype=np.float32),
        "seeds": np.array([args.seed + 1], dtype=np.int32),
        "source": np.array(["starvla_embodik_stage1"], dtype=object),
    }

    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_npz, **payload)

    summary = {
        "stage1_npz": args.stage1_npz,
        "output_npz": str(output_npz),
        "offset_cm": args.offset_cm,
        "seed": args.seed,
        "positions_shape": list(positions.shape),
        "quaternions_shape": list(quaternions.shape),
        "gripper_shape": list(gripper.shape),
    }

    if args.save_json:
        save_json = Path(args.save_json)
        save_json.parent.mkdir(parents=True, exist_ok=True)
        save_json.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
