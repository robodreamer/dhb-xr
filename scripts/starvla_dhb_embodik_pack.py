from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-npz", nargs="+", required=True, help="One stage1 npz per seed, in seed order.")
    parser.add_argument("--offset-cm", nargs="+", type=float, default=[0.0, 3.0])
    parser.add_argument("--output-npz", default="/tmp/dhb_embodik_poses.npz")
    parser.add_argument("--save-json", default=None)
    args = parser.parse_args()

    offsets_cm = [float(v) for v in args.offset_cm]
    payload: dict[str, object] = {
        "offsets_cm": np.asarray(offsets_cm, dtype=np.float32),
        "seeds": np.asarray([len(args.stage1_npz)], dtype=np.int32),
        "source": np.asarray(["starvla_embodik_stage1_multiseed"], dtype=object),
    }

    summary_runs = []
    for seed, path_str in enumerate(args.stage1_npz):
        path = Path(path_str)
        data = np.load(path, allow_pickle=True)
        positions = np.asarray(data["positions"], dtype=np.float32)
        quaternions = np.asarray(data["quaternions"], dtype=np.float32)
        gripper = np.asarray(data["gripper"], dtype=np.float32)
        init_state = np.asarray(data["init_state"])
        episode_idx = int(np.asarray(data["episode_idx"]).item()) if "episode_idx" in data else seed

        payload[f"gripper_seed{seed}"] = gripper
        payload[f"init_state_seed{seed}"] = init_state
        payload[f"episode_idx_seed{seed}"] = np.asarray([episode_idx], dtype=np.int32)
        if seed == 0:
            payload["demo_ee_pos"] = positions.copy()

        for offset_cm in offsets_cm:
            key = f"offset{offset_cm:.0f}_seed{seed}"
            shifted_pos = positions.copy()
            if offset_cm != 0:
                shifted_pos[:, 1] += offset_cm / 100.0
            payload[f"{key}_pos"] = shifted_pos
            payload[f"{key}_quat"] = quaternions

        summary_runs.append(
            {
                "seed": seed,
                "stage1_npz": str(path),
                "positions_shape": list(positions.shape),
                "quaternions_shape": list(quaternions.shape),
                "episode_idx": episode_idx,
                "first_position": positions[0].tolist(),
                "last_position": positions[-1].tolist(),
            }
        )

    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_npz, **payload)

    summary = {
        "output_npz": str(output_npz),
        "offsets_cm": offsets_cm,
        "num_seeds": len(args.stage1_npz),
        "runs": summary_runs,
    }
    if args.save_json:
        Path(args.save_json).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
