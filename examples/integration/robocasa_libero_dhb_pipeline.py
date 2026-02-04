"""End-to-end DHB-XR pipeline demo for RoboCASA and Libero."""

from __future__ import annotations

import argparse
from pathlib import Path

from dhb_xr.database.motion_db import MotionDatabase
from dhb_xr.integration.vla import RoboCASAAdapter, LiberoAdapter, DHBVLAPipeline


def run_pipeline(name: str, dataset_path: Path, max_episodes: int = 10) -> None:
    adapter = RoboCASAAdapter() if name == "robocasa" else LiberoAdapter()
    pipeline = DHBVLAPipeline()

    episodes = []
    for i, ep in enumerate(adapter.load_dataset(str(dataset_path))):
        episodes.append(ep)
        if i + 1 >= max_episodes:
            break

    print(f"[{name}] Loaded {len(episodes)} episodes from {dataset_path}")

    outputs = pipeline.process_dataset(episodes)
    print(f"[{name}] Tokenized {len(outputs)} trajectories")
    if outputs:
        print(f"[{name}] Example token shape: {outputs[0]['tokens'].shape}")

    # Minimal retrieval demo
    db = MotionDatabase()
    for ep in episodes:
        db.add(ep["positions"], ep["quaternions"], metadata=ep.get("metadata"))

    if episodes:
        query = episodes[0]
        results = db.retrieve(query["positions"], query["quaternions"], k=min(3, len(episodes)))
        print(f"[{name}] Retrieval top-{len(results)} distances:", [round(r[2], 4) for r in results])


def main() -> None:
    parser = argparse.ArgumentParser(description="DHB-XR VLA integration demo")
    parser.add_argument("--robocasa", type=str, default="", help="Path to RoboCASA HDF5 dataset")
    parser.add_argument("--libero", type=str, default="", help="Path to Libero HDF5 dataset")
    parser.add_argument("--max-episodes", type=int, default=10, help="Max episodes per dataset")
    args = parser.parse_args()

    if args.robocasa:
        run_pipeline("robocasa", Path(args.robocasa), args.max_episodes)
    if args.libero:
        run_pipeline("libero", Path(args.libero), args.max_episodes)
    if not args.robocasa and not args.libero:
        raise SystemExit("Provide at least one dataset path via --robocasa or --libero.")


if __name__ == "__main__":
    main()
