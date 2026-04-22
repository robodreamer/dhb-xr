"""End-to-end DHB-XR pipeline demo for RoboCASA and Libero.

This example uses the validated OAT streaming token contract by default, so the
example output matches the intended VLA chunk-execution use case rather than the
older VQ-only export path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dhb_xr.database.motion_db import MotionDatabase
from dhb_xr.integration.vla import RoboCASAAdapter, LiberoAdapter, DHBVLAPipeline
from dhb_xr.integration.vla.pipeline import DHBVLAPipelineConfig


def run_pipeline(
    name: str,
    dataset_path: Path,
    max_episodes: int = 10,
    tokenizer_kind: str = "oat",
) -> None:
    adapter = RoboCASAAdapter() if name == "robocasa" else LiberoAdapter()
    pipeline = DHBVLAPipeline(
        DHBVLAPipelineConfig(
            tokenizer_kind=tokenizer_kind,
            oat_num_registers=8,
            oat_emb_dim=64,
            oat_num_heads=4,
            oat_encoder_depth=2,
            oat_decoder_depth=2,
        )
    )

    episodes = []
    for i, ep in enumerate(adapter.load_dataset(str(dataset_path))):
        episodes.append(ep)
        if i + 1 >= max_episodes:
            break

    print(f"[{name}] Loaded {len(episodes)} episodes from {dataset_path}")

    outputs = pipeline.process_dataset(episodes)
    print(f"[{name}] Tokenized {len(outputs)} trajectories with {tokenizer_kind}")
    if outputs:
        print(f"[{name}] Example token shape: {outputs[0]['tokens'].shape}")
        print(f"[{name}] Example invariant shape: {outputs[0]['invariants'].shape}")
        if tokenizer_kind == "oat":
            streamed = pipeline.decode_token_stream(
                outputs[0]["tokens"],
                invariant_dim=outputs[0]["invariants"].shape[1],
                horizon=outputs[0]["invariants"].shape[0],
                prefix_lengths=[1, 2, len(outputs[0]["tokens"])],
            )
            for item in streamed:
                recon = item["reconstruction"][0]
                mse = float(((recon - outputs[0]["invariants"]) ** 2).mean())
                print(
                    f"[{name}] stream keep_k={item['keep_k']} "
                    f"reconstruction shape={recon.shape} mse={mse:.6f}"
                )

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
    parser.add_argument("--tokenizer-kind", type=str, default="oat", choices=["oat", "vq"])
    args = parser.parse_args()

    if args.robocasa:
        run_pipeline("robocasa", Path(args.robocasa), args.max_episodes, args.tokenizer_kind)
    if args.libero:
        run_pipeline("libero", Path(args.libero), args.max_episodes, args.tokenizer_kind)
    if not args.robocasa and not args.libero:
        raise SystemExit("Provide at least one dataset path via --robocasa or --libero.")


if __name__ == "__main__":
    main()
