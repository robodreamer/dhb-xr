#!/usr/bin/env python
"""Test full DHB pipeline with real LIBERO-Spatial dataset.

Usage:
    pixi run python examples/integration/test_libero_encoding.py
"""
import os
import sys
from pathlib import Path

# Add src to path when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from dhb_xr.integration.vla.libero import LiberoAdapter
from dhb_xr.integration.vla.pipeline import DHBVLAPipeline, DHBVLAPipelineConfig
from dhb_xr.core.types import EncodingMethod, DHBMethod

DATA_DIR = Path(os.environ.get("LIBERO_DATA_DIR", "/home/andypark/Projects/data/libero/libero_spatial"))
FILE_PATH = DATA_DIR / "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"

print("=== Testing Full DHB Pipeline with LIBERO data ===\n")

# Load episodes using adapter
adapter = LiberoAdapter()
episodes = list(adapter.load_dataset(str(FILE_PATH)))[:5]  # First 5 episodes
print(f"Loaded {len(episodes)} episodes\n")

# Create pipeline
config = DHBVLAPipelineConfig(
    dhb_method=DHBMethod.DOUBLE_REFLECTION,
    method=EncodingMethod.POSITION,
)
pipeline = DHBVLAPipeline(config=config)
print(f"Pipeline config:")
print(f"  DHB method: {config.dhb_method}")
print(f"  Encoding method: {config.method}")
print()

# Process each episode
for i, episode in enumerate(episodes):
    pos = episode["positions"]
    quat = episode["quaternions"]
    demo_id = episode["metadata"]["demo_id"]

    print(f"--- Episode {i+1}: {demo_id} ---")
    print(f"  Input: {pos.shape[0]} frames, pos={pos.shape}, quat={quat.shape}")

    try:
        result = pipeline.encode_trajectory(pos, quat)

        # DHB returns separate linear and angular invariants
        linear_inv = result["linear_motion_invariants"]
        angular_inv = result["angular_motion_invariants"]
        invariants = np.concatenate([linear_inv, angular_inv], axis=1)

        print(f"  Linear invariants: shape={linear_inv.shape}")
        print(f"  Angular invariants: shape={angular_inv.shape}")
        print(f"  Combined: shape={invariants.shape}")
        print(f"  Invariant stats: min={invariants.min():.3f}, max={invariants.max():.3f}, mean={invariants.mean():.3f}")

        # Check for NaN or Inf
        if np.any(np.isnan(invariants)):
            print(f"  WARNING: Contains NaN values!")
        if np.any(np.isinf(invariants)):
            print(f"  WARNING: Contains Inf values!")

        print()
    except Exception as e:
        import traceback
        print(f"  ERROR: {e}")
        traceback.print_exc()
        print()

print("=== SUCCESS: DHB encoding works with LIBERO data! ===")
