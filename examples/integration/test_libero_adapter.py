#!/usr/bin/env python
"""Test LiberoAdapter with real LIBERO-Spatial dataset.

Usage:
    pixi run python examples/integration/test_libero_adapter.py

Requirements:
    - LIBERO dataset downloaded to /home/andypark/Projects/data/libero/libero_spatial/
    - Or set LIBERO_DATA_DIR environment variable
"""
import os
import sys
from pathlib import Path

# Add src to path when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dhb_xr.integration.vla.libero import LiberoAdapter

DATA_DIR = Path(os.environ.get("LIBERO_DATA_DIR", "/home/andypark/Projects/data/libero/libero_spatial"))
FILE_PATH = DATA_DIR / "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"

print("=== Testing LiberoAdapter with real LIBERO data ===\n")

adapter = LiberoAdapter()
print(f"Adapter config:")
print(f"  Position keys: {adapter.pos_keys}")
print(f"  Quaternion keys: {adapter.quat_keys}")
print(f"  Robot states key: {adapter.robot_states_key}")
print(f"  Robot states quat slice: {adapter.robot_states_quat_slice}")
print(f"  Robot states quat format: {adapter.robot_states_quat_format}")
print()

count = 0
for episode in adapter.load_dataset(str(FILE_PATH)):
    count += 1
    if count <= 3:  # Show first 3 episodes
        print(f"Episode {count}: {episode['metadata']['demo_id']}")
        print(f"  Positions: shape={episode['positions'].shape}")
        print(f"  Quaternions: shape={episode['quaternions'].shape}")
        print(f"  Quat source: {episode['metadata']['quat_source']}")
        print(f"  First position: {episode['positions'][0]}")
        print(f"  First quaternion (x,y,z,w): {episode['quaternions'][0]}")
        print()

print(f"Total episodes loaded: {count}")
print("\n=== SUCCESS: LiberoAdapter works with real LIBERO data! ===")
