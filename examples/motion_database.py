"""
Motion Database Example

This example demonstrates the DHB-XR motion database for trajectory similarity search.
The database stores trajectories as DHB invariants and supports:

- Efficient similarity search using L2 distance on invariants
- Optional FAISS for GPU-accelerated nearest neighbor search
- Dynamic Time Warping (DTW) for variable-length trajectories
- Metadata storage and retrieval

Applications:
- Motion planning: find similar trajectories from past experience
- Imitation learning: retrieve demonstrations by motion similarity
- Trajectory libraries: organize and search motion collections

The example:
1. Creates 4 synthetic trajectories with varying randomness
2. Adds them to the motion database
3. Queries for the 3 most similar trajectories to a new query
"""

import numpy as np
from dhb_xr.database.motion_db import MotionDatabase


def run_example():
    """Run the motion database example."""
    print("DHB-XR Motion Database Example")
    print("=" * 35)

    # Initialize database
    np.random.seed(3)
    n = 35
    db = MotionDatabase(dhb_method="double_reflection", use_faiss=False)

    print(f"Creating motion database with {n}-pose trajectories...")

    # Add trajectories to database
    trajectories_added = []
    for i in range(4):
        # Create trajectory with increasing randomness
        positions = np.cumsum(np.random.randn(n, 3) * (0.01 + i * 0.005), axis=0)
        quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))

        metadata = {"id": i, "label": f"traj_{i}", "randomness": 0.01 + i * 0.005}
        db.add(positions, quaternions, metadata=metadata)

        trajectories_added.append({
            'start': positions[0],
            'end': positions[-1],
            'metadata': metadata
        })

    print(f"✅ Added {len(trajectories_added)} trajectories to database")

    # Create query trajectory
    print("\nCreating query trajectory...")
    q_pos = np.cumsum(np.random.randn(n, 3) * 0.012, axis=0)
    q_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))

    print(f"Query trajectory: {q_pos[0]} → {q_pos[-1]}")

    # Retrieve similar trajectories
    print("\nSearching for similar trajectories...")
    results = db.retrieve(q_pos, q_quat, k=3, use_dtw=False)

    print("Top 3 matches (by L2 invariant distance):")
    print("-" * 45)
    for rank, (invariants, metadata, distance) in enumerate(results, 1):
        traj_info = trajectories_added[metadata['id']]
        print(f"{rank}. Distance: {distance:.4f}")
        print(f"   Trajectory: {metadata['label']}")
        print(f"   Start: {traj_info['start']}")
        print(f"   End:   {traj_info['end']}")
        print(f"   Randomness: {metadata['randomness']:.3f}")
        print()

    return db, results


def main():
    """Main function for command-line execution."""
    run_example()


if __name__ == "__main__":
    main()
