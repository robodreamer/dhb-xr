"""Motion database: add trajectories, retrieve by invariant similarity."""

import numpy as np
from dhb_xr.encoder.dhb_dr import encode_dhb_dr
from dhb_xr.core.types import DHBMethod
from dhb_xr.database.motion_db import MotionDatabase

def main():
    np.random.seed(3)
    n = 35
    db = MotionDatabase(dhb_method="double_reflection", use_faiss=False)

    for i in range(4):
        positions = np.cumsum(np.random.randn(n, 3) * (0.01 + i * 0.005), axis=0)
        quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
        db.add(positions, quaternions, metadata={"id": i, "label": f"traj_{i}"})

    print("Added 4 trajectories to database.")
    q_pos = np.cumsum(np.random.randn(n, 3) * 0.012, axis=0)
    q_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    results = db.retrieve(q_pos, q_quat, k=3, use_dtw=False)
    print("Retrieved top 3 by L2 invariant distance:")
    for j, (inv, meta, dist) in enumerate(results):
        print(f"  {j+1}: dist={dist:.4f}, metadata={meta}")

if __name__ == "__main__":
    main()
