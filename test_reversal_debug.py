import numpy as np

n = 51
positions = np.zeros((n, 3))
positions[:26, 0] = np.linspace(0, 1, 26)
positions[26:, 0] = np.linspace(1 - 1/25, 0, 25)
print("Positions around midpoint:")
for i in range(23, 30):
    print(f"  {i}: {positions[i]}")

# Compute tangents
tangents = np.diff(positions, axis=0)
norms = np.linalg.norm(tangents, axis=1, keepdims=True)
tangents_normalized = tangents / (norms + 1e-10)

print("\nTangents around midpoint:")
for i in range(22, 28):
    t1 = tangents_normalized[i]
    t2 = tangents_normalized[i+1]
    dot = np.dot(t1, t2)
    print(f"  diff[{i}]: {tangents[i]} norm={norms[i,0]:.4f}")
    print(f"  diff[{i+1}]: {tangents[i+1]} norm={norms[i+1,0]:.4f}")
    print(f"  dot product: {dot:.6f}")
    print()

# Now test the actual detection function
from dhb_xr.preprocessing import detect_reversals
indices, min_dot = detect_reversals(positions, threshold=-0.5)
print(f"Detected reversals: {indices}")
print(f"Min dot product: {min_dot}")
