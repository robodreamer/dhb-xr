#!/usr/bin/env python3
"""
Minimal isolated toy to reproduce and diagnose the EE rotation bug
in the DHB-XR decode/IK pipeline.

Steps:
  A. Load demo_0 from HDF5 -> GT EE trajectory (positions + quaternions)
  B. Encode to DHB invariants (encode_dhb_dr)
  C. Decode back (decode_dhb_dr) using GT initial pose
  D. Measure rotation at each step -- does GT roundtrip produce spinning?
  E. Decode using PREDICTED invariants from p2_aug_targets.npz
  F. Measure rotation -- does Fatrop trajectory produce spinning?
  G. Load predicted poses from /tmp/dhb_spatial_p2_poses.npz
  H. Measure rotation in those predicted poses directly

Produces: _workspace/debug_rotation_analysis.png
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from dhb_xr.encoder.dhb_dr import encode_dhb_dr
from dhb_xr.decoder.dhb_dr import decode_dhb_dr
from dhb_xr.core.types import DHBMethod, EncodingMethod

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def angular_velocity_deg(quats_wxyz):
    """Returns angular change in degrees between consecutive timesteps.
    Input: (T, 4) wxyz quaternions.
    """
    # wxyz -> xyzw for scipy
    quats_xyzw = quats_wxyz[:, [1, 2, 3, 0]]
    rotations = Rotation.from_quat(quats_xyzw)
    angles = []
    for t in range(1, len(rotations)):
        rel = rotations[t - 1].inv() * rotations[t]
        angle_deg = np.degrees(rel.magnitude())
        angles.append(angle_deg)
    return np.array(angles)


def print_stats(label, ang_per_step):
    """Print summary statistics for angular change per step."""
    print(f"\n--- {label} ---")
    print(f"  Steps       : {len(ang_per_step)}")
    print(f"  Mean         : {ang_per_step.mean():.2f} deg/step")
    print(f"  Median       : {np.median(ang_per_step):.2f} deg/step")
    print(f"  Max          : {ang_per_step.max():.2f} deg/step")
    print(f"  Min          : {ang_per_step.min():.2f} deg/step")
    n_spin = np.sum(ang_per_step > 5.0)
    print(f"  Steps > 5 deg: {n_spin} / {len(ang_per_step)}")
    n_big = np.sum(ang_per_step > 30.0)
    print(f"  Steps >30 deg: {n_big} / {len(ang_per_step)}")
    if n_spin > 0:
        first_spin = np.where(ang_per_step > 5.0)[0][0]
        print(f"  First >5 deg at step: {first_spin}")


# ===========================================================================
# Step A: Load GT demo_0
# ===========================================================================
print("=" * 70)
print("STEP A: Loading GT demo_0 from HDF5")
print("=" * 70)

import h5py

DEMO_HDF5 = (
    "/home/andypark/Projects/data/libero/libero_spatial/"
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_"
    "and_place_it_on_the_plate_demo.hdf5"
)

f = h5py.File(DEMO_HDF5, "r")
d = f["data/demo_0"]
rs = np.array(d["robot_states"])
ee_pos = rs[:, 2:5].astype(np.float64)
ee_quat_xyzw = rs[:, 5:9].astype(np.float64)  # xyzw from robosuite
ee_quat_wxyz = ee_quat_xyzw[:, [3, 0, 1, 2]]  # -> wxyz for DHB

print(f"  GT trajectory: {len(ee_pos)} timesteps")
print(f"  GT quat[0] (wxyz): {ee_quat_wxyz[0]}")
print(f"  GT quat norm range: [{np.linalg.norm(ee_quat_wxyz, axis=1).min():.6f}, "
      f"{np.linalg.norm(ee_quat_wxyz, axis=1).max():.6f}]")

gt_ang = angular_velocity_deg(ee_quat_wxyz)
print_stats("GT (raw) angular change", gt_ang)

# ===========================================================================
# Step B: Encode to DHB invariants
# ===========================================================================
print("\n" + "=" * 70)
print("STEP B: Encoding GT trajectory to DHB invariants")
print("=" * 70)

enc = encode_dhb_dr(
    ee_pos,
    ee_quat_wxyz,
    method="pos",
    use_default_initial_frames=True,
    dhb_method=DHBMethod.DOUBLE_REFLECTION,
)
lin_inv = enc["linear_motion_invariants"]
ang_inv = enc["angular_motion_invariants"]
print(f"  Linear invariants shape: {lin_inv.shape}")
print(f"  Angular invariants shape: {ang_inv.shape}")
print(f"  Angular inv range: [{ang_inv.min():.6f}, {ang_inv.max():.6f}]")
print(f"  Angular inv ch0 (magnitude) range: [{ang_inv[:, 0].min():.6f}, {ang_inv[:, 0].max():.6f}]")
print(f"  Angular inv ch1-3 (euler) range: [{ang_inv[:, 1:].min():.6f}, {ang_inv[:, 1:].max():.6f}]")

# ===========================================================================
# Step C: Decode back using GT initial pose
# ===========================================================================
print("\n" + "=" * 70)
print("STEP C: Decode roundtrip with GT initial pose")
print("=" * 70)

initial_pose = {
    "position": ee_pos[0].copy(),
    "quaternion": ee_quat_wxyz[0].copy(),
}
dec = decode_dhb_dr(
    lin_inv,
    ang_inv,
    initial_pose=initial_pose,
    method=EncodingMethod.POSITION,
    dhb_method=DHBMethod.DOUBLE_REFLECTION,
)
decoded_pos = dec["positions"]
decoded_quat_wxyz = dec["quaternions"]  # wxyz

print(f"  Decoded trajectory: {len(decoded_pos)} timesteps")
print(f"  Decoded quat[0] (wxyz): {decoded_quat_wxyz[0]}")

# ===========================================================================
# Step D: Measure rotation in GT roundtrip
# ===========================================================================
print("\n" + "=" * 70)
print("STEP D: GT roundtrip rotation analysis")
print("=" * 70)

gt_rt_ang = angular_velocity_deg(decoded_quat_wxyz)
print_stats("GT roundtrip angular change", gt_rt_ang)

# Also measure quat drift from GT
T = min(len(ee_quat_wxyz), len(decoded_quat_wxyz))
quat_diff = []
for t in range(T):
    q_gt = ee_quat_wxyz[t]
    q_dec = decoded_quat_wxyz[t]
    # Ensure same hemisphere
    if np.dot(q_gt, q_dec) < 0:
        q_dec = -q_dec
    r_gt = Rotation.from_quat(q_gt[[1, 2, 3, 0]])
    r_dec = Rotation.from_quat(q_dec[[1, 2, 3, 0]])
    diff_deg = np.degrees((r_gt.inv() * r_dec).magnitude())
    quat_diff.append(diff_deg)
quat_diff = np.array(quat_diff)
print(f"\n  Quat drift from GT:")
print(f"    Max: {quat_diff.max():.4f} deg")
print(f"    Mean: {quat_diff.mean():.4f} deg")

# ===========================================================================
# Step E: Decode using Fatrop-adapted invariants
# ===========================================================================
print("\n" + "=" * 70)
print("STEP E: Decode Fatrop-adapted invariants (p2_aug_targets.npz)")
print("=" * 70)

targets = np.load(
    "/home/andypark/Projects/repos/dhb_xr/_workspace/p2_aug_targets.npz",
    allow_pickle=True,
)
p2_targets = targets["p2_targets"]  # (50, 100, 8)
orig_targets = targets["orig_targets"]  # (50, 100, 8)
print(f"  p2_targets shape: {p2_targets.shape}")
print(f"  orig_targets shape: {orig_targets.shape}")

# Use demo 0 from orig_targets (should be close to GT invariants)
demo0_inv = orig_targets[0]  # (100, 8)
fatrop_lin_inv = demo0_inv[:, :4].astype(np.float64)
fatrop_ang_inv = demo0_inv[:, 4:].astype(np.float64)
print(f"  Demo 0 invariants shape: lin={fatrop_lin_inv.shape}, ang={fatrop_ang_inv.shape}")
print(f"  Fatrop angular inv range: [{fatrop_ang_inv.min():.6f}, {fatrop_ang_inv.max():.6f}]")
print(f"  Fatrop angular ch0 (mag): [{fatrop_ang_inv[:, 0].min():.6f}, {fatrop_ang_inv[:, 0].max():.6f}]")
print(f"  Fatrop angular ch1-3 (euler): [{fatrop_ang_inv[:, 1:].min():.6f}, {fatrop_ang_inv[:, 1:].max():.6f}]")

# Compare invariant magnitudes: GT encoded vs Fatrop
gt_ang_mag = np.abs(ang_inv[:, 0])
fatrop_ang_mag = np.abs(fatrop_ang_inv[:, 0])
print(f"\n  GT angular magnitude: mean={gt_ang_mag.mean():.6f}, max={gt_ang_mag.max():.6f}")
print(f"  Fatrop angular magnitude: mean={fatrop_ang_mag.mean():.6f}, max={fatrop_ang_mag.max():.6f}")

# Decode Fatrop invariants with GT initial pose
dec_fatrop = decode_dhb_dr(
    fatrop_lin_inv,
    fatrop_ang_inv,
    initial_pose=initial_pose,
    method=EncodingMethod.POSITION,
    dhb_method=DHBMethod.DOUBLE_REFLECTION,
)
fatrop_quat_wxyz = dec_fatrop["quaternions"]  # wxyz

# ===========================================================================
# Step F: Measure rotation in Fatrop decode
# ===========================================================================
print("\n" + "=" * 70)
print("STEP F: Fatrop trajectory rotation analysis")
print("=" * 70)

fatrop_ang_per_step = angular_velocity_deg(fatrop_quat_wxyz)
print_stats("Fatrop decode angular change", fatrop_ang_per_step)

# Also decode p2-adapted (seed 0)
p2_demo0_inv = p2_targets[0]  # (100, 8)
p2_lin_inv = p2_demo0_inv[:, :4].astype(np.float64)
p2_ang_inv = p2_demo0_inv[:, 4:].astype(np.float64)

dec_p2 = decode_dhb_dr(
    p2_lin_inv,
    p2_ang_inv,
    initial_pose=initial_pose,
    method=EncodingMethod.POSITION,
    dhb_method=DHBMethod.DOUBLE_REFLECTION,
)
p2_quat_wxyz = dec_p2["quaternions"]  # wxyz
p2_ang_per_step = angular_velocity_deg(p2_quat_wxyz)
print_stats("P2-adapted decode angular change", p2_ang_per_step)

# ===========================================================================
# Step G: Load predicted poses from stage 1 eval
# ===========================================================================
print("\n" + "=" * 70)
print("STEP G: Load DHBActionHead predicted poses")
print("=" * 70)

poses = np.load("/tmp/dhb_spatial_p2_poses.npz", allow_pickle=True)
pred_pos = poses["seed0_pos"]  # (98, 3)
pred_quat = poses["seed0_quat"]  # (98, 4) -- should be wxyz per eval script

print(f"  Predicted poses: {len(pred_pos)} timesteps")
print(f"  Predicted quat[0]: {pred_quat[0]}")
print(f"  Predicted quat norms: [{np.linalg.norm(pred_quat, axis=1).min():.6f}, "
      f"{np.linalg.norm(pred_quat, axis=1).max():.6f}]")

# Verify convention: the eval script stores wxyz. The w component (col 0)
# should be the largest for near-identity rotations.
# If col 1 is largest, it might be xyzw or the robot has a specific orientation.
print(f"\n  Checking quaternion convention:")
print(f"    Mean |col0|: {np.mean(np.abs(pred_quat[:, 0])):.4f}")
print(f"    Mean |col1|: {np.mean(np.abs(pred_quat[:, 1])):.4f}")
print(f"    Mean |col2|: {np.mean(np.abs(pred_quat[:, 2])):.4f}")
print(f"    Mean |col3|: {np.mean(np.abs(pred_quat[:, 3])):.4f}")
print(f"    GT wxyz col0 (w): {np.mean(np.abs(ee_quat_wxyz[:, 0])):.4f}")
print(f"    GT wxyz col1 (x): {np.mean(np.abs(ee_quat_wxyz[:, 1])):.4f}")

# ===========================================================================
# Step H: Measure rotation in predicted poses
# ===========================================================================
print("\n" + "=" * 70)
print("STEP H: Predicted poses rotation analysis")
print("=" * 70)

# The predicted quats are wxyz from the decode step in eval
pred_quat_wxyz = pred_quat  # wxyz
pred_ang_per_step = angular_velocity_deg(pred_quat_wxyz)
print_stats("Predicted poses (stage 1 output) angular change", pred_ang_per_step)

# ===========================================================================
# Additional diagnostic: check angular invariant magnitude comparison
# ===========================================================================
print("\n" + "=" * 70)
print("ADDITIONAL: Invariant magnitude comparison")
print("=" * 70)

# Compare GT invariants vs what would produce the predicted trajectory
# First check if the angular frame is drifting
print("\nDecoder angular frame evolution check (GT roundtrip):")
# Re-run decode manually to inspect angular frame
from dhb_xr.core import geometry as geom

angular_frame = np.eye(4)
init_quat_wxyz = ee_quat_wxyz[0]
angular_rot_temp = geom.quat_to_rot(init_quat_wxyz)

# Check angular_rot_temp determinant and orthogonality over time
print(f"  Initial angular_rot_temp det: {np.linalg.det(angular_rot_temp):.10f}")

for i in range(min(20, len(ang_inv))):
    rvec = angular_frame[:3, :3] @ np.array([ang_inv[i, 0], 0.0, 0.0])
    R_ang = geom.euler_to_rot(ang_inv[i, 1:4])
    angular_rot_temp = angular_rot_temp @ geom.axis_angle_to_rot(rvec).T
    angular_frame[:3, :3] = angular_frame[:3, :3] @ R_ang

    det = np.linalg.det(angular_rot_temp)
    orth_err = np.linalg.norm(angular_rot_temp @ angular_rot_temp.T - np.eye(3))
    quat_i = geom.rot_to_quat(angular_rot_temp)
    quat_norm = np.linalg.norm(quat_i)
    if i < 5 or i % 5 == 0:
        print(f"  Step {i:3d}: det={det:.10f}, orth_err={orth_err:.2e}, "
              f"quat_norm={quat_norm:.10f}, rvec_mag={np.linalg.norm(rvec):.6f}")

# Check the angular frame itself
print(f"\nAngular frame at step 20:")
print(f"  det: {np.linalg.det(angular_frame[:3, :3]):.10f}")
print(f"  orth_err: {np.linalg.norm(angular_frame[:3, :3] @ angular_frame[:3, :3].T - np.eye(3)):.2e}")

# ===========================================================================
# CRITICAL CHECK: Does decode accumulate quaternion without normalization?
# ===========================================================================
print("\n" + "=" * 70)
print("CRITICAL CHECK: Quaternion normalization in decoder")
print("=" * 70)

# Run decode in non-robust mode (default) and check quat norms
dec_check = decode_dhb_dr(
    lin_inv,
    ang_inv,
    initial_pose=initial_pose,
    method=EncodingMethod.POSITION,
    dhb_method=DHBMethod.DOUBLE_REFLECTION,
    robust_mode=False,
)
check_quats = dec_check["quaternions"]
check_norms = np.linalg.norm(check_quats, axis=1)
print(f"  Non-robust decode quat norms: min={check_norms.min():.10f}, max={check_norms.max():.10f}")
print(f"  Norm drift from 1.0: max={np.abs(check_norms - 1.0).max():.2e}")

# Now check with robust mode
dec_robust = decode_dhb_dr(
    lin_inv,
    ang_inv,
    initial_pose=initial_pose,
    method=EncodingMethod.POSITION,
    dhb_method=DHBMethod.DOUBLE_REFLECTION,
    robust_mode=True,
)
robust_quats = dec_robust["quaternions"]
robust_norms = np.linalg.norm(robust_quats, axis=1)
print(f"  Robust decode quat norms: min={robust_norms.min():.10f}, max={robust_norms.max():.10f}")

# ===========================================================================
# CHECK: angular_rot_temp @ axis_angle_to_rot(rvec).T -- is .T correct?
# ===========================================================================
print("\n" + "=" * 70)
print("CRITICAL CHECK: Is the .T (transpose) in angular rotation correct?")
print("=" * 70)

# In the encoder, the rotation diff is computed as:
#   R_rel = R_curr @ R_prev.T
#   rotation_diff[i] = rot_to_axis_angle(R_rel)
#
# In the decoder, accumulation is:
#   angular_rot_temp = angular_rot_temp @ axis_angle_to_rot(rvec).T
#
# This means: R_new = R_old @ R_rel.T = R_old @ R_prev @ R_curr.T
# But it should be: R_new = R_rel @ R_old = R_curr @ R_prev.T @ R_old
# OR:               R_new = R_old @ R_rel = R_old @ R_curr @ R_prev.T
#
# The .T makes it: R_new = R_old @ (R_curr @ R_prev.T).T = R_old @ R_prev @ R_curr.T
# That's WRONG if the intent is forward accumulation!
#
# Let's test both ways:

print("\nTest: Decode with .T vs without .T")
angular_frame_t = np.eye(4)
angular_frame_no_t = np.eye(4)
rot_with_t = geom.quat_to_rot(init_quat_wxyz).copy()
rot_no_t = geom.quat_to_rot(init_quat_wxyz).copy()

quats_with_t = [init_quat_wxyz.copy()]
quats_no_t = [init_quat_wxyz.copy()]

for i in range(min(len(ang_inv), len(ee_quat_wxyz) - 1)):
    # With .T (current decoder)
    rvec_t = angular_frame_t[:3, :3] @ np.array([ang_inv[i, 0], 0.0, 0.0])
    rot_with_t = rot_with_t @ geom.axis_angle_to_rot(rvec_t).T
    R_ang_t = geom.euler_to_rot(ang_inv[i, 1:4])
    angular_frame_t[:3, :3] = angular_frame_t[:3, :3] @ R_ang_t
    quats_with_t.append(geom.rot_to_quat(rot_with_t))

    # Without .T (proposed fix)
    rvec_no = angular_frame_no_t[:3, :3] @ np.array([ang_inv[i, 0], 0.0, 0.0])
    rot_no_t = rot_no_t @ geom.axis_angle_to_rot(rvec_no)
    angular_frame_no_t[:3, :3] = angular_frame_no_t[:3, :3] @ R_ang_t  # same frame update
    quats_no_t.append(geom.rot_to_quat(rot_no_t))

quats_with_t = np.array(quats_with_t)
quats_no_t = np.array(quats_no_t)

ang_with_t = angular_velocity_deg(quats_with_t)
ang_no_t = angular_velocity_deg(quats_no_t)

# Compare to GT
T_compare = min(len(ee_quat_wxyz), len(quats_with_t))
err_with_t = []
err_no_t = []
for t in range(T_compare):
    q_gt = ee_quat_wxyz[t]

    q_wt = quats_with_t[t]
    if np.dot(q_gt, q_wt) < 0:
        q_wt = -q_wt
    r_gt = Rotation.from_quat(q_gt[[1, 2, 3, 0]])
    r_wt = Rotation.from_quat(q_wt[[1, 2, 3, 0]])
    err_with_t.append(np.degrees((r_gt.inv() * r_wt).magnitude()))

    q_nt = quats_no_t[t]
    if np.dot(q_gt, q_nt) < 0:
        q_nt = -q_nt
    r_nt = Rotation.from_quat(q_nt[[1, 2, 3, 0]])
    err_no_t.append(np.degrees((r_gt.inv() * r_nt).magnitude()))

err_with_t = np.array(err_with_t)
err_no_t = np.array(err_no_t)

print(f"  With .T:    max_err_from_GT = {err_with_t.max():.4f} deg, mean = {err_with_t.mean():.4f} deg")
print(f"  Without .T: max_err_from_GT = {err_no_t.max():.4f} deg, mean = {err_no_t.mean():.4f} deg")
print(f"  With .T:    max_ang_step = {ang_with_t.max():.4f} deg/step")
print(f"  Without .T: max_ang_step = {ang_no_t.max():.4f} deg/step")

if err_no_t.max() < err_with_t.max():
    print("\n  >>> REMOVING .T IMPROVES GT roundtrip! Bug is the .T in decoder. <<<")
    fix_needed = "remove_transpose"
elif err_with_t.max() < 1.0:
    print("\n  >>> Both versions match GT well. Bug is NOT in decode roundtrip. <<<")
    fix_needed = "not_decode"
else:
    print("\n  >>> Neither version matches GT well. Deeper investigation needed. <<<")
    fix_needed = "deeper"

# ===========================================================================
# PLOT
# ===========================================================================
print("\n" + "=" * 70)
print("Generating plot...")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("DHB-XR EE Rotation Bug Diagnosis", fontsize=14, fontweight="bold")

# Panel 1: GT roundtrip angular change per step
ax = axes[0, 0]
ax.plot(gt_rt_ang, "b-", linewidth=0.8, label="GT roundtrip (with .T)")
ax.plot(ang_no_t, "r--", linewidth=0.8, alpha=0.7, label="GT roundtrip (without .T)")
ax.axhline(y=5.0, color="orange", linestyle=":", alpha=0.6, label="5 deg threshold")
ax.set_title("Panel 1: GT Roundtrip - Angular Change per Step")
ax.set_xlabel("Step")
ax.set_ylabel("Angular change (deg)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 2: Fatrop-adapted trajectory angular change per step
ax = axes[0, 1]
ax.plot(fatrop_ang_per_step, "g-", linewidth=0.8, label="Fatrop orig demo 0")
ax.plot(p2_ang_per_step, "m-", linewidth=0.8, alpha=0.7, label="Fatrop P2-adapted demo 0")
ax.axhline(y=5.0, color="orange", linestyle=":", alpha=0.6, label="5 deg threshold")
ax.axhline(y=30.0, color="red", linestyle=":", alpha=0.6, label="30 deg threshold")
ax.set_title("Panel 2: Fatrop Decode - Angular Change per Step")
ax.set_xlabel("Step")
ax.set_ylabel("Angular change (deg)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 3: Predicted poses angular change per step
ax = axes[1, 0]
ax.plot(pred_ang_per_step, "r-", linewidth=0.8, label="Stage 1 predicted poses")
ax.axhline(y=5.0, color="orange", linestyle=":", alpha=0.6, label="5 deg threshold")
ax.axhline(y=30.0, color="red", linestyle=":", alpha=0.6, label="30 deg (spinning)")
ax.set_title("Panel 3: DHBActionHead Predicted Poses - Angular Change")
ax.set_xlabel("Step")
ax.set_ylabel("Angular change (deg)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 4: Quaternion w component over time
ax = axes[1, 1]
ax.plot(ee_quat_wxyz[:, 0], "b-", linewidth=1.0, label="GT w")
ax.plot(decoded_quat_wxyz[:, 0], "b--", linewidth=0.8, alpha=0.6, label="GT roundtrip w (with .T)")
ax.plot(quats_no_t[:, 0], "r--", linewidth=0.8, alpha=0.6, label="GT roundtrip w (no .T)")
if len(fatrop_quat_wxyz) > 0:
    ax.plot(fatrop_quat_wxyz[:, 0], "g-", linewidth=0.8, alpha=0.6, label="Fatrop w")
ax.plot(pred_quat_wxyz[:, 0], "r-", linewidth=0.8, alpha=0.6, label="Predicted w")
ax.set_title("Panel 4: Quaternion w Component Over Time")
ax.set_xlabel("Timestep")
ax.set_ylabel("w (scalar part)")
ax.legend(fontsize=7, loc="best")
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "debug_rotation_analysis.png")
plt.savefig(out_path, dpi=150)
print(f"  Saved: {out_path}")

# ===========================================================================
# VERIFICATION SUMMARY
# ===========================================================================
print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

gt_max = gt_rt_ang.max()
fatrop_max = fatrop_ang_per_step.max()
p2_max = p2_ang_per_step.max()
pred_max = pred_ang_per_step.max()

status = lambda val, thresh: "PASS" if val < thresh else "FAIL"

print(f"  GT roundtrip (with .T):  max angular step = {gt_max:.1f} deg  "
      f"(target: <5 deg)  [{status(gt_max, 5)}]")
print(f"  GT roundtrip (no .T):    max angular step = {ang_no_t.max():.1f} deg  "
      f"(target: <5 deg)  [{status(ang_no_t.max(), 5)}]")
print(f"  Fatrop orig decode:      max angular step = {fatrop_max:.1f} deg  "
      f"(target: <10 deg) [{status(fatrop_max, 10)}]")
print(f"  Fatrop P2-adapted:       max angular step = {p2_max:.1f} deg  "
      f"(target: <10 deg) [{status(p2_max, 10)}]")
print(f"  Predicted poses (stage1):max angular step = {pred_max:.1f} deg  "
      f"(target: <15 deg) [{status(pred_max, 15)}]")

print(f"\n  GT error with .T:    max={err_with_t.max():.2f} deg, mean={err_with_t.mean():.2f} deg")
print(f"  GT error without .T: max={err_no_t.max():.2f} deg, mean={err_no_t.mean():.2f} deg")

# Diagnose
print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)

if gt_max > 5.0:
    print("  [BUG LOCATION] decode_dhb_dr itself -- orientation integration is broken")
    print(f"  GT roundtrip has {gt_max:.1f} deg/step max angular change (should be <5)")
elif fatrop_max > 10.0 or p2_max > 10.0:
    print("  [BUG LOCATION] P2-adapted invariant generation (gen_p2_aug_targets.py)")
    print(f"  Fatrop orig decode has {fatrop_max:.1f} deg/step (clean)")
    print(f"  P2-adapted decode has {p2_max:.1f} deg/step (SPINNING)")
    print()
    print("  ROOT CAUSE 1: Quaternion convention bug in gen_p2_aug_targets.py line 95:")
    print("    ee_quat = rs[:, 5:9][:, [1,2,3,0]]")
    print("    This gives [y,z,w,x] NOT [w,x,y,z] (wxyz).")
    print("    Correct: rs[:, 5:9][:, [3,0,1,2]]  (xyzw -> wxyz)")
    print()
    print("  ROOT CAUSE 2: IK stage ignores decoded orientation entirely")
    print("    eval_dhb_spatial_p2.py stage_ik() sets orientation_tolerance=100.0")
    print("    and never loads the decoded quaternions from the poses NPZ.")
    print("    The IK solver can produce wrist-flip solutions freely.")
    print()
    print("  FIX 1: Fix quaternion reorder in gen_p2_aug_targets.py:")
    print("    Line 95: rs[:, 5:9][:, [3,0,1,2]]  (not [1,2,3,0])")
    print("    Then regenerate p2_aug_targets.npz and retrain.")
    print()
    print("  FIX 2: Add orientation constraint to IK stage:")
    print("    - Load decoded quaternions from poses NPZ")
    print("    - Set orientation_tolerance to ~0.1 rad (~6 deg)")
    print("    - Use decoded quaternion as IK target rotation")
    print("    - This prevents wrist flips even with small orientation errors.")
elif pred_max > 15.0:
    print("  [BUG LOCATION] DHBActionHead angular invariant predictions")
    print(f"  Predicted poses have {pred_max:.1f} deg/step max angular change")
else:
    print("  [BUG LOCATION] Stage 2 (IK) or Stage 3 (controller)")
    print("  All decoded poses look clean. The spinning is introduced by IK wrist-flips")
    print("  or the JOINT_POSITION controller.")

# ===========================================================================
# ADDITIONAL: Verify the quaternion convention bug in gen_p2_aug_targets.py
# ===========================================================================
print("\n" + "=" * 70)
print("VERIFICATION: Quaternion convention bug in gen_p2_aug_targets.py")
print("=" * 70)

raw_quat = rs[0, 5:9]  # xyzw from robosuite
wrong_wxyz = raw_quat[[1, 2, 3, 0]]   # [y, z, w, x] -- gen_p2_aug_targets.py line 95
correct_wxyz = raw_quat[[3, 0, 1, 2]]  # [w, x, y, z] -- correct

print(f"  Raw HDF5 quat (xyzw): {raw_quat}")
print(f"  Wrong  reorder [1,2,3,0]: {wrong_wxyz}  <- gen_p2_aug_targets.py uses this")
print(f"  Correct reorder [3,0,1,2]: {correct_wxyz}  <- correct wxyz")
print()

# Verify: encode with wrong quaternions, check angular invariants are different
wrong_quats_wxyz = ee_quat_xyzw[:, [1, 2, 3, 0]]  # wrong reordering
try:
    enc_wrong = encode_dhb_dr(
        ee_pos, wrong_quats_wxyz, method="pos", use_default_initial_frames=True
    )
    wrong_ang_inv = enc_wrong["angular_motion_invariants"]
    print(f"  Angular inv with WRONG quats: range=[{wrong_ang_inv.min():.4f}, {wrong_ang_inv.max():.4f}]")
    print(f"  Angular inv with CORRECT quats: range=[{ang_inv.min():.4f}, {ang_inv.max():.4f}]")

    # Decode wrong invariants
    dec_wrong = decode_dhb_dr(
        enc_wrong["linear_motion_invariants"],
        wrong_ang_inv,
        initial_pose=initial_pose,
        method=EncodingMethod.POSITION,
        dhb_method=DHBMethod.DOUBLE_REFLECTION,
    )
    wrong_decoded_ang = angular_velocity_deg(dec_wrong["quaternions"])
    print(f"  Decode wrong invariants: max angular step = {wrong_decoded_ang.max():.1f} deg/step")
    print(f"  Decode correct invariants: max angular step = {gt_rt_ang.max():.1f} deg/step")
except Exception as e:
    print(f"  Error encoding with wrong quats: {e}")

# ===========================================================================
# ADDITIONAL: Check IK orientation usage
# ===========================================================================
print("\n" + "=" * 70)
print("IK STAGE ANALYSIS")
print("=" * 70)
print("  eval_dhb_spatial_p2.py stage_ik() configuration:")
print("    orientation_tolerance = 100.0  (effectively DISABLED)")
print("    orientation_gain = 0.1")
print("    Only loads ee_positions, NEVER loads ee_quaternions from poses NPZ")
print("    Uses FK rotation of current config as target rotation (line 474)")
print()
print("  Impact: IK has no orientation constraint, can produce wrist-flip")
print("  solutions that cause continuous spinning in joint space.")
print()
print("  Proposed fix for eval_dhb_spatial_p2.py stage_ik():")
print("    1. Load decoded quaternions: ee_quats = data[f'{key}_quat']")
print("    2. Use decoded quaternion as target rotation:")
print("       from scipy.spatial.transform import Rotation")
print("       target_rot = Rotation.from_quat(ee_quats[t, [1,2,3,0]]).as_matrix()")
print("    3. Set orientation_tolerance = 0.1  (6 degrees)")
print("    4. Set orientation_gain = 1.0")

f.close()
print("\nDone.")
