#!/usr/bin/env python3
"""Generate adaptation_comparison.pdf for CoRL 2026 paper.

Two-panel XZ side view comparing DHB-XR (shape preserved) vs Affine (shape distorted).
"""

import sys
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, "/home/andypark/Projects/repos/dhb_xr/src")
from dhb_xr.encoder.dhb_dr import encode_dhb_dr
from dhb_xr.decoder.dhb_dr import decode_dhb_dr
from dhb_xr.core import geometry as geom
from dhb_xr.core.types import EncodingMethod

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import matplotlib.patheffects as pe

# ---------------------------------------------------------------------------
# 1. Generate synthetic trajectory (position-only for clarity)
# ---------------------------------------------------------------------------
N = 100
t = np.linspace(0, 1, N)
positions = np.column_stack([
    0.30 + 0.40 * t,
    0.10 * np.sin(np.pi * t),
    0.25 + 0.30 * np.sin(np.pi * t),
])

# Orientation: identity + small yaw ramp for visual interest
quats_wxyz = np.zeros((N, 4))
for i in range(N):
    r = Rotation.from_euler("z", 10 * t[i], degrees=True)
    q_xyzw = r.as_quat()
    quats_wxyz[i] = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

# ---------------------------------------------------------------------------
# 2. Encode with DHB-XR
# ---------------------------------------------------------------------------
enc = encode_dhb_dr(
    positions, quats_wxyz, method="pos", use_default_initial_frames=True
)

# ---------------------------------------------------------------------------
# 3. DHB-XR adaptation: decode from new start (offset only)
#    Shape is preserved exactly — the arc clears the obstacle.
# ---------------------------------------------------------------------------
offset = np.array([0.05, 0.0, 0.0])
new_start_pos = enc["initial_pose"]["position"] + offset

dhb_adapted = decode_dhb_dr(
    enc["linear_motion_invariants"],
    enc["angular_motion_invariants"],
    initial_pose={"position": new_start_pos,
                  "quaternion": enc["initial_pose"]["quaternion"]},
    method=EncodingMethod.POSITION,
    drop_padded=True,
)

# ---------------------------------------------------------------------------
# 4. Affine adaptation: map old start/goal -> new start/goal
#    New goal is 12 cm lower, so the affine rotation tilts the arc DOWNWARD,
#    reducing Z clearance and clipping the obstacle.
# ---------------------------------------------------------------------------
old_start = positions[0]
old_goal = positions[-1]
new_start = old_start + offset
new_goal = old_goal + np.array([0.05, 0.0, -0.12])  # goal 12 cm lower

old_vec = old_goal - old_start
new_vec = new_goal - new_start
scale = np.linalg.norm(new_vec) / np.linalg.norm(old_vec)

# Rotation from old_vec direction to new_vec direction (Rodrigues)
old_dir = old_vec / np.linalg.norm(old_vec)
new_dir = new_vec / np.linalg.norm(new_vec)
v = np.cross(old_dir, new_dir)
c = np.dot(old_dir, new_dir)
vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
R_affine = np.eye(3) + vx + vx @ vx / (1 + c)

affine_positions = scale * ((positions - old_start) @ R_affine.T) + new_start

affine_quats = np.zeros_like(quats_wxyz)
for i in range(N):
    old_r = Rotation.from_quat([quats_wxyz[i, 1], quats_wxyz[i, 2],
                                 quats_wxyz[i, 3], quats_wxyz[i, 0]])
    new_r = Rotation.from_matrix(R_affine) * old_r
    q_xyzw = new_r.as_quat()
    affine_quats[i] = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

# ---------------------------------------------------------------------------
# 5. Obstacle definition
# ---------------------------------------------------------------------------
obs_x0, obs_x1 = 0.46, 0.54
obs_z0, obs_z1 = 0.20, 0.42


def compute_clearance(pos, obs_x0, obs_x1, obs_z0, obs_z1):
    """Return positive clearance (above) or negative (clips), in mm."""
    mask = (pos[:, 0] >= obs_x0) & (pos[:, 0] <= obs_x1)
    if not np.any(mask):
        return 999.0
    min_z = pos[mask, 2].min()
    return (min_z - obs_z1) * 1000

dhb_pos = dhb_adapted["positions"]
clearance_dhb = compute_clearance(dhb_pos, obs_x0, obs_x1, obs_z0, obs_z1)
clearance_affine = compute_clearance(affine_positions, obs_x0, obs_x1, obs_z0, obs_z1)

print(f"DHB clearance: {clearance_dhb:.1f} mm")
print(f"Affine clearance: {clearance_affine:.1f} mm")

# ---------------------------------------------------------------------------
# 6. Drawing helpers
# ---------------------------------------------------------------------------
def draw_frame_triad(ax, pos_xz, rot_3x3, length=0.035, alpha=1.0, gray=False):
    """Draw RGB coordinate frame triad in XZ view.

    X-axis (red), Y-axis (green), Z-axis (blue).
    Projects all three 3D axes into the XZ plane.
    """
    origin = np.array(pos_xz)
    colors = ["#999999"] * 3 if gray else ["#d62728", "#2ca02c", "#1f77b4"]
    lw = 0.7 if gray else 1.2
    ms = 3.5 if gray else 5
    # Draw all three axes projected into XZ
    for col_idx, color in zip([0, 1, 2], colors):
        axis_3d = rot_3x3[:, col_idx]
        dx = axis_3d[0] * length
        dz = axis_3d[2] * length
        if abs(dx) < 1e-5 and abs(dz) < 1e-5:
            continue
        ax.annotate("", xy=(origin[0] + dx, origin[1] + dz),
                     xytext=origin,
                     arrowprops=dict(arrowstyle="-|>", color=color,
                                     lw=lw, mutation_scale=ms),
                     zorder=10)


def draw_panel(ax, orig_pos, adapted_pos, orig_quats, adapted_quats, title,
               clearance_mm, is_dhb=True):
    """Draw one panel of the comparison figure."""
    # Obstacle — prominent red rectangle
    rect = Rectangle((obs_x0, obs_z0), obs_x1 - obs_x0, obs_z1 - obs_z0,
                      linewidth=0.8, edgecolor="#a93226", facecolor="#e74c3c",
                      alpha=0.75, zorder=5)
    ax.add_patch(rect)
    ax.text((obs_x0 + obs_x1) / 2, (obs_z0 + obs_z1) / 2, "wall",
            ha="center", va="center", fontsize=4.5, color="white",
            fontweight="bold", zorder=6)

    # Original trajectory (gray dashed)
    ax.plot(orig_pos[:, 0], orig_pos[:, 2], color="#bbbbbb", ls="--", lw=0.7,
            zorder=3, label="Original")

    # Adapted trajectory
    traj_color = "#2471a3" if is_dhb else "#d35400"
    ax.plot(adapted_pos[:, 0], adapted_pos[:, 2], color=traj_color, lw=1.5,
            zorder=4, label="Adapted",
            path_effects=[pe.Stroke(linewidth=2.2, foreground="white"), pe.Normal()])

    # --- Coordinate frame triads ---
    # Original start/goal (gray)
    orig_R0 = geom.quat_to_rot(orig_quats[0])
    orig_RN = geom.quat_to_rot(orig_quats[-1])
    draw_frame_triad(ax, [orig_pos[0, 0], orig_pos[0, 2]], orig_R0, gray=True)
    draw_frame_triad(ax, [orig_pos[-1, 0], orig_pos[-1, 2]], orig_RN, gray=True)

    # Adapted start/goal (colored RGB)
    adp_R0 = geom.quat_to_rot(adapted_quats[0])
    adp_RN = geom.quat_to_rot(adapted_quats[-1])
    draw_frame_triad(ax, [adapted_pos[0, 0], adapted_pos[0, 2]], adp_R0)
    draw_frame_triad(ax, [adapted_pos[-1, 0], adapted_pos[-1, 2]], adp_RN)

    # --- Labels ---
    label_y_off = -0.04
    ax.text(orig_pos[0, 0] - 0.015, orig_pos[0, 2] + label_y_off, "S",
            ha="center", va="top", fontsize=5, color="#888888",
            fontstyle="italic")
    ax.text(orig_pos[-1, 0] + 0.02, orig_pos[-1, 2] + 0.01, "G",
            ha="left", va="bottom", fontsize=5, color="#888888",
            fontstyle="italic")
    ax.text(adapted_pos[0, 0] + 0.02, adapted_pos[0, 2] + label_y_off,
            r"$\mathbf{S'}$", ha="center", va="top", fontsize=5.5,
            color="black")
    # Place G' label — offset upward if goal is low (affine case)
    gx, gz = adapted_pos[-1, 0], adapted_pos[-1, 2]
    g_label_y = gz + 0.035 if gz < 0.18 else gz + label_y_off
    g_label_va = "bottom" if gz < 0.18 else "top"
    ax.text(gx + 0.02, g_label_y, r"$\mathbf{G'}$",
            ha="left", va=g_label_va, fontsize=5.5, color="black")

    # Title
    ax.set_title(title, fontsize=5.5, fontweight="bold", pad=4)

    # Status badge at bottom
    if clearance_mm >= 0:
        badge_text = f"Clears obstacle by {clearance_mm:.0f} mm"
        badge_color = "#27ae60"
        badge_symbol = u"\u2713"  # checkmark
    else:
        badge_text = f"CLIPS obstacle by {abs(clearance_mm):.0f} mm"
        badge_color = "#c0392b"
        badge_symbol = u"\u2717"  # X mark

    ax.text(0.5, 0.01, f"{badge_symbol}  {badge_text}", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=4.5, color="white",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=badge_color,
                      edgecolor="none", alpha=0.92))

    # Axis formatting
    ax.set_xlabel("x (m)", fontsize=5, labelpad=1)
    ax.set_ylabel("z (m)", fontsize=5, labelpad=1)
    ax.tick_params(labelsize=4.5, pad=1.5)
    ax.set_aspect("equal")
    ax.set_xlim(0.24, 0.80)
    ax.set_ylim(0.10, 0.62)
    ax.grid(True, lw=0.25, alpha=0.35, color="#cccccc")


# ---------------------------------------------------------------------------
# 7. Create figure
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 5.5,
    "axes.linewidth": 0.4,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "pdf.fonttype": 42,  # TrueType for editability
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.3, 1.8),
                                gridspec_kw={"wspace": 0.5})
fig.subplots_adjust(bottom=0.15, top=0.87, left=0.09, right=0.97)

# DHB-XR panel (left)
draw_panel(ax1, positions, dhb_pos, quats_wxyz, dhb_adapted["quaternions"],
           "DHB-XR (shape preserved)", clearance_dhb, is_dhb=True)

# Affine panel (right)
draw_panel(ax2, positions, affine_positions, quats_wxyz, affine_quats,
           "Affine transform (shape distorted)", clearance_affine, is_dhb=False)

out_path = "/home/andypark/Projects/repos/dhb_xr_paper/manuscript/figures/adaptation_comparison.pdf"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close()
