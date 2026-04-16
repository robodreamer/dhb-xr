"""
P2 Failure Debug Plots
Generates 4-panel diagnostic figure for DHB-XR P2 evaluation failure (0/10 success).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import h5py
import os

# ─── Load data ────────────────────────────────────────────────────────────────
poses_file = '/tmp/dhb_spatial_p2_poses.npz'
demo_hdf5  = ('/home/andypark/Projects/data/libero/libero_spatial/'
              'pick_up_the_black_bowl_between_the_plate_and_the_ramekin_'
              'and_place_it_on_the_plate_demo.hdf5')

poses = np.load(poses_file, allow_pickle=True)

# Collect per-seed arrays
n_seeds    = 10
traj_pos   = [poses[f'seed{i}_pos']     for i in range(n_seeds)]   # (98,3)
traj_grip  = [poses[f'gripper_seed{i}'] for i in range(n_seeds)]   # (98,)  1/-1

def gripper_close_idx(grip):
    """First timestep where gripper is closed (>0)."""
    idx = np.where(grip > 0)[0]
    return int(idx[0]) if len(idx) > 0 else None

close_ts  = [gripper_close_idx(traj_grip[i]) for i in range(n_seeds)]
# Clamp to max 97 (trajectory length)
close_ts  = [min(ct, 97) if ct is not None else None for ct in close_ts]
close_pos = [traj_pos[i][close_ts[i]] if close_ts[i] is not None else None
             for i in range(n_seeds)]

# Reference positions (world/sim frame, XY plane)
BOWL_XY      = np.array([0.10,  0.21])
P2_PLATE_XY  = np.array([0.083, 0.042])
ORIG_PLATE_XY= np.array([0.073, 0.212])
BOWL_Z       = 0.86
P2_PLATE_Z   = 0.902

# Colors per seed
CMAP = plt.get_cmap('tab10')
seed_colors = [CMAP(i) for i in range(n_seeds)]

# ─── Figure setup ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('P2 Failure Debug: DHB-XR Predicted Trajectories (0/10 success)',
             fontsize=14, fontweight='bold')

# ══════════════════════════════════════════════════════════════════════════════
# Panel 1 — XY Top-Down View
# ══════════════════════════════════════════════════════════════════════════════
ax1 = axes[0, 0]

for i in range(n_seeds):
    pos = traj_pos[i]
    c   = seed_colors[i]
    ax1.plot(pos[:, 0], pos[:, 1], color=c, alpha=0.6, linewidth=1.2, label=f'seed{i}')
    ax1.plot(pos[0, 0], pos[0, 1], 'o', color=c, markersize=7)        # start
    ax1.plot(pos[-1, 0], pos[-1, 1], 'x', color=c, markersize=9, markeredgewidth=2)  # end
    ct = close_ts[i]
    if ct is not None:
        ax1.plot(pos[ct, 0], pos[ct, 1], '*', color='red', markersize=12, zorder=5)

# Reference landmarks
ax1.plot(BOWL_XY[0], BOWL_XY[1], 'o', color='limegreen', markersize=18,
         markeredgecolor='darkgreen', markeredgewidth=2, zorder=6, label='BOWL (pick)')
ax1.annotate('BOWL\n(pick)', BOWL_XY, textcoords='offset points', xytext=(8, 5),
             fontsize=9, color='darkgreen', fontweight='bold')

ax1.plot(P2_PLATE_XY[0], P2_PLATE_XY[1], 's', color='royalblue', markersize=16,
         markeredgecolor='navy', markeredgewidth=2, zorder=6, label='P2 PLATE (place)')
ax1.annotate('P2 PLATE\n(place)', P2_PLATE_XY, textcoords='offset points', xytext=(5, -20),
             fontsize=9, color='navy', fontweight='bold')

ax1.plot(ORIG_PLATE_XY[0], ORIG_PLATE_XY[1], 's', color='lightgray', markersize=14,
         markeredgecolor='gray', markeredgewidth=2, zorder=6, label='orig plate')
ax1.annotate('orig plate', ORIG_PLATE_XY, textcoords='offset points', xytext=(5, 5),
             fontsize=8, color='gray')

# Success zone box around P2 plate
rect = mpatches.FancyBboxPatch((0.04, 0.00), 0.08, 0.08,
                                boxstyle='square,pad=0.005',
                                linewidth=2, edgecolor='royalblue',
                                facecolor='none', linestyle='--', zorder=4)
ax1.add_patch(rect)
ax1.text(0.08, -0.015, 'success zone', ha='center', fontsize=8, color='royalblue')

# Stars legend
ax1.plot([], [], '*', color='red', markersize=10, label='gripper close (★)')
ax1.plot([], [], 'o', color='gray', markersize=6, label='traj start (○)')
ax1.plot([], [], 'x', color='gray', markersize=8, markeredgewidth=2, label='traj end (×)')

ax1.set_xlabel('X position (m)')
ax1.set_ylabel('Y position (m)')
ax1.set_title('XY Top-Down: Predicted Trajectories (all 10 seeds)')
ax1.legend(fontsize=6, loc='upper right', ncol=2)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# ══════════════════════════════════════════════════════════════════════════════
# Panel 2 — Y-position vs timestep
# ══════════════════════════════════════════════════════════════════════════════
ax2 = axes[0, 1]
timesteps = np.arange(98)

for i in range(n_seeds):
    pos = traj_pos[i]
    c   = seed_colors[i]
    ax2.plot(timesteps, pos[:, 1], color=c, alpha=0.7, linewidth=1.2, label=f'seed{i}')
    ct = close_ts[i]
    if ct is not None:
        ax2.axvline(ct, color=c, alpha=0.3, linewidth=0.8, linestyle=':')
        ax2.plot(ct, pos[ct, 1], '*', color='red', markersize=8, zorder=5)

# Reference horizontal lines
ax2.axhline(0.042,  color='royalblue', linewidth=2, linestyle='--', label='P2 plate Y=0.042', alpha=0.8)
ax2.axhline(0.212,  color='gray',      linewidth=1.5, linestyle='--', label='orig plate Y=0.212', alpha=0.8)
ax2.axhline(0.21,   color='limegreen', linewidth=2,   linestyle='--', label='bowl pick Y=0.21', alpha=0.8)

ax2.set_xlabel('Timestep')
ax2.set_ylabel('Y position (m)')
ax2.set_title('Y-position over time (plate target = Y=0.042)')
ax2.legend(fontsize=7, loc='upper right')
ax2.grid(True, alpha=0.3)

# Annotate the failure: most seeds close gripper at wrong Y
ax2.text(2, 0.055, '← gripper closes here\n  (wrong region)', fontsize=7, color='red', alpha=0.8)

# ══════════════════════════════════════════════════════════════════════════════
# Panel 3 — Gripper state heatmap + expected GRASP window
# ══════════════════════════════════════════════════════════════════════════════
ax3 = axes[1, 0]

# Build gripper matrix: 10 x 98, values 0 (open) / 1 (closed)
# Gripper arrays may have variable length — pad/clip to 98
grip_matrix = np.zeros((n_seeds, 98))
for i in range(n_seeds):
    g = (traj_grip[i] > 0).astype(float)
    L = min(len(g), 98)
    grip_matrix[i, :L] = g[:L]

# Get expected GRASP timesteps from GT demos
expected_close_steps = []
try:
    with h5py.File(demo_hdf5, 'r') as f:
        for demo_key in f['data'].keys():
            actions = f['data'][demo_key]['actions'][:]
            close_idx = np.where(actions[:, 6] > 0.5)[0]
            if len(close_idx) > 0:
                expected_close_steps.append(close_idx[0])
    print(f"Loaded {len(expected_close_steps)} demo grasp steps: {expected_close_steps[:10]}")
except Exception as e:
    print(f"Warning: could not load demo HDF5: {e}")
    expected_close_steps = list(range(33, 42))  # fallback estimate

expected_close_steps = np.array(expected_close_steps)
gt_min = int(expected_close_steps.min()) if len(expected_close_steps) > 0 else 33
gt_max = int(expected_close_steps.max()) if len(expected_close_steps) > 0 else 42

# Plot heatmap
cmap_grip = ListedColormap(['#1a1a2e', '#e63946'])
im = ax3.imshow(grip_matrix, aspect='auto', cmap=cmap_grip,
                extent=[0, 97, n_seeds - 0.5, -0.5], interpolation='none')

# Expected GRASP window (green shaded vertical band)
ax3.axvspan(gt_min, gt_max, alpha=0.25, color='limegreen', zorder=2,
            label=f'GT GRASP window ({gt_min}–{gt_max})')

# Mark actual first-close timesteps
for i in range(n_seeds):
    ct = close_ts[i]
    if ct is not None:
        ax3.plot(ct, i, 'w*', markersize=10, zorder=5)

ax3.set_xlabel('Timestep')
ax3.set_ylabel('Seed')
ax3.set_yticks(range(n_seeds))
ax3.set_yticklabels([f'seed{i}' for i in range(n_seeds)])
ax3.set_title('Gripper timing: predicted (red) vs expected GRASP window (green)')

legend_handles = [
    mpatches.Patch(color='#1a1a2e', label='open'),
    mpatches.Patch(color='#e63946', label='closed (predicted)'),
    mpatches.Patch(color='limegreen', alpha=0.4, label=f'GT GRASP window ({gt_min}–{gt_max})'),
    plt.Line2D([0], [0], marker='*', color='white', markersize=8,
               markerfacecolor='white', linestyle='None', label='first close (★)'),
]
ax3.legend(handles=legend_handles, fontsize=7, loc='upper right')

# ══════════════════════════════════════════════════════════════════════════════
# Panel 4 — Final EE positions scatter
# ══════════════════════════════════════════════════════════════════════════════
ax4 = axes[1, 1]

final_pos = np.array([traj_pos[i][-1] for i in range(n_seeds)])  # (10, 3)

# Color by Z: too high = red, in range = green, too low = orange
z_colors = []
z_labels = []
for z in final_pos[:, 2]:
    if z > 0.95:
        z_colors.append('red')
        z_labels.append('too high (Z>0.95)')
    elif 0.88 <= z <= 0.95:
        z_colors.append('limegreen')
        z_labels.append('placement range [0.88-0.95]')
    else:
        z_colors.append('orange')
        z_labels.append('too low (Z<0.88)')

sc = ax4.scatter(final_pos[:, 0], final_pos[:, 1],
                 c=z_colors, s=120, zorder=5, edgecolors='k', linewidths=0.8)

# Label each seed
for i in range(n_seeds):
    ax4.annotate(f's{i}\nZ={final_pos[i,2]:.3f}',
                 (final_pos[i, 0], final_pos[i, 1]),
                 textcoords='offset points', xytext=(5, 5), fontsize=6)

# Success zone box: X=[0.04,0.12], Y=[0.00,0.08]
rect2 = mpatches.FancyBboxPatch((0.04, 0.00), 0.08, 0.08,
                                 boxstyle='square,pad=0.005',
                                 linewidth=2.5, edgecolor='royalblue',
                                 facecolor='lightyellow', alpha=0.3,
                                 linestyle='--', zorder=2)
ax4.add_patch(rect2)
ax4.text(0.08, 0.085, 'success zone\n(plate region)', ha='center',
         fontsize=8, color='royalblue', fontweight='bold')

# P2 plate marker
ax4.plot(P2_PLATE_XY[0], P2_PLATE_XY[1], 's', color='royalblue', markersize=14,
         markeredgecolor='navy', markeredgewidth=2, zorder=6, label='P2 PLATE')
ax4.plot(BOWL_XY[0], BOWL_XY[1], 'o', color='limegreen', markersize=12,
         markeredgecolor='darkgreen', markeredgewidth=2, zorder=6, label='BOWL')

# Custom legend for Z color coding
legend_handles4 = [
    mpatches.Patch(color='red',       label='too high (Z>0.95)'),
    mpatches.Patch(color='limegreen', label='placement range Z∈[0.88,0.95]'),
    mpatches.Patch(color='orange',    label='too low (Z<0.88)'),
    plt.Line2D([0], [0], marker='s', color='royalblue', markersize=9,
               markerfacecolor='royalblue', linestyle='None', label='P2 PLATE'),
    plt.Line2D([0], [0], marker='o', color='limegreen', markersize=9,
               markerfacecolor='limegreen', linestyle='None', label='BOWL'),
]
ax4.legend(handles=legend_handles4, fontsize=7, loc='upper right')

ax4.set_xlabel('X position (m)')
ax4.set_ylabel('Y position (m)')
ax4.set_title('Final EE positions (are seeds reaching the plate zone?)')
ax4.grid(True, alpha=0.3)
ax4.set_aspect('equal')

# ─── Save ─────────────────────────────────────────────────────────────────────
out_path = '/home/andypark/Projects/repos/dhb_xr/_workspace/p2_failure_debug.png'
plt.tight_layout()
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")

# ─── Print summary stats ───────────────────────────────────────────────────────
print("\n=== Summary Statistics ===")
print(f"GT expected GRASP window: t={gt_min}..{gt_max}")
print()
print(f"{'Seed':<6} {'Close@t':<9} {'Close_X':<10} {'Close_Y':<10} {'Close_Z':<10} {'Final_X':<10} {'Final_Y':<10} {'Final_Z':<10} {'In_zone?'}")
for i in range(n_seeds):
    ct = close_ts[i]
    if ct is not None:
        cx, cy, cz = traj_pos[i][ct]
    else:
        cx, cy, cz = float('nan'), float('nan'), float('nan')
    fx, fy, fz = final_pos[i]
    in_zone = (0.04 <= fx <= 0.12) and (0.00 <= fy <= 0.08) and (0.88 <= fz <= 0.92)
    print(f"seed{i}  {str(ct):<9} {cx:<10.4f} {cy:<10.4f} {cz:<10.4f} {fx:<10.4f} {fy:<10.4f} {fz:<10.4f} {'YES' if in_zone else 'no'}")
