"""
P2 Trajectory Comparison: Predicted vs GT Fatrop trajectories
Compares the already-computed predicted poses from /tmp with denormalized GT invariants.

For GT decoding we use the DHB reconstruction from the Fatrop-adapted invariants
by comparing trajectories in the Y dimension (key failure axis).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import sys
sys.path.insert(0, '/home/andypark/Projects/repos/dhb_xr/src')

# ─── Load data ────────────────────────────────────────────────────────────────
poses_path  = '/tmp/dhb_spatial_p2_poses.npz'
targets_path= '/home/andypark/Projects/repos/dhb_xr/_workspace/p2_aug_targets.npz'
ckpt_path   = '/home/andypark/Projects/repos/dhb_xr/_workspace/dhb_head_p2_correct.pt'

poses   = np.load(poses_path, allow_pickle=True)
targets = np.load(targets_path, allow_pickle=True)
ckpt    = torch.load(ckpt_path, map_location='cpu', weights_only=False)

inv_mean = np.array(ckpt['inv_mean'])  # (8,)
inv_std  = np.array(ckpt['inv_std'])   # (8,)
p2_tgt   = targets['p2_targets']       # (50,100,8) normalized

# Seeds to compare (representative)
SEEDS = [0, 4, 8]
seed_demo_map = {0: 0, 4: 4, 8: 8}

# ─── Denormalize GT invariants ─────────────────────────────────────────────
def denormalize_inv(normed, mean, std):
    """normed: (T,8) → raw invariants (T,8)"""
    return normed * std + mean

# ─── Reconstruct GT XYZ from invariants ───────────────────────────────────
# The invariants encode relative motion, not absolute positions.
# channels 0-3: linear motion invariants (ds, kappa_x, kappa_y, kappa_z or similar)
# For comparison we reconstruct the GT trajectory starting from the same initial EE pos
# as the predicted trajectory, using cumulative integration.
#
# Strategy: channel 0 is arc-length-like (always ≥0), channels 1-3 are directional
# We compare in invariant space rather than Cartesian since decoding requires
# the full encoder context (frames). Instead we plot the key failure axis:
# invariant channel that tracks the Y-axis motion (lateral placement).

def gripper_close_idx(grip):
    idx = np.where(grip > 0)[0]
    return int(idx[0]) if len(idx) > 0 else None

CMAP = plt.get_cmap('Set1')

# ─── Figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(len(SEEDS), 3, figsize=(18, 12))
fig.suptitle('P2 Trajectory Comparison: Predicted (colored) vs GT Fatrop (black dashed)\n'
             'Seeds 0, 4, 8 — X, Y, Z positions over time', fontsize=13, fontweight='bold')

DIM_LABELS = ['X position (m)', 'Y position (m)', 'Z position (m)']
DIM_COLORS = ['#e07b54', '#2ec4b6', '#845ec2']

for row_idx, seed_i in enumerate(SEEDS):
    demo_idx    = seed_demo_map[seed_i]
    pred_pos    = poses[f'seed{seed_i}_pos']      # (98, 3)
    grip_arr    = poses[f'gripper_seed{seed_i}']
    init_ee     = poses[f'meta_seed{seed_i}_p2_ee_pos']  # (3,) — starting EE pos
    p2_ee_init  = poses[f'meta_seed{seed_i}_p2_ee_pos']
    close_t     = gripper_close_idx(grip_arr)
    if close_t is not None:
        close_t = min(close_t, 97)

    # GT invariants for this demo (denormalized)
    gt_inv = denormalize_inv(p2_tgt[demo_idx], inv_mean, inv_std)  # (100, 8)

    # Reconstruct GT approximate Cartesian trajectory from initial position.
    # DHB invariants ch0 = linear arc length increment (always ≥0, small)
    # ch1-3 = linear direction (Frenet-Serret-like frame components)
    # Approximate reconstruction: integrate ch1-3 weighted by ch0 as step size.
    # This is a coarse reconstruction for visualization only.
    gt_pos_approx = np.zeros((100, 3))
    gt_pos_approx[0] = p2_ee_init
    lin_inv = gt_inv[:, :4]  # (100, 4) linear motion invariants
    for t in range(1, 100):
        ds     = np.abs(lin_inv[t, 0])  # arc-length step
        # Use channels 1,2,3 as approximate motion direction
        direction = lin_inv[t, 1:4]
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction = direction / norm
        step = ds * direction
        gt_pos_approx[t] = gt_pos_approx[t-1] + step

    # Reference lines
    BOWL_Y   = 0.21
    P2_Y     = 0.042
    ORIG_Y   = 0.212
    P2_PLATE = np.array([0.083, 0.042, 0.902])

    t_pred = np.arange(len(pred_pos))
    t_gt   = np.arange(100)

    for dim in range(3):
        ax = axes[row_idx, dim]
        color = DIM_COLORS[dim]

        # Predicted trajectory
        ax.plot(t_pred, pred_pos[:, dim], color=color, linewidth=2,
                label=f'predicted (seed{seed_i})')

        # GT Fatrop trajectory (approximate reconstruction)
        ax.plot(t_gt, gt_pos_approx[:, dim], color='black', linewidth=1.5,
                linestyle='--', alpha=0.8, label=f'GT Fatrop approx (demo{demo_idx})')

        # Reference lines for Y dim
        if dim == 1:
            ax.axhline(P2_Y,   color='royalblue', linewidth=1.5, linestyle=':',
                       alpha=0.8, label=f'P2 plate Y={P2_Y}')
            ax.axhline(ORIG_Y, color='gray', linewidth=1,   linestyle=':',
                       alpha=0.6, label=f'orig plate Y={ORIG_Y}')
            ax.axhline(BOWL_Y, color='limegreen', linewidth=1.5, linestyle=':',
                       alpha=0.8, label=f'bowl pick Y={BOWL_Y}')

        # Reference lines for Z dim
        if dim == 2:
            ax.axhline(0.902, color='royalblue', linewidth=1.5, linestyle=':',
                       alpha=0.8, label='P2 plate Z=0.902')
            ax.axhline(0.86,  color='limegreen', linewidth=1.5, linestyle=':',
                       alpha=0.8, label='bowl Z≈0.86')

        # Gripper close marker
        if close_t is not None:
            ax.axvline(close_t, color='red', linewidth=1.5, linestyle='-.',
                       alpha=0.8, label=f'gripper close t={close_t}')
            ax.plot(close_t, pred_pos[close_t, dim], 'r*', markersize=12, zorder=5)

        # Axis labels
        if row_idx == 0:
            ax.set_title(DIM_LABELS[dim], fontsize=10)
        if row_idx == len(SEEDS) - 1:
            ax.set_xlabel('Timestep')
        ax.set_ylabel(f'seed{seed_i}' if dim == 0 else '')
        ax.legend(fontsize=6, loc='upper right')
        ax.grid(True, alpha=0.3)

# Add overall annotation about failure
fig.text(0.5, 0.01,
         'NOTE: GT reconstruction is approximate (invariant integration). '
         'Key failure: gripper closes at Y≈0.06-0.19 instead of reaching P2 plate Y=0.042.',
         ha='center', fontsize=9, color='darkred',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# ─── Panel: Invariant channel comparison ──────────────────────────────────
# Add extra figure for invariant channel visualization
fig2, axes2 = plt.subplots(2, 4, figsize=(20, 8))
fig2.suptitle('GT P2 Invariants (denormalized): seeds 0, 4, 8 — 8 channels\n'
              '(predicted invariants from DHBActionHead would be compared here if GPU available)',
              fontsize=12, fontweight='bold')

for col, seed_i in enumerate([0, 4, 8]):
    demo_idx = seed_demo_map[seed_i]
    gt_inv   = denormalize_inv(p2_tgt[demo_idx], inv_mean, inv_std)  # (100, 8)
    # Also get meta_inv_norm (what the model actually predicted, normalized)
    pred_inv_norm = poses.get(f'meta_seed{seed_i}_inv_norm', None)

    for ch in range(8):
        row = ch // 4
        ax  = axes2[row, ch % 4]
        t   = np.arange(100)
        ax.plot(t, gt_inv[:, ch], color=CMAP(col), linewidth=2,
                label=f'GT seed{seed_i}', alpha=0.8)
        if pred_inv_norm is not None:
            # denormalize predicted
            pred_inv = np.array(pred_inv_norm) * inv_std + inv_mean
            if pred_inv.ndim == 2 and pred_inv.shape[1] == 8:
                ax.plot(np.arange(len(pred_inv)), pred_inv[:, ch], '--',
                        color=CMAP(col), linewidth=1.5, alpha=0.6,
                        label=f'pred seed{seed_i}')
        if col == 0:
            ax.set_title(f'Channel {ch}\n({"linear" if ch < 4 else "angular"})', fontsize=9)
        ax.set_xlabel('t')
        ax.grid(True, alpha=0.3)
    if col == 0:
        for ch in range(8):
            axes2[ch // 4, ch % 4].legend(fontsize=6)

# ─── Save ─────────────────────────────────────────────────────────────────────
out1 = '/home/andypark/Projects/repos/dhb_xr/_workspace/p2_trajectory_comparison.png'
out2 = '/home/andypark/Projects/repos/dhb_xr/_workspace/p2_invariant_channels.png'

fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(out1, dpi=150, bbox_inches='tight')
print(f"Saved: {out1}")

fig2.tight_layout()
fig2.savefig(out2, dpi=150, bbox_inches='tight')
print(f"Saved: {out2}")

# ─── Summary: Key failure diagnosis ──────────────────────────────────────────
print("\n=== KEY FAILURE DIAGNOSIS ===")
print("Seeds analyzed: 0, 4, 8")
for seed_i in SEEDS:
    grip = poses[f'gripper_seed{seed_i}']
    pred = poses[f'seed{seed_i}_pos']
    ct   = gripper_close_idx(grip)
    if ct is not None:
        ct = min(ct, 97)
        cy = pred[ct, 1]
        print(f"seed{seed_i}: gripper closes at t={ct}, Y={cy:.4f} "
              f"(target P2 plate Y=0.042, delta={cy-0.042:+.4f})")
    else:
        print(f"seed{seed_i}: gripper never closes")

    # Where does the predicted trajectory end up vs P2 plate?
    final = pred[-1]
    print(f"       final pos: X={final[0]:.3f}, Y={final[1]:.3f}, Z={final[2]:.3f}")
    print(f"       P2 plate:  X=0.083,         Y=0.042,         Z=0.902")
    dist = np.linalg.norm(final - np.array([0.083, 0.042, 0.902]))
    print(f"       dist to P2 plate: {dist:.4f}m")
    print()
