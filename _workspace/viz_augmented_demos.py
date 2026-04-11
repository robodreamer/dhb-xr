#!/usr/bin/env python3
"""
Visualize DHB-augmented vs original vs naive-augmented trajectories.

Creates a figure showing:
- Original demo trajectory
- 9 DHB-augmented trajectories (different offsets)
- 9 naive-augmented trajectories
- Demonstrating shape preservation of DHB vs shape distortion of naive

Usage:
    pixi run python _workspace/viz_augmented_demos.py
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, "/home/andypark/Projects/repos/LIBERO")


def load_augmented_data(path="_workspace/augmented_demos_task0.npz"):
    d = np.load(path, allow_pickle=True)

    # Reconstruct individual demo actions from flat arrays
    orig_lengths = d["original_lengths"]
    orig_actions = d["original_actions"]
    dhb_lengths = d["dhb_augmented_lengths"]
    dhb_actions = d["dhb_augmented_actions"]
    naive_lengths = d["naive_augmented_lengths"]
    naive_actions = d["naive_augmented_actions"]
    offsets = d["offsets_used"]

    def split_by_lengths(flat, lengths):
        demos = []
        idx = 0
        for l in lengths:
            demos.append(flat[idx:idx+l])
            idx += l
        return demos

    return {
        "original": split_by_lengths(orig_actions, orig_lengths),
        "dhb": split_by_lengths(dhb_actions, dhb_lengths),
        "naive": split_by_lengths(naive_actions, naive_lengths),
        "offsets": offsets,
        "task_name": str(d["task_name"]),
        "n_augmented_per_demo": int(d["n_augmented_per_demo"]),
    }


def actions_to_trajectory(actions, start_pos=np.zeros(3)):
    """Convert action deltas to EE trajectory."""
    positions = [start_pos.copy()]
    for a in actions:
        positions.append(positions[-1] + a[:3])
    return np.array(positions)


def main():
    data = load_augmented_data()
    n_aug = data["n_augmented_per_demo"]

    # Take first demo and its augmentations
    orig_traj = actions_to_trajectory(data["original"][0])

    fig = plt.figure(figsize=(16, 6))

    # --- Panel 1: Original + DHB augmented ---
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(orig_traj[:, 0], orig_traj[:, 1], orig_traj[:, 2],
             'k-', linewidth=2, label='Original', zorder=10)
    ax1.scatter(orig_traj[0, 0], orig_traj[0, 1], orig_traj[0, 2],
                c='green', s=60, zorder=11)
    ax1.scatter(orig_traj[-1, 0], orig_traj[-1, 1], orig_traj[-1, 2],
                c='red', s=60, zorder=11)

    colors = plt.cm.Blues(np.linspace(0.3, 0.9, n_aug))
    for i in range(n_aug):
        offset = data["offsets"][i]
        dhb_traj = actions_to_trajectory(data["dhb"][i], start_pos=offset)
        ax1.plot(dhb_traj[:, 0], dhb_traj[:, 1], dhb_traj[:, 2],
                 '-', color=colors[i], linewidth=1, alpha=0.7)

    ax1.set_title('DHB-XR Augmented\n(shape preserved)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')

    # --- Panel 2: Original + Naive augmented ---
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot(orig_traj[:, 0], orig_traj[:, 1], orig_traj[:, 2],
             'k-', linewidth=2, label='Original', zorder=10)
    ax2.scatter(orig_traj[0, 0], orig_traj[0, 1], orig_traj[0, 2],
                c='green', s=60, zorder=11)
    ax2.scatter(orig_traj[-1, 0], orig_traj[-1, 1], orig_traj[-1, 2],
                c='red', s=60, zorder=11)

    colors_naive = plt.cm.Oranges(np.linspace(0.3, 0.9, n_aug))
    for i in range(n_aug):
        offset = data["offsets"][i]
        naive_traj = actions_to_trajectory(data["naive"][i], start_pos=offset)
        ax2.plot(naive_traj[:, 0], naive_traj[:, 1], naive_traj[:, 2],
                 '-', color=colors_naive[i], linewidth=1, alpha=0.7)

    ax2.set_title('Naive Offset Augmented\n(identical actions)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')

    # --- Panel 3: Stats ---
    ax3 = fig.add_subplot(133)
    ax3.axis('off')

    stats_text = f"""Task: {data['task_name'][:40]}...

Original demos: {len(data['original'])}
DHB augmented: {len(data['dhb'])}
Naive augmented: {len(data['naive'])}
Offset range: [-5, +5] cm per axis

Key difference:
- DHB preserves motion SHAPE
  (invariant encode-decode)
- Naive preserves action DELTAS
  (same relative motion)

For purely translational offsets,
both produce identical trajectories.
DHB's advantage emerges with:
- Obstacles (shape constraint)
- Rotational perturbations
- Non-uniform workspace geometry
"""
    ax3.text(0.1, 0.95, stats_text, transform=ax3.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('DHB-XR: Free SE(3) Data Augmentation from Single Demonstration',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('_workspace/augmentation_comparison.png', dpi=200, bbox_inches='tight')
    print("Saved: _workspace/augmentation_comparison.png")
    plt.close()


if __name__ == "__main__":
    main()
