"""Script to add visualization improvements to the UMI-XDOF notebook."""

import json
from pathlib import Path

notebook_path = Path("notebooks/tutorial_umi_xdof_motion_analysis.ipynb")

with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Find the cell after encoding (cell 10 should be encoding, we want to add after cell 11)
# Let me find cell 11 content first
cells = nb['cells']

# Find the visualization cell and update it or add a new one after encoding
# Cell 11 should be the matplotlib import and plotting

# New improved visualization cell to insert after encoding
improved_viz = {
    "cell_type": "code",
    "id": "improved_viz",
    "metadata": {},
    "source": [
        "# Improved visualizations using standardized helpers\n",
        "from dhb_xr.visualization.plot import plot_se3_trajectory, plot_invariants\n",
        "\n",
        "# Pick a sample trajectory for detailed visualization\n",
        "sample_idx = 0\n",
        "sample_pos, sample_quat = preprocessed_trajectories[sample_idx]\n",
        "sample_inv = invariants_list[sample_idx]\n",
        "\n",
        "fig = plt.figure(figsize=(14, 5))\n",
        "\n",
        "# 1. 3D trajectory with orientation cubes\n",
        "ax1 = fig.add_subplot(131, projection='3d')\n",
        "plot_se3_trajectory(\n",
        "    sample_pos, sample_quat,\n",
        "    ax=ax1,\n",
        "    title=f'Sample Trajectory ({meta[sample_idx][\"task\"]})',\n",
        "    show_orientation=True,\n",
        "    vis_type='cube',\n",
        "    num_frames=8,\n",
        "    color='blue',\n",
        ")\n",
        "\n",
        "# 2. Invariants time series\n",
        "ax2 = fig.add_subplot(132)\n",
        "ax3 = fig.add_subplot(133)\n",
        "\n",
        "# Linear invariants\n",
        "lin_inv = sample_inv[:, :4]\n",
        "for i in range(4):\n",
        "    ax2.plot(lin_inv[:, i], label=['mag', 'ex', 'ey', 'ez'][i], alpha=0.8)\n",
        "ax2.set_xlabel('Time step')\n",
        "ax2.set_ylabel('Linear invariants')\n",
        "ax2.set_title('Linear Motion Invariants')\n",
        "ax2.legend(loc='upper right', fontsize=8)\n",
        "ax2.grid(True, alpha=0.3)\n",
        "\n",
        "# Angular invariants\n",
        "ang_inv = sample_inv[:, 4:]\n",
        "for i in range(4):\n",
        "    ax3.plot(ang_inv[:, i], label=['mag', 'ex', 'ey', 'ez'][i], alpha=0.8)\n",
        "ax3.set_xlabel('Time step')\n",
        "ax3.set_ylabel('Angular invariants')\n",
        "ax3.set_title('Angular Motion Invariants')\n",
        "ax3.legend(loc='upper right', fontsize=8)\n",
        "ax3.grid(True, alpha=0.3)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ],
    "outputs": [],
    "execution_count": None
}

multi_task_viz = {
    "cell_type": "code",
    "id": "multi_task_viz",
    "metadata": {},
    "source": [
        "# Multi-task trajectory comparison\n",
        "from dhb_xr.visualization.plot import plot_se3_trajectories\n",
        "\n",
        "# Select one trajectory per task (first few tasks)\n",
        "task_samples = {}\n",
        "seen_tasks = set()\n",
        "for i, m in enumerate(meta[:20]):\n",
        "    task = m['task'].replace('pick_place_', '')\n",
        "    if task not in seen_tasks and len(seen_tasks) < 5:\n",
        "        pos, quat = preprocessed_trajectories[i]\n",
        "        task_samples[task] = {'positions': pos, 'quaternions': quat}\n",
        "        seen_tasks.add(task)\n",
        "\n",
        "fig = plt.figure(figsize=(10, 8))\n",
        "ax = fig.add_subplot(111, projection='3d')\n",
        "plot_se3_trajectories(\n",
        "    task_samples,\n",
        "    ax=ax,\n",
        "    show_orientation=True,\n",
        "    vis_type='cube',\n",
        "    num_frames=4,\n",
        "    title='Trajectories Across Different Tasks',\n",
        ")\n",
        "plt.show()"
    ],
    "outputs": [],
    "execution_count": None
}

# Find where to insert (after the existing visualization cell, around index 11-12)
# Insert after cell 11 (existing matplotlib import)
insert_idx = 12

# Insert the new cells
cells.insert(insert_idx, improved_viz)
cells.insert(insert_idx + 1, multi_task_viz)

nb['cells'] = cells

# Save the notebook
with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Added improved visualization cells at indices {insert_idx} and {insert_idx + 1}")
print(f"Total cells: {len(cells)}")
