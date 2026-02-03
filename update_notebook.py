"""Script to update the UMI-XDOF notebook with new sections."""

import json
from pathlib import Path

notebook_path = Path("notebooks/tutorial_umi_xdof_motion_analysis.ipynb")

with open(notebook_path, 'r') as f:
    nb = json.load(f)

# New cells to insert

# Section 2a: Trajectory Quality Diagnostics (after cell 4, which is Section 2 header)
diagnostics_markdown = {
    "cell_type": "markdown",
    "id": "diagnostics_section",
    "metadata": {},
    "source": [
        "## 2a. Trajectory Quality Diagnostics\n",
        "\n",
        "Before encoding, analyze trajectory quality to detect potential issues:\n",
        "- **180° direction reversals**: Can cause Euler singularities in frame transport\n",
        "- **Zero-motion segments**: Undefined tangent frames when the robot is stationary\n",
        "- **Initial direction misalignment**: Reconstruction error with default frames\n",
        "\n",
        "This section uses the new `analyze_trajectory()` function to assess each trajectory."
    ]
}

diagnostics_code = {
    "cell_type": "code",
    "id": "diagnostics_code",
    "metadata": {},
    "source": [
        "from dhb_xr.preprocessing import (\n",
        "    TrajectoryPreprocessor,\n",
        "    analyze_trajectory,\n",
        "    TrajectoryDiagnostics,\n",
        ")\n",
        "\n",
        "# Analyze all trajectories\n",
        "diagnostics_results = []\n",
        "for i, (pos, quat) in enumerate(trajectories):\n",
        "    diag = analyze_trajectory(pos, quat)\n",
        "    diagnostics_results.append({\n",
        "        'idx': i,\n",
        "        'task': meta[i]['task'],\n",
        "        'num_samples': diag.num_samples,\n",
        "        'trajectory_length': diag.trajectory_length,\n",
        "        'num_reversals': diag.num_reversals,\n",
        "        'num_zero_motion': diag.num_zero_motion_segments,\n",
        "        'alignment_angle_deg': np.degrees(diag.alignment_angle_to_x),\n",
        "        'is_well_conditioned': diag.is_well_conditioned,\n",
        "    })\n",
        "\n",
        "# Summary statistics\n",
        "total = len(diagnostics_results)\n",
        "well_conditioned = sum(1 for d in diagnostics_results if d['is_well_conditioned'])\n",
        "with_reversals = sum(1 for d in diagnostics_results if d['num_reversals'] > 0)\n",
        "with_zero_motion = sum(1 for d in diagnostics_results if d['num_zero_motion'] > 0)\n",
        "needs_alignment = sum(1 for d in diagnostics_results if d['alignment_angle_deg'] > 30)\n",
        "\n",
        "print(f\"Trajectory Quality Summary ({total} trajectories)\")\n",
        "print(f\"  Well-conditioned: {well_conditioned}/{total} ({100*well_conditioned/total:.1f}%)\")\n",
        "print(f\"  With reversals: {with_reversals}/{total}\")\n",
        "print(f\"  With zero-motion: {with_zero_motion}/{total}\")\n",
        "print(f\"  Needs alignment (>30°): {needs_alignment}/{total}\")"
    ],
    "outputs": [],
    "execution_count": None
}

diagnostics_viz_code = {
    "cell_type": "code",
    "id": "diagnostics_viz",
    "metadata": {},
    "source": [
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# Visualize diagnostics results\n",
        "fig, axes = plt.subplots(2, 2, figsize=(12, 8))\n",
        "\n",
        "# 1. Distribution of trajectory lengths\n",
        "lengths = [d['trajectory_length'] * 1000 for d in diagnostics_results]  # Convert to mm\n",
        "axes[0, 0].hist(lengths, bins=20, edgecolor='black', alpha=0.7)\n",
        "axes[0, 0].set_xlabel('Trajectory Length (mm)')\n",
        "axes[0, 0].set_ylabel('Count')\n",
        "axes[0, 0].set_title('Trajectory Length Distribution')\n",
        "\n",
        "# 2. Initial alignment angles\n",
        "angles = [d['alignment_angle_deg'] for d in diagnostics_results]\n",
        "axes[0, 1].hist(angles, bins=20, edgecolor='black', alpha=0.7, color='orange')\n",
        "axes[0, 1].axvline(30, color='red', linestyle='--', label='Alignment threshold (30°)')\n",
        "axes[0, 1].set_xlabel('Alignment Angle to +X (degrees)')\n",
        "axes[0, 1].set_ylabel('Count')\n",
        "axes[0, 1].set_title('Initial Direction Alignment')\n",
        "axes[0, 1].legend()\n",
        "\n",
        "# 3. Issues per trajectory\n",
        "issues = ['Reversals', 'Zero-motion', 'Misaligned (>30°)']\n",
        "counts = [with_reversals, with_zero_motion, needs_alignment]\n",
        "colors = ['red' if c > 5 else 'orange' if c > 0 else 'green' for c in counts]\n",
        "axes[1, 0].bar(issues, counts, color=colors, edgecolor='black')\n",
        "axes[1, 0].set_ylabel('Number of Trajectories')\n",
        "axes[1, 0].set_title('Detected Issues')\n",
        "\n",
        "# 4. Quality by task\n",
        "task_quality = {}\n",
        "for d in diagnostics_results:\n",
        "    task = d['task'].replace('pick_place_', 'PP_')\n",
        "    if task not in task_quality:\n",
        "        task_quality[task] = {'total': 0, 'good': 0}\n",
        "    task_quality[task]['total'] += 1\n",
        "    if d['is_well_conditioned']:\n",
        "        task_quality[task]['good'] += 1\n",
        "\n",
        "tasks = list(task_quality.keys())[:8]  # Limit to 8 for readability\n",
        "good_pct = [100 * task_quality[t]['good'] / task_quality[t]['total'] for t in tasks]\n",
        "axes[1, 1].barh(tasks, good_pct, color='green', alpha=0.7)\n",
        "axes[1, 1].set_xlabel('Well-conditioned (%)')\n",
        "axes[1, 1].set_title('Quality by Task')\n",
        "axes[1, 1].set_xlim(0, 100)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ],
    "outputs": [],
    "execution_count": None
}

preprocessing_markdown = {
    "cell_type": "markdown",
    "id": "preprocessing_section",
    "metadata": {},
    "source": [
        "### Preprocessing with TrajectoryPreprocessor\n",
        "\n",
        "For trajectories with detected issues, we can apply preprocessing:\n",
        "- **Align to +X**: Rotate so initial motion direction aligns with +x axis\n",
        "- **Gaussian smoothing**: Reduce sensor noise\n",
        "- **Interpolate zero-motion**: Replace stationary segments with interpolated motion\n",
        "- **Smooth reversals**: Apply local smoothing around sharp direction changes"
    ]
}

preprocessing_code = {
    "cell_type": "code",
    "id": "preprocessing_code",
    "metadata": {},
    "source": [
        "# Create preprocessor with standard settings for real-world data\n",
        "preprocessor = TrajectoryPreprocessor(\n",
        "    align_to_x=True,\n",
        "    smooth_reversals=True,\n",
        "    remove_zero_motion=True,\n",
        "    interpolate_zero_motion=True,  # Interpolate rather than remove\n",
        "    gaussian_sigma=0.5,  # Light smoothing\n",
        ")\n",
        "\n",
        "# Preprocess all trajectories\n",
        "preprocessed_trajectories = []\n",
        "alignment_rotations = []  # Store for decoding back to original frame\n",
        "\n",
        "for pos, quat in trajectories:\n",
        "    result = preprocessor.process(pos, quat, return_diagnostics=False)\n",
        "    preprocessed_trajectories.append((result.positions, result.quaternions))\n",
        "    alignment_rotations.append(result.alignment_rotation)\n",
        "\n",
        "print(f\"Preprocessed {len(preprocessed_trajectories)} trajectories\")\n",
        "\n",
        "# Re-analyze after preprocessing\n",
        "post_diagnostics = []\n",
        "for pos, quat in preprocessed_trajectories:\n",
        "    diag = analyze_trajectory(pos, quat)\n",
        "    post_diagnostics.append(diag.is_well_conditioned)\n",
        "\n",
        "improved = sum(post_diagnostics)\n",
        "print(f\"Well-conditioned after preprocessing: {improved}/{len(post_diagnostics)} ({100*improved/len(post_diagnostics):.1f}%)\")"
    ],
    "outputs": [],
    "execution_count": None
}

# Updated encoding cell (replaces cell 5)
updated_encoding = {
    "cell_type": "code",
    "id": "bcdf03d1",
    "metadata": {},
    "source": [
        "from dhb_xr.encoder.dhb_dr import encode_dhb_dr\n",
        "from dhb_xr.core.types import DHBMethod\n",
        "\n",
        "# Use preprocessed trajectories for encoding\n",
        "invariants_list = []\n",
        "initial_poses = []  # Store for decoding\n",
        "\n",
        "for pos, quat in preprocessed_trajectories:\n",
        "    out = encode_dhb_dr(\n",
        "        pos, quat,\n",
        "        method=\"pos\",\n",
        "        use_default_initial_frames=True,\n",
        "        dhb_method=DHBMethod.DOUBLE_REFLECTION,\n",
        "        robust_mode=True,  # Enable robustness features\n",
        "        return_diagnostics=True,\n",
        "    )\n",
        "    U = np.concatenate([\n",
        "        out[\"linear_motion_invariants\"],\n",
        "        out[\"angular_motion_invariants\"],\n",
        "    ], axis=1)\n",
        "    invariants_list.append(U.astype(np.float32))\n",
        "    initial_poses.append(out[\"initial_pose\"])\n",
        "\n",
        "print(f\"Encoded {len(invariants_list)} trajectories\")\n",
        "print(f\"Invariant shape per trajectory: {invariants_list[0].shape} (T, 8 for linear+angular)\")"
    ],
    "outputs": [],
    "execution_count": None
}

# VFM section (add after Section 6)
vfm_markdown = {
    "cell_type": "markdown",
    "id": "vfm_section",
    "metadata": {},
    "source": [
        "## 7. Variational Flow Matching for Multi-Modal Generation\n",
        "\n",
        "**Variational Flow Matching (VFM)** extends standard flow matching with a latent variable to capture multi-modal trajectory distributions:\n",
        "\n",
        "- **Multi-modality**: Sample diverse trajectories for the same task by varying the latent code\n",
        "- **Mode separation**: Categorical latent selects among discrete motion strategies\n",
        "- **Smooth interpolation**: Gaussian latent allows continuous variation within modes\n",
        "\n",
        "This section demonstrates VFM on the UMI-XDOF invariants."
    ]
}

vfm_code = {
    "cell_type": "code",
    "id": "vfm_demo_code",
    "metadata": {},
    "source": [
        "# VFM requires torch\n",
        "HAS_VFM = False\n",
        "try:\n",
        "    import torch\n",
        "    from dhb_xr.generative import (\n",
        "        VelocityNetwork,\n",
        "        VariationalFlowMatcher,\n",
        "        LatentEncoder,\n",
        "        euler_solve,\n",
        "    )\n",
        "    HAS_VFM = True\n",
        "    print(\"VFM modules loaded successfully\")\n",
        "except ImportError as e:\n",
        "    print(f\"VFM not available: {e}\")"
    ],
    "outputs": [],
    "execution_count": None
}

vfm_train_code = {
    "cell_type": "code",
    "id": "vfm_train",
    "metadata": {},
    "source": [
        "if HAS_VFM:\n",
        "    # Prepare data: pad/truncate invariants to fixed length\n",
        "    MAX_LEN = 250\n",
        "    DIM = 8  # 4 linear + 4 angular invariants\n",
        "    \n",
        "    # Pad/truncate and stack\n",
        "    padded_data = []\n",
        "    for inv in invariants_list:\n",
        "        if len(inv) >= MAX_LEN:\n",
        "            padded = inv[:MAX_LEN]\n",
        "        else:\n",
        "            pad_len = MAX_LEN - len(inv)\n",
        "            padded = np.pad(inv, ((0, pad_len), (0, 0)), mode='edge')\n",
        "        padded_data.append(padded.flatten())\n",
        "    \n",
        "    X = torch.tensor(np.stack(padded_data), dtype=torch.float32)\n",
        "    print(f\"Training data shape: {X.shape}\")\n",
        "    \n",
        "    # Create VFM model\n",
        "    data_dim = X.shape[1]\n",
        "    latent_dim = 8\n",
        "    hidden_dim = 128\n",
        "    \n",
        "    velocity_net = VelocityNetwork(\n",
        "        data_dim=data_dim,\n",
        "        hidden_dim=hidden_dim,\n",
        "        conditioning_dim=latent_dim,\n",
        "        num_layers=3,\n",
        "    )\n",
        "    \n",
        "    latent_encoder = LatentEncoder(\n",
        "        input_dim=data_dim,\n",
        "        latent_dim=latent_dim,\n",
        "        hidden_dim=64,\n",
        "    )\n",
        "    \n",
        "    vfm = VariationalFlowMatcher(\n",
        "        velocity_network=velocity_net,\n",
        "        latent_encoder=latent_encoder,\n",
        "        data_dim=data_dim,\n",
        "        latent_dim=latent_dim,\n",
        "    )\n",
        "    \n",
        "    print(f\"VFM model created: {sum(p.numel() for p in vfm.parameters())} parameters\")"
    ],
    "outputs": [],
    "execution_count": None
}

vfm_train_loop = {
    "cell_type": "code",
    "id": "vfm_train_loop",
    "metadata": {},
    "source": [
        "if HAS_VFM:\n",
        "    # Quick training demo (few epochs)\n",
        "    optimizer = torch.optim.AdamW(vfm.parameters(), lr=1e-3)\n",
        "    vfm.train()\n",
        "    \n",
        "    num_epochs = 50\n",
        "    batch_size = min(32, len(X))\n",
        "    \n",
        "    losses = []\n",
        "    for epoch in range(num_epochs):\n",
        "        # Random batch\n",
        "        idx = torch.randperm(len(X))[:batch_size]\n",
        "        batch = X[idx]\n",
        "        \n",
        "        optimizer.zero_grad()\n",
        "        loss, loss_dict = vfm.loss(batch, kl_weight=0.01)\n",
        "        loss.backward()\n",
        "        optimizer.step()\n",
        "        \n",
        "        losses.append(loss.item())\n",
        "        if (epoch + 1) % 10 == 0:\n",
        "            print(f\"Epoch {epoch+1}: loss={loss.item():.4f}, flow={loss_dict['flow_loss']:.4f}, kl={loss_dict['kl_loss']:.4f}\")\n",
        "    \n",
        "    print(f\"\\nTraining complete. Final loss: {losses[-1]:.4f}\")"
    ],
    "outputs": [],
    "execution_count": None
}

vfm_sample_code = {
    "cell_type": "code",
    "id": "vfm_sample",
    "metadata": {},
    "source": [
        "if HAS_VFM:\n",
        "    # Generate samples with different latent codes\n",
        "    vfm.eval()\n",
        "    \n",
        "    # Multi-modal sampling: generate 5 samples with different latent codes\n",
        "    with torch.no_grad():\n",
        "        samples = vfm.sample_multimodal(n_samples=5, n_modes=5, steps=50)\n",
        "    \n",
        "    print(f\"Generated {len(samples)} samples, shape: {samples[0].shape}\")\n",
        "    \n",
        "    # Reshape back to trajectory format\n",
        "    sample_trajectories = []\n",
        "    for s in samples:\n",
        "        traj_inv = s.reshape(MAX_LEN, DIM).numpy()\n",
        "        sample_trajectories.append(traj_inv)\n",
        "    \n",
        "    # Visualize first invariant component across samples\n",
        "    fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n",
        "    \n",
        "    # Plot linear magnitude (first component)\n",
        "    for i, traj in enumerate(sample_trajectories[:5]):\n",
        "        axes[0].plot(traj[:, 0], alpha=0.7, label=f'Sample {i+1}')\n",
        "    axes[0].set_xlabel('Time step')\n",
        "    axes[0].set_ylabel('Linear magnitude')\n",
        "    axes[0].set_title('Generated Trajectories (Linear Invariant)')\n",
        "    axes[0].legend()\n",
        "    \n",
        "    # Plot angular magnitude (fifth component)\n",
        "    for i, traj in enumerate(sample_trajectories[:5]):\n",
        "        axes[1].plot(traj[:, 4], alpha=0.7, label=f'Sample {i+1}')\n",
        "    axes[1].set_xlabel('Time step')\n",
        "    axes[1].set_ylabel('Angular magnitude')\n",
        "    axes[1].set_title('Generated Trajectories (Angular Invariant)')\n",
        "    axes[1].legend()\n",
        "    \n",
        "    plt.tight_layout()\n",
        "    plt.show()"
    ],
    "outputs": [],
    "execution_count": None
}

vfm_summary = {
    "cell_type": "markdown",
    "id": "vfm_summary",
    "metadata": {},
    "source": [
        "### VFM Summary\n",
        "\n",
        "| Feature | Description |\n",
        "|---------|-------------|\n",
        "| **Latent variable** | Enables multi-modal generation |\n",
        "| **Flow matching** | Simpler training than diffusion |\n",
        "| **Few-step sampling** | Fast generation (10-50 steps) |\n",
        "| **Mode selection** | Sample from specific trajectory types |\n",
        "\n",
        "For more details, see the demo notebook (`demo_dhb_features_and_use_cases.ipynb`)."
    ]
}

# Insert cells into notebook
cells = nb['cells']

# Insert diagnostics section after cell 4 (Section 2 header)
# We'll insert at index 5 (before current cell 5 which is encoding)
insert_idx = 5
new_cells = [
    diagnostics_markdown,
    diagnostics_code,
    diagnostics_viz_code,
    preprocessing_markdown,
    preprocessing_code,
]

# Replace cell 5 (encoding) with updated version - it will be at insert_idx + len(new_cells)
# Actually, we need to first insert, then the old cell 5 moves to insert_idx + len(new_cells)

# Build new cell list
new_cell_list = cells[:insert_idx] + new_cells
# Now add the updated encoding cell (replacing old cell 5)
new_cell_list.append(updated_encoding)
# Add remaining cells (from old cell 6 onwards)
new_cell_list.extend(cells[6:])

# Add VFM section at the end
new_cell_list.extend([
    vfm_markdown,
    vfm_code,
    vfm_train_code,
    vfm_train_loop,
    vfm_sample_code,
    vfm_summary,
])

nb['cells'] = new_cell_list

# Save the notebook
with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Updated notebook saved with {len(new_cell_list)} cells")
print(f"Added: {len(new_cells)} diagnostics/preprocessing cells")
print(f"Added: 6 VFM cells at the end")
