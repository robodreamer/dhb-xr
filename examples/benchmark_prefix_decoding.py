"""
Prefix Decoding Benchmark and Visualization

Reproduces the key OAT visualization: show reconstruction quality as a
function of the number of decoded tokens (1, 2, 4, 8).

Demonstrates anytime decoding capability where early tokens capture
global structure and later tokens refine details.

Compares:
  - OAT-style (Register + FSQ + Nested Dropout)
  - RVQ (partial codebooks)
  - Hierarchical (truncated levels)
  - Progressive (truncated refinements)

Requires: PyTorch (`pip install dhb_xr[tokenization]`)
"""

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def generate_figure8_invariants(
    n_steps: int = 50, dim: int = 8, seed: int = 42
) -> np.ndarray:
    """Generate a figure-8 motion pattern as invariant-like data."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 4 * np.pi, n_steps)

    # Create structured invariants (magnitude + smooth angles)
    magnitude = 0.1 + 0.05 * np.sin(t)
    angles = np.column_stack([
        0.3 * np.sin(t),
        0.2 * np.cos(2 * t),
        0.1 * np.sin(3 * t),
    ])
    # Angular component (similar structure)
    mag_r = 0.05 + 0.02 * np.cos(t)
    angles_r = np.column_stack([
        0.15 * np.cos(t),
        0.1 * np.sin(2 * t),
        0.05 * np.cos(3 * t),
    ])

    inv = np.column_stack([magnitude, angles, mag_r, angles_r])
    inv += rng.randn(*inv.shape) * 0.005  # Small noise
    return inv.astype(np.float32)


def run_prefix_benchmark():
    """Run the prefix decoding benchmark."""
    print("=" * 70)
    print("Prefix Decoding Benchmark: Anytime Action Reconstruction")
    print("=" * 70)

    if not HAS_TORCH:
        print("\nPyTorch required. Install with: pip install dhb_xr[tokenization]")
        return

    # Generate test data
    inv = generate_figure8_invariants(n_steps=50)
    data = torch.from_numpy(inv).unsqueeze(0)  # (1, 50, 8)
    print(f"\nTest data: {data.shape} (batch x timesteps x dim)")

    results = {}
    prefix_lengths = [1, 2, 4, 8]

    # ---- OAT-style ----
    print("\n[1/4] Training OAT-style tokenizer...")
    from dhb_xr.tokenization.oat_tokenizer import OATTokenizer

    oat = OATTokenizer(
        input_dim=8, input_horizon=50, emb_dim=64, latent_dim=4,
        num_registers=8, fsq_levels=[8, 5, 5, 5],
        encoder_depth=3, decoder_depth=3, num_heads=4,
        nested_dropout_mode="uniform",
    )

    # Train
    optimizer = oat.get_optimizer(learning_rate=1e-3)
    # Batch up training data
    train_data = data.repeat(20, 1, 1)  # Repeat for more training signal
    oat.train()
    for epoch in range(100):
        loss = oat(train_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"  Final training loss: {loss.item():.6f}")

    # Evaluate at different prefix lengths
    oat.eval()
    oat_mses = {}
    oat_recons = {}
    with torch.no_grad():
        for k in prefix_lengths:
            recon = oat.autoencode(data, eval_keep_k=[k])
            mse = float(torch.nn.functional.mse_loss(recon, data).item())
            oat_mses[k] = mse
            oat_recons[k] = recon[0].numpy()
            print(f"  k={k}: MSE = {mse:.6f}")
    results["OAT-style"] = oat_mses

    # ---- RVQ partial ----
    print("\n[2/4] Training RVQ tokenizer...")
    from dhb_xr.tokenization.rvq import ResidualVQTokenizer

    rvq = ResidualVQTokenizer(
        invariant_dim=8, latent_dim=16, codebook_size=256, num_codebooks=8
    )
    opt_rvq = torch.optim.Adam(rvq.parameters(), lr=1e-3)
    rvq.train()
    for _ in range(100):
        all_idx, recon, z, z_sum = rvq(train_data)
        loss = torch.nn.functional.mse_loss(recon, train_data)
        opt_rvq.zero_grad()
        loss.backward()
        opt_rvq.step()
    print(f"  Final training loss: {loss.item():.6f}")

    rvq.eval()
    rvq_mses = {}
    rvq_recons = {}
    with torch.no_grad():
        for k in prefix_lengths:
            num_cb = min(k, 8)
            _, z_sum, recon = rvq.encode_partial(data, num_codebooks=num_cb)
            mse = float(torch.nn.functional.mse_loss(recon, data).item())
            rvq_mses[k] = mse
            rvq_recons[k] = recon[0].numpy()
            print(f"  k={k} codebooks: MSE = {mse:.6f}")
    results["RVQ (partial)"] = rvq_mses

    # ---- Hierarchical ----
    print("\n[3/4] Training Hierarchical tokenizer...")
    from dhb_xr.tokenization.hierarchical import HierarchicalTokenizer

    hier = HierarchicalTokenizer(
        invariant_dim=8, latent_dim=16, codebook_size=256,
        num_levels=4, temporal_downsample=2
    )
    opt_hier = torch.optim.Adam(hier.parameters(), lr=1e-3)
    hier.train()
    for _ in range(100):
        all_tok, recon = hier(train_data)
        loss = torch.nn.functional.mse_loss(recon, train_data)
        opt_hier.zero_grad()
        loss.backward()
        opt_hier.step()
    print(f"  Final training loss: {loss.item():.6f}")

    hier.eval()
    hier_mses = {}
    hier_recons = {}
    with torch.no_grad():
        for k in prefix_lengths:
            max_level = min(k, 4)
            all_tok, recon = hier(data, max_level=max_level)
            mse = float(torch.nn.functional.mse_loss(recon, data).item())
            hier_mses[k] = mse
            hier_recons[k] = recon[0].numpy()
            print(f"  max_level={max_level}: MSE = {mse:.6f}")
    results["Hierarchical"] = hier_mses

    # ---- Progressive ----
    print("\n[4/4] Training Progressive tokenizer...")
    from dhb_xr.tokenization.hierarchical import ProgressiveTokenizer

    prog = ProgressiveTokenizer(
        invariant_dim=8, latent_dim=16, codebook_size=256, num_refinements=7
    )
    opt_prog = torch.optim.Adam(prog.parameters(), lr=1e-3)
    prog.train()
    for _ in range(100):
        all_tok, recon = prog(train_data)
        loss = torch.nn.functional.mse_loss(recon, train_data)
        opt_prog.zero_grad()
        loss.backward()
        opt_prog.step()
    print(f"  Final training loss: {loss.item():.6f}")

    prog.eval()
    prog_mses = {}
    prog_recons = {}
    with torch.no_grad():
        for k in prefix_lengths:
            num_refine = min(k - 1, 7)  # k=1 -> base only
            all_tok, recon = prog(data, num_refine=num_refine)
            mse = float(torch.nn.functional.mse_loss(recon, data).item())
            prog_mses[k] = mse
            prog_recons[k] = recon[0].numpy()
            print(f"  num_refine={num_refine}: MSE = {mse:.6f}")
    results["Progressive"] = prog_mses

    # ---- Summary table ----
    print(f"\n{'Method':<20}", end="")
    for k in prefix_lengths:
        print(f"  k={k:>2}", end="")
    print()
    print("-" * 60)
    for method, mses in results.items():
        print(f"{method:<20}", end="")
        for k in prefix_lengths:
            print(f"  {mses[k]:.4f}", end="")
        print()

    # ---- Visualization ----
    if not HAS_MPL:
        return results

    gt = inv  # Ground truth

    # Figure 1: MSE vs prefix length
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"OAT-style": "#e74c3c", "RVQ (partial)": "#3498db",
              "Hierarchical": "#2ecc71", "Progressive": "#f39c12"}
    for method, mses in results.items():
        ks = sorted(mses.keys())
        vals = [mses[k] for k in ks]
        ax.plot(ks, vals, "o-", label=method, color=colors.get(method, "gray"),
                linewidth=2, markersize=8)
    ax.set_xlabel("Prefix length (k tokens)", fontsize=12)
    ax.set_ylabel("Reconstruction MSE", fontsize=12)
    ax.set_title("Prefix Decoding: MSE vs Number of Tokens", fontsize=13)
    ax.set_xticks(prefix_lengths)
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    try:
        plt.savefig("figures/prefix_decoding_mse.png", dpi=150, bbox_inches="tight")
        print("\nFigure saved to figures/prefix_decoding_mse.png")
    except Exception:
        pass
    plt.close()

    # Figure 2: OAT prefix reconstruction visualization
    fig, axes = plt.subplots(1, len(prefix_lengths) + 1, figsize=(4 * (len(prefix_lengths) + 1), 4))

    # Plot ground truth
    ax = axes[0]
    for d in range(min(4, gt.shape[1])):
        ax.plot(gt[:, d], alpha=0.7, label=f"dim {d}")
    ax.set_title("Ground Truth", fontsize=11)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Value")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Plot OAT reconstructions
    for i, k in enumerate(prefix_lengths):
        ax = axes[i + 1]
        recon = oat_recons[k]
        for d in range(min(4, recon.shape[1])):
            ax.plot(recon[:, d], alpha=0.7)
        mse = oat_mses[k]
        ax.set_title(f"OAT k={k}\nMSE={mse:.4f}", fontsize=11)
        ax.set_xlabel("Timestep")
        ax.grid(True, alpha=0.3)

    plt.suptitle("OAT-Style Prefix Decoding: Coarse-to-Fine Reconstruction", fontsize=13, y=1.02)
    plt.tight_layout()
    try:
        plt.savefig("figures/prefix_decoding_oat_detail.png", dpi=150, bbox_inches="tight")
        print("Figure saved to figures/prefix_decoding_oat_detail.png")
    except Exception:
        pass
    plt.close()

    return results


def main():
    run_prefix_benchmark()


if __name__ == "__main__":
    main()
