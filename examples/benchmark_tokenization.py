"""
Comprehensive Action Tokenization Benchmark

Compares 7 tokenization methods on DHB invariant sequences:
  1. Naive per-dimension binning (baseline)
  2. VQ-VAE (DHBTokenizer) -- current DHB-XR
  3. RVQ (ResidualVQTokenizer) -- current DHB-XR
  4. HierarchicalTokenizer -- current DHB-XR
  5. ProgressiveTokenizer -- current DHB-XR
  6. FASTTokenizer (DCT + BPE) -- new, non-learned
  7. OATTokenizer (Register + FSQ + NestedDropout) -- new, learned

Metrics measured:
  - Rate: tokens per action chunk (compression ratio)
  - Distortion: MSE reconstruction error
  - Encode/decode latency (ms)
  - Codebook utilization (VQ/RVQ only)

Output: console table + matplotlib figures saved to figures/

References:
  - FAST (Physical Intelligence, 2025): arXiv:2501.09747
  - OAT (Liu et al., 2026): arXiv:2602.04215

Requires: PyTorch (`pip install dhb_xr[tokenization]`)
"""

import time
import numpy as np

# Attempt imports
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ============================================================================
# Data generation
# ============================================================================

def generate_test_invariants(
    num_sequences: int = 50,
    seq_len: int = 50,
    dim: int = 8,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic DHB invariant sequences for benchmarking.

    Creates a mix of motion patterns: random walk, sinusoidal, and constant.

    Returns:
        (num_sequences, seq_len, dim) array of invariant values.
    """
    rng = np.random.RandomState(seed)
    data = []
    for i in range(num_sequences):
        pattern = i % 3
        if pattern == 0:
            # Random walk (typical real motion)
            seq = np.cumsum(rng.randn(seq_len, dim) * 0.02, axis=0)
        elif pattern == 1:
            # Sinusoidal (smooth periodic)
            t = np.linspace(0, 4 * np.pi, seq_len)
            seq = np.column_stack([
                np.sin(t * (1 + 0.1 * d)) * (0.1 + 0.05 * d)
                for d in range(dim)
            ])
        else:
            # Constant + noise (near-static segments)
            seq = rng.randn(1, dim) * 0.1 + rng.randn(seq_len, dim) * 0.005
        data.append(seq.astype(np.float32))
    return np.array(data)


# ============================================================================
# Naive binning baseline
# ============================================================================

def naive_binning(chunk: np.ndarray, num_bins: int = 256) -> dict:
    """Naive per-dimension, per-timestep binning.

    Each continuous value is independently discretized into num_bins bins.
    """
    T, D = chunk.shape
    v_min, v_max = chunk.min(), chunk.max()
    eps = 1e-8
    normalized = (chunk - v_min) / (v_max - v_min + eps)
    indices = np.clip((normalized * num_bins).astype(int), 0, num_bins - 1)
    # Reconstruct
    recon = indices.astype(float) / num_bins * (v_max - v_min + eps) + v_min
    tokens_flat = indices.flatten()
    mse = float(np.mean((chunk - recon) ** 2))
    return {
        "num_tokens": len(tokens_flat),
        "mse": mse,
        "compression_ratio": chunk.size / len(tokens_flat),
    }


# ============================================================================
# Benchmark runner
# ============================================================================

def train_tokenizer(model, data_tensor, epochs=30, lr=1e-3):
    """Quick training loop for learned tokenizers."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        if hasattr(model, 'forward') and callable(model.forward):
            result = model(data_tensor)
            if isinstance(result, torch.Tensor) and result.ndim == 0:
                loss = result  # OATTokenizer returns loss directly
            elif isinstance(result, tuple):
                # VQ-VAE style: (indices, recon, z, z_q)
                if len(result) == 4:
                    indices, recon, z, z_q = result
                    loss = model.loss(data_tensor, recon, z, z_q)
                elif len(result) == 2:
                    tokens, recon = result
                    loss = torch.nn.functional.mse_loss(recon, data_tensor)
                else:
                    break
            else:
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def benchmark_vqvae(data_np, data_tensor):
    """Benchmark VQ-VAE tokenizer."""
    from dhb_xr.tokenization.vqvae import DHBTokenizer

    model = DHBTokenizer(invariant_dim=8, latent_dim=16, codebook_size=256)
    train_tokenizer(model, data_tensor, epochs=50)
    model.eval()

    with torch.no_grad():
        t0 = time.perf_counter()
        indices, recon, z, z_q = model(data_tensor)
        latency = (time.perf_counter() - t0) * 1000 / data_tensor.shape[0]

    mse = float(torch.nn.functional.mse_loss(recon, data_tensor).item())
    B, T = indices.shape
    unique_codes = len(torch.unique(indices))

    return {
        "method": "VQ-VAE",
        "tokens_per_chunk": T,
        "compression_ratio": (T * 8) / T,  # 8 floats per step -> 1 token
        "mse": mse,
        "latency_ms": latency,
        "codebook_util": unique_codes / 256,
    }


def benchmark_rvq(data_np, data_tensor):
    """Benchmark RVQ tokenizer."""
    from dhb_xr.tokenization.rvq import ResidualVQTokenizer

    model = ResidualVQTokenizer(
        invariant_dim=8, latent_dim=16, codebook_size=256, num_codebooks=3
    )
    # RVQ returns 4-tuple but has no .loss() -- train with MSE directly
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(50):
        all_idx, recon, z, z_sum = model(data_tensor)
        loss = torch.nn.functional.mse_loss(recon, data_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()

    with torch.no_grad():
        t0 = time.perf_counter()
        all_indices, recon, z, z_sum = model(data_tensor)
        latency = (time.perf_counter() - t0) * 1000 / data_tensor.shape[0]

    mse = float(torch.nn.functional.mse_loss(recon, data_tensor).item())
    T = all_indices[0].shape[1]
    total_tokens = T * 3  # 3 codebooks

    return {
        "method": "RVQ (3 codebooks)",
        "tokens_per_chunk": total_tokens,
        "compression_ratio": (T * 8) / total_tokens,
        "mse": mse,
        "latency_ms": latency,
        "codebook_util": None,
    }


def benchmark_hierarchical(data_np, data_tensor):
    """Benchmark Hierarchical tokenizer."""
    from dhb_xr.tokenization.hierarchical import HierarchicalTokenizer

    model = HierarchicalTokenizer(
        invariant_dim=8, latent_dim=16, codebook_size=256,
        num_levels=3, temporal_downsample=2
    )
    # Simple training (limited since loss API differs)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(30):
        all_tokens, recon = model(data_tensor)
        loss = torch.nn.functional.mse_loss(recon, data_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        all_tokens, recon = model(data_tensor)
        latency = (time.perf_counter() - t0) * 1000 / data_tensor.shape[0]

    mse = float(torch.nn.functional.mse_loss(recon, data_tensor).item())
    total_tokens = sum(t.shape[-1] for t in all_tokens)

    return {
        "method": "Hierarchical (3 levels)",
        "tokens_per_chunk": total_tokens,
        "compression_ratio": (data_tensor.shape[1] * 8) / total_tokens,
        "mse": mse,
        "latency_ms": latency,
        "codebook_util": None,
    }


def benchmark_progressive(data_np, data_tensor):
    """Benchmark Progressive tokenizer."""
    from dhb_xr.tokenization.hierarchical import ProgressiveTokenizer

    model = ProgressiveTokenizer(
        invariant_dim=8, latent_dim=16, codebook_size=256, num_refinements=3
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(30):
        all_tokens, recon = model(data_tensor)
        loss = torch.nn.functional.mse_loss(recon, data_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        all_tokens, recon = model(data_tensor)
        latency = (time.perf_counter() - t0) * 1000 / data_tensor.shape[0]

    mse = float(torch.nn.functional.mse_loss(recon, data_tensor).item())
    total_tokens = sum(t.shape[-1] for t in all_tokens)

    return {
        "method": "Progressive (3 refine)",
        "tokens_per_chunk": total_tokens,
        "compression_ratio": (data_tensor.shape[1] * 8) / total_tokens,
        "mse": mse,
        "latency_ms": latency,
        "codebook_util": None,
    }


def benchmark_fast(data_np):
    """Benchmark FAST tokenizer (no PyTorch needed)."""
    from dhb_xr.tokenization.fast_tokenizer import FASTTokenizer

    tokenizer = FASTTokenizer(scale=10.0, vocab_size=1024, num_merges=200)

    # Fit on corpus
    corpus = [data_np[i] for i in range(data_np.shape[0])]
    tokenizer.fit(corpus)

    # Benchmark
    total_tokens = 0
    total_mse = 0.0
    t0 = time.perf_counter()
    for chunk in corpus:
        tokens = tokenizer.encode(chunk)
        recon = tokenizer.decode(tokens, chunk.shape[0], chunk.shape[1])
        total_tokens += len(tokens)
        total_mse += float(np.mean((chunk - recon) ** 2))
    latency = (time.perf_counter() - t0) * 1000 / len(corpus)

    avg_tokens = total_tokens / len(corpus)
    avg_mse = total_mse / len(corpus)
    T, D = corpus[0].shape

    return {
        "method": "FAST (DCT+BPE)",
        "tokens_per_chunk": avg_tokens,
        "compression_ratio": (T * D) / avg_tokens if avg_tokens > 0 else 1.0,
        "mse": avg_mse,
        "latency_ms": latency,
        "codebook_util": None,
    }


def benchmark_oat(data_np, data_tensor):
    """Benchmark OAT-style tokenizer."""
    from dhb_xr.tokenization.oat_tokenizer import OATTokenizer

    T = data_tensor.shape[1]
    model = OATTokenizer(
        input_dim=8, input_horizon=T, emb_dim=64, latent_dim=4,
        num_registers=8, fsq_levels=[8, 5, 5, 5],
        encoder_depth=2, decoder_depth=2, num_heads=4,
        nested_dropout_mode="uniform",
    )

    # Train
    optimizer = model.get_optimizer(learning_rate=1e-3)
    model.train()
    for _ in range(50):
        loss = model(data_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        tokens = model.tokenize(data_tensor)
        recon = model.detokenize(tokens)
        latency = (time.perf_counter() - t0) * 1000 / data_tensor.shape[0]

    mse = float(torch.nn.functional.mse_loss(recon, data_tensor).item())

    return {
        "method": "OAT-style (Reg+FSQ+ND)",
        "tokens_per_chunk": 8,  # num_registers
        "compression_ratio": T / 8,
        "mse": mse,
        "latency_ms": latency,
        "codebook_util": None,
    }


# ============================================================================
# Main benchmark
# ============================================================================

def run_benchmark():
    """Run the full tokenization benchmark."""
    print("=" * 70)
    print("Action Tokenization Benchmark: DHB-XR vs FAST vs OAT")
    print("=" * 70)

    # Generate test data
    data_np = generate_test_invariants(num_sequences=50, seq_len=50, dim=8)
    print(f"\nTest data: {data_np.shape} (sequences x timesteps x dim)")

    results = []

    # 1. Naive binning (always available)
    print("\n[1/7] Naive per-dimension binning...")
    naive_results = []
    t0 = time.perf_counter()
    for i in range(data_np.shape[0]):
        naive_results.append(naive_binning(data_np[i]))
    latency = (time.perf_counter() - t0) * 1000 / data_np.shape[0]
    results.append({
        "method": "Naive Binning",
        "tokens_per_chunk": naive_results[0]["num_tokens"],
        "compression_ratio": np.mean([r["compression_ratio"] for r in naive_results]),
        "mse": np.mean([r["mse"] for r in naive_results]),
        "latency_ms": latency,
        "codebook_util": None,
    })

    if HAS_TORCH:
        data_tensor = torch.from_numpy(data_np)

        # 2. VQ-VAE
        print("[2/7] VQ-VAE (DHBTokenizer)...")
        try:
            results.append(benchmark_vqvae(data_np, data_tensor))
        except Exception as e:
            print(f"  Skipped: {e}")

        # 3. RVQ
        print("[3/7] RVQ (ResidualVQTokenizer)...")
        try:
            results.append(benchmark_rvq(data_np, data_tensor))
        except Exception as e:
            print(f"  Skipped: {e}")

        # 4. Hierarchical
        print("[4/7] Hierarchical tokenizer...")
        try:
            results.append(benchmark_hierarchical(data_np, data_tensor))
        except Exception as e:
            print(f"  Skipped: {e}")

        # 5. Progressive
        print("[5/7] Progressive tokenizer...")
        try:
            results.append(benchmark_progressive(data_np, data_tensor))
        except Exception as e:
            print(f"  Skipped: {e}")

    # 6. FAST (no torch needed)
    print("[6/7] FAST (DCT + BPE)...")
    try:
        results.append(benchmark_fast(data_np))
    except Exception as e:
        print(f"  Skipped: {e}")

    if HAS_TORCH:
        # 7. OAT-style
        print("[7/7] OAT-style (Register + FSQ + NestedDropout)...")
        try:
            results.append(benchmark_oat(data_np, data_tensor))
        except Exception as e:
            print(f"  Skipped: {e}")

    # Print results table
    print("\n" + "=" * 70)
    print(f"{'Method':<28} {'Tokens':>7} {'Ratio':>7} {'MSE':>10} {'ms':>7} {'CB Util':>8}")
    print("-" * 70)
    for r in results:
        cb = f"{r['codebook_util']:.2f}" if r["codebook_util"] is not None else "N/A"
        print(
            f"{r['method']:<28} {r['tokens_per_chunk']:>7.0f} "
            f"{r['compression_ratio']:>7.1f}x "
            f"{r['mse']:>10.6f} "
            f"{r['latency_ms']:>6.2f} "
            f"{cb:>8}"
        )
    print("=" * 70)

    # Plot results
    if HAS_MPL and len(results) >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        methods = [r["method"] for r in results]
        colors = plt.cm.Set2(np.linspace(0, 1, len(results)))

        # Rate-distortion scatter
        ax = axes[0]
        for i, r in enumerate(results):
            ax.scatter(
                r["tokens_per_chunk"], r["mse"],
                color=colors[i], s=100, zorder=5, label=r["method"]
            )
        ax.set_xlabel("Tokens per chunk (rate)")
        ax.set_ylabel("MSE (distortion)")
        ax.set_title("Rate-Distortion")
        ax.set_yscale("log")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

        # Compression ratio bar chart
        ax = axes[1]
        x = range(len(results))
        ax.bar(x, [r["compression_ratio"] for r in results], color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels([r["method"] for r in results], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Compression ratio")
        ax.set_title("Compression Ratio")
        ax.grid(True, alpha=0.3, axis="y")

        # Latency bar chart
        ax = axes[2]
        ax.bar(x, [r["latency_ms"] for r in results], color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels([r["method"] for r in results], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Latency (ms/chunk)")
        ax.set_title("Encode + Decode Latency")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        try:
            plt.savefig("figures/tokenization_benchmark.png", dpi=150, bbox_inches="tight")
            print("\nFigure saved to figures/tokenization_benchmark.png")
        except Exception:
            pass
        plt.close()

    return results


def main():
    """Main entry point."""
    run_benchmark()


if __name__ == "__main__":
    main()
