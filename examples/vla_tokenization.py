"""
VLA Tokenization Example (DHB action token streaming)

This example demonstrates the validated DHB token streaming path for VLA:
1. Encode a trajectory to DHB invariants
2. Tokenize invariants with OAT
3. Decode streamed token prefixes back to full-horizon invariant chunks

This matches the intended chunked VLA execution contract more closely than the
older VQ-only roundtrip because OAT supports anytime prefix decoding.
"""

import numpy as np


def run_example():
    """Run the VLA tokenization example."""
    print("DHB-XR VLA Token Streaming Example")
    print("=" * 38)

    try:
        from dhb_xr.core.types import DHBMethod, EncodingMethod
        from dhb_xr.integration.vla.pipeline import DHBVLAPipeline, DHBVLAPipelineConfig
    except ImportError as e:
        print("❌ PyTorch required for tokenization")
        print(f"Install with: pip install dhb_xr[gpu]")
        print(f"Error: {e}")
        return None

    # Create synthetic trajectory
    np.random.seed(2)
    n = 40
    positions = np.cumsum(np.random.randn(n, 3) * 0.01, axis=0)
    quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))

    print(f"Trajectory: {n} poses")
    print(f"Positions shape: {positions.shape}")

    pipeline = DHBVLAPipeline(
        DHBVLAPipelineConfig(
            dhb_method=DHBMethod.DOUBLE_REFLECTION,
            method=EncodingMethod.POSITION,
            tokenizer_kind="oat",
            oat_num_registers=8,
            oat_emb_dim=64,
            oat_num_heads=4,
            oat_encoder_depth=2,
            oat_decoder_depth=2,
        )
    )

    print("\nEncoding to DHB-DR invariants...")
    out = pipeline.encode_trajectory(positions, quaternions)
    invariants = np.concatenate([
        out["linear_motion_invariants"],
        out["angular_motion_invariants"],
    ], axis=1).astype(np.float32)

    print(f"Combined invariants shape: {invariants.shape}")
    print(f"Values per timestep: {invariants.shape[1]} (4 linear + 4 angular)")

    print("\nTokenizing with OAT...")
    tokens = pipeline.tokenize_invariants(invariants)
    full_recon = pipeline.detokenize_tokens(
        tokens,
        invariant_dim=invariants.shape[1],
        horizon=invariants.shape[0],
    )[0]
    stream = pipeline.decode_token_stream(
        tokens,
        invariant_dim=invariants.shape[1],
        horizon=invariants.shape[0],
        prefix_lengths=[1, 2, len(tokens)],
    )

    full_mse = float(np.mean((full_recon - invariants) ** 2))
    compression_ratio = float(invariants.shape[0] / len(tokens))

    print("✅ Tokenization complete")
    print(f"Token shape: {tokens.shape}")
    print(f"Full reconstruction shape: {full_recon.shape}")
    print("\nResults:")
    print(f"  Full reconstruction MSE: {full_mse:.6f}")
    print(f"  Temporal compression ratio: {compression_ratio:.2f}x")
    print(f"  Sample tokens: {tokens.tolist()}")
    print("  Streamed prefix reconstructions:")
    for item in stream:
        keep_k = item["keep_k"]
        recon = item["reconstruction"][0]
        mse = float(np.mean((recon - invariants) ** 2))
        print(f"    keep_k={keep_k}: mse={mse:.6f}, shape={recon.shape}")

    return {
        "invariants": invariants,
        "tokens": tokens,
        "reconstructed": full_recon,
        "stream": stream,
        "loss": full_mse,
    }


def main():
    """Main function for command-line execution."""
    run_example()


if __name__ == "__main__":
    main()
