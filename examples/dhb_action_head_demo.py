#!/usr/bin/env python3
"""
DHB Action Head Demo: Train and evaluate on synthetic data.

Demonstrates the DHBActionHead as a drop-in VLA action head that:
1. Takes VLM hidden states as input
2. Predicts DHB invariants via cross-attention
3. Decodes to SE(3) trajectories from any start pose

Usage:
    pixi run python examples/dhb_action_head_demo.py
"""
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dhb_xr.integration.vla.action_head import DHBActionHead
from dhb_xr.encoder.dhb_dr import encode_dhb_dr


def generate_synthetic_data(n_demos=100, T=30, vlm_seq_len=10, vlm_dim=128):
    """Generate synthetic training data: VLM embeddings + DHB invariants."""
    data = []
    for _ in range(n_demos):
        # Random trajectory (smooth random walk)
        t = np.linspace(0, 2 * np.pi, T + 2)
        positions = np.column_stack([
            0.5 + 0.1 * np.sin(t) + 0.01 * np.random.randn(T + 2),
            0.0 + 0.1 * np.cos(t) + 0.01 * np.random.randn(T + 2),
            0.3 + 0.05 * np.sin(2 * t) + 0.01 * np.random.randn(T + 2),
        ])
        quaternions = np.tile([1, 0, 0, 0], (T + 2, 1)).astype(np.float64)

        # Encode to DHB invariants
        enc = encode_dhb_dr(positions, quaternions, method="pos",
                            use_default_initial_frames=True)
        lin_inv = enc["linear_motion_invariants"]
        ang_inv = enc["angular_motion_invariants"]
        invariants = np.concatenate([lin_inv, ang_inv], axis=1)[:T]

        # Synthetic VLM embeddings (would come from actual VLM in practice)
        vlm_hidden = torch.randn(vlm_seq_len, vlm_dim)

        data.append({
            "vlm_hidden": vlm_hidden,
            "invariants": torch.from_numpy(invariants).float(),
        })

    return data


def train(head, data, epochs=50, lr=1e-3):
    """Train the action head on synthetic data."""
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    B = 8  # batch size

    for epoch in range(epochs):
        np.random.shuffle(data)
        total_loss = 0
        n_batches = 0

        for i in range(0, len(data) - B, B):
            batch = data[i:i+B]
            vlm_hidden = torch.stack([d["vlm_hidden"] for d in batch])
            target_inv = torch.stack([d["invariants"] for d in batch])

            result = head(vlm_hidden, target_invariants=target_inv)
            loss = result["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: loss = {avg_loss:.4f}")

    return avg_loss


def evaluate(head, data):
    """Evaluate prediction quality."""
    head.eval()
    all_errors = []

    for d in data[:10]:
        vlm_hidden = d["vlm_hidden"].unsqueeze(0)
        target_inv = d["invariants"]

        result = head.predict(vlm_hidden)
        pred_inv = result["pred_invariants"][0]

        error = (pred_inv - target_inv).pow(2).mean().item()
        all_errors.append(error)

    return np.mean(all_errors)


def main():
    print("DHB Action Head Demo")
    print("=" * 50)

    # Create action head
    T = 30
    head = DHBActionHead(
        vlm_dim=128,
        action_horizon=T,
        invariant_dim=8,
        hidden_dim=128,
        num_heads=4,
        num_layers=3,
        lambda_inv=1.0,
        lambda_act=0.0,  # invariant-only loss for this demo
    )

    param_count = sum(p.numel() for p in head.parameters())
    print(f"Parameters: {param_count:,} ({param_count/1e6:.1f}M)")

    # Generate data
    print("\nGenerating synthetic training data...")
    train_data = generate_synthetic_data(n_demos=200, T=T)
    test_data = generate_synthetic_data(n_demos=20, T=T)
    print(f"  Train: {len(train_data)} demos, Test: {len(test_data)} demos")

    # Evaluate before training
    mse_before = evaluate(head, test_data)
    print(f"\nBefore training: MSE = {mse_before:.4f}")

    # Train
    print("\nTraining...")
    final_loss = train(head, train_data, epochs=50)

    # Evaluate after training
    mse_after = evaluate(head, test_data)
    print(f"\nAfter training: MSE = {mse_after:.4f}")
    print(f"Improvement: {mse_before/mse_after:.1f}x")

    # Demo: predict and decode from a new pose
    print("\n--- Inference Demo ---")
    test_vlm = test_data[0]["vlm_hidden"].unsqueeze(0)
    initial_pose = {
        "position": torch.tensor([[0.5, 0.1, 0.3]]),  # offset start
        "quaternion": torch.tensor([[1.0, 0, 0, 0]]),
    }
    result = head.predict(test_vlm, initial_pose=initial_pose)
    print(f"Predicted invariants: {result['pred_invariants'].shape}")
    print(f"Decoded positions: {result['positions'].shape}")
    print(f"  Start: {result['positions'][0, 0].numpy()}")
    print(f"  End:   {result['positions'][0, -1].numpy()}")

    print("\nDone!")


if __name__ == "__main__":
    main()
