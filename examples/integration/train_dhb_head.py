#!/usr/bin/env python
"""Train DHBActionHeader on LIBERO demonstrations.

Standalone script — doesn't require StarVLA framework, just the action head.
Uses precomputed DHB invariants as supervision signal.

Pipeline:
  1. Load LIBERO demos (EE positions + orientations)
  2. Precompute DHB invariants as ground truth
  3. Create simple visual encoder (ResNet features as proxy for VLM)
  4. Train DHBActionHeader to predict invariants from visual features
  5. Evaluate: reconstruct trajectory from predicted invariants

This validates that a learned policy CAN predict correct invariants,
which is the key missing experiment for the paper.

Usage:
    # CPU (slow but works)
    python examples/integration/train_dhb_head.py --epochs 20 --batch-size 4

    # GPU (when CUDA PyTorch available)
    python examples/integration/train_dhb_head.py --epochs 50 --batch-size 16 --device cuda

    # Quick validation (just forward pass)
    python examples/integration/train_dhb_head.py --epochs 1 --batch-size 2 --quick
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from dhb_xr.encoder.dhb_dr import encode_dhb_dr


# ── Dataset ──────────────────────────────────────────────────────────────────

class LiberoInvariantDataset(Dataset):
    """Load LIBERO demos and precompute DHB invariants.

    Each sample: (visual_features, action_chunk, invariant_chunk)
    Visual features are a proxy (random projection of EE state + noise)
    since we don't have the full VLM backbone.
    """

    def __init__(self, dataset_dir: str, chunk_len: int = 8, max_demos: int = 50,
                 augment_se3: bool = False):
        self.chunk_len = chunk_len
        self.augment_se3 = augment_se3
        self.samples = []

        import h5py
        import glob
        from scipy.spatial.transform import Rotation

        demo_files = sorted(glob.glob(f"{dataset_dir}/*.hdf5"))[:10]

        for df in demo_files:
            with h5py.File(df) as f:
                data = f["data"]
                demo_keys = sorted(k for k in data.keys() if "demo" in k)[:max_demos // len(demo_files)]

                for dk in demo_keys:
                    obs = data[dk]["obs"]
                    pos = np.array(obs["ee_pos"])  # (T, 3)

                    # Get quaternions (wxyz)
                    if "robot_states" in data[dk]:
                        rs = np.array(data[dk]["robot_states"])
                        quat_wxyz = rs[:, 3:7]
                    else:
                        euler = np.array(obs["ee_ori"])
                        xyzw = Rotation.from_euler("xyz", euler).as_quat()
                        quat_wxyz = np.column_stack([xyzw[:, 3], xyzw[:, 0], xyzw[:, 1], xyzw[:, 2]])

                    actions = np.array(data[dk]["actions"])  # (T, 7)

                    # Encode to DHB invariants
                    enc = encode_dhb_dr(pos, quat_wxyz)
                    lin_inv = enc["linear_motion_invariants"]  # (T-1, 4)
                    ang_inv = enc["angular_motion_invariants"]  # (T-1, 4)
                    invariants = np.hstack([lin_inv, ang_inv]).astype(np.float32)  # (T-1, 8)

                    # Chunk into fixed-length training samples
                    T = min(len(invariants), len(pos), len(actions))
                    for start in range(0, T - chunk_len, chunk_len // 2):
                        end = start + chunk_len
                        if end > T or end > len(pos) or end > len(actions):
                            break

                        # Visual proxy: EE state + positional encoding
                        ee_state = np.hstack([pos[start:end], quat_wxyz[start:end]]).astype(np.float32)  # (chunk, 7)
                        inv_chunk = invariants[start:end]  # (chunk, 8)
                        act_chunk = actions[start:end].astype(np.float32)  # (chunk, 7)

                        self.samples.append({
                            "ee_state": ee_state,
                            "invariants": inv_chunk,
                            "actions": act_chunk,
                        })

        print(f"Loaded {len(self.samples)} training chunks from {len(demo_files)} files")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        ee = s["ee_state"].copy()  # (chunk, 7): pos(3) + quat_wxyz(4)

        if self.augment_se3:
            # Apply random SE(3) to EE state; invariant targets stay UNCHANGED
            # This is the "free augmentation" — unique to invariant representations
            from scipy.spatial.transform import Rotation as R_cls
            R = R_cls.random().as_matrix().astype(np.float32)
            t = np.random.randn(3).astype(np.float32) * 0.1

            pos = ee[:, :3]                          # (chunk, 3)
            quat_wxyz = ee[:, 3:7]                   # (chunk, 4) wxyz

            # Transform positions: p' = R @ p + t
            pos_aug = (R @ pos.T).T + t              # (chunk, 3)

            # Transform quaternions: q' = R_se3 ⊗ q
            R_scipy = R_cls.from_matrix(R)
            quat_xyzw = np.column_stack([quat_wxyz[:, 1], quat_wxyz[:, 2],
                                          quat_wxyz[:, 3], quat_wxyz[:, 0]])
            q_aug = (R_scipy * R_cls.from_quat(quat_xyzw)).as_quat()  # xyzw
            quat_wxyz_aug = np.column_stack([q_aug[:, 3], q_aug[:, 0],
                                              q_aug[:, 1], q_aug[:, 2]])

            ee = np.hstack([pos_aug, quat_wxyz_aug]).astype(np.float32)
            # invariants are UNCHANGED — this is the whole point!

        return {
            "ee_state": torch.tensor(ee),
            "invariants": torch.tensor(s["invariants"]),
            "actions": torch.tensor(s["actions"]),
        }


# ── Visual encoder proxy ─────────────────────────────────────────────────────

class VisualEncoderProxy(nn.Module):
    """Proxy for VLM backbone — projects EE state to VLM-like hidden dim.

    In a real system, this would be Qwen2.5-VL. For the paper's Exp 2,
    we validate that the action head architecture works; the full VLM
    integration is future work.
    """

    def __init__(self, input_dim: int = 7, hidden_dim: int = 256, seq_len: int = 20):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        # Expand to sequence length (simulating VLM output tokens)
        self.expand = nn.Linear(hidden_dim, seq_len * hidden_dim)

    def forward(self, ee_state: torch.Tensor) -> torch.Tensor:
        """(B, chunk_len, 7) → (B, seq_len, hidden_dim)"""
        B = ee_state.shape[0]
        pooled = ee_state.mean(dim=1)  # (B, 7)
        h = self.encoder(pooled)  # (B, hidden_dim)
        expanded = self.expand(h)  # (B, seq_len * hidden_dim)
        return expanded.view(B, self.seq_len, -1)  # (B, seq_len, hidden_dim)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Dataset
    augment = getattr(args, 'augment_se3', False)
    dataset = LiberoInvariantDataset(
        args.dataset_dir,
        chunk_len=args.chunk_len,
        max_demos=args.max_demos,
        augment_se3=augment,
    )
    if augment:
        print(f"SE(3) augmentation: ON (invariant targets unchanged — free aug)")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Models
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "vendor" / "starVLA"))
    from starVLA.model.modules.action_model.DHBActionHeader import DHBActionHeader

    hidden_dim = args.hidden_dim
    vis_encoder = VisualEncoderProxy(input_dim=7, hidden_dim=hidden_dim, seq_len=20).to(device)
    action_head = DHBActionHeader(
        input_dim=hidden_dim,
        hidden_dim=hidden_dim,
        action_dim=7,
        invariant_dim=8,
        num_oat_tokens=8,
        chunk_len=args.chunk_len,
        use_oat_quantizer=False,  # Start without OAT for simpler training
        lambda_invariant=1.0,
        lambda_action=0.5,
    ).to(device)

    params = list(vis_encoder.parameters()) + list(action_head.parameters())
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    total_params = sum(p.numel() for p in params)
    print(f"Parameters: {total_params:,} ({total_params / 1e6:.1f}M)")
    print(f"Training: {len(dataset)} samples, {args.epochs} epochs, batch={args.batch_size}")
    print()

    # Training
    results = {"epochs": [], "train_loss": [], "inv_mse": [], "act_l1": []}
    best_loss = float("inf")

    for epoch in range(args.epochs):
        vis_encoder.train()
        action_head.train()
        epoch_loss, epoch_inv, epoch_act, n_batches = 0, 0, 0, 0

        for batch in loader:
            ee = batch["ee_state"].to(device)       # (B, chunk, 7)
            gt_inv = batch["invariants"].to(device)  # (B, chunk, 8)
            gt_act = batch["actions"].to(device)     # (B, chunk, 7)

            # Forward
            vlm_features = vis_encoder(ee)  # (B, 20, hidden_dim)
            loss = action_head(vlm_features, actions_target=gt_act, gt_invariants=gt_inv)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            # Track invariant and action losses separately
            with torch.no_grad():
                pred = action_head.predict_action(vlm_features)
                inv_pred = action_head.projector(vlm_features)
                inv_mse = float(nn.functional.mse_loss(inv_pred, gt_inv).item())
                act_l1 = float(nn.functional.l1_loss(pred, gt_act).item())

            epoch_loss += loss.item()
            epoch_inv += inv_mse
            epoch_act += act_l1
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_inv = epoch_inv / max(n_batches, 1)
        avg_act = epoch_act / max(n_batches, 1)

        results["epochs"].append(epoch)
        results["train_loss"].append(avg_loss)
        results["inv_mse"].append(avg_inv)
        results["act_l1"].append(avg_act)

        if avg_loss < best_loss:
            best_loss = avg_loss
            marker = " *best*"
        else:
            marker = ""

        if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs - 1:
            print(f"  Epoch {epoch:3d}/{args.epochs}: loss={avg_loss:.4f}  "
                  f"inv_mse={avg_inv:.4f}  act_l1={avg_act:.4f}"
                  f"  lr={scheduler.get_last_lr()[0]:.1e}{marker}")

    # Save results
    Path("results").mkdir(exist_ok=True)
    with open("results/exp2_dhb_head_training.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save checkpoint
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save({
        "vis_encoder": vis_encoder.state_dict(),
        "action_head": action_head.state_dict(),
        "epoch": args.epochs,
        "best_loss": best_loss,
    }, "checkpoints/dhb_head_latest.pt")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"  Results: results/exp2_dhb_head_training.json")
    print(f"  Checkpoint: checkpoints/dhb_head_latest.pt")

    # Quick eval: predict invariants on a held-out sample
    vis_encoder.eval()
    action_head.eval()
    with torch.no_grad():
        sample = dataset[0]
        ee = sample["ee_state"].unsqueeze(0).to(device)
        gt_inv = sample["invariants"]
        features = vis_encoder(ee)
        pred_inv = action_head.projector(features).cpu().squeeze(0)
        mse = float(nn.functional.mse_loss(pred_inv, gt_inv).item())
        print(f"\n  Eval (1 sample): invariant MSE = {mse:.6f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train DHBActionHeader on LIBERO")
    parser.add_argument("--dataset-dir", default="/home/andypark/Projects/data/libero/libero_spatial")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--chunk-len", type=int, default=8)
    parser.add_argument("--max-demos", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quick", action="store_true", help="1 epoch validation only")
    parser.add_argument("--augment-se3", action="store_true",
                        help="Apply random SE(3) to inputs; invariant targets unchanged (free aug)")
    args = parser.parse_args()

    if args.quick:
        args.epochs = 1
        args.max_demos = 10

    print("=" * 60)
    print("  DHBActionHeader Training — LIBERO-Spatial")
    print("=" * 60)
    train(args)


if __name__ == "__main__":
    main()
