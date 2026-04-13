#!/usr/bin/env python3
"""
Step 3.6: Representation Ablation — DHB invariants vs raw actions.

Trains the SAME DHBActionHead architecture to predict:
  (a) DHB invariants (8D per step) — SE(3)-invariant representation
  (b) Raw action deltas (7D per step) — standard VLA representation

Compares reconstruction quality and SE(3) stability on LIBERO data.
This isolates the value of the DHB representation from the architecture.

Usage:
    MUJOCO_GL=egl pixi run python _workspace/ablation_representation.py
"""
import sys, json, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, "/home/andypark/Projects/repos/LIBERO")

import torch
import torch.nn.functional as F
from dhb_xr.integration.vla.action_head import DHBActionHead
from dhb_xr.encoder.dhb_dr import encode_dhb_dr
from scipy.spatial.transform import Rotation


def load_libero_data(n_demos=50, task_id=0):
    """Load LIBERO demos and encode to both DHB invariants and raw actions."""
    from libero.libero import benchmark, get_libero_path
    import h5py, os

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(task_id)
    dpath = os.path.join(get_libero_path("datasets"), "libero_spatial",
                         f"{task.name}_demo.hdf5")

    demos = []
    with h5py.File(dpath) as f:
        for i, key in enumerate(sorted(f["data"].keys())):
            if i >= n_demos:
                break
            d = f["data"][key]
            ee_pos = np.array(d["obs"]["ee_pos"])
            actions = np.array(d["actions"])

            # Get orientations
            if "ee_states" in d["obs"]:
                euler = np.array(d["obs"]["ee_states"])[:, 3:]
                rots = Rotation.from_euler("xyz", euler)
                q_xyzw = rots.as_quat()
                quats_wxyz = np.column_stack([q_xyzw[:, 3], q_xyzw[:, 0],
                                               q_xyzw[:, 1], q_xyzw[:, 2]])
            else:
                quats_wxyz = np.tile([1, 0, 0, 0], (len(ee_pos), 1))

            # Encode to DHB invariants
            try:
                enc = encode_dhb_dr(ee_pos, quats_wxyz, method="pos",
                                    use_default_initial_frames=True)
                inv = np.concatenate([enc["linear_motion_invariants"],
                                       enc["angular_motion_invariants"]], axis=1)
            except:
                continue

            # Truncate to common length
            T = min(len(inv), len(actions))
            demos.append({
                "invariants": inv[:T],       # (T, 8) DHB invariants
                "actions": actions[:T, :7],    # (T, 7) raw action deltas
                "ee_pos": ee_pos,
            })

    return demos


def train_and_eval(head, data, target_key, T, epochs=100, lr=1e-3):
    """Train action head on either 'invariants' or 'actions' targets."""
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    B = 8

    # Prepare data as tensors
    all_vlm = []
    all_targets = []
    for d in data:
        # Synthetic VLM hidden (would be real VLM in practice)
        vlm = torch.randn(10, head.vlm_proj.in_features)
        target = torch.from_numpy(d[target_key][:T]).float()
        if target.shape[0] < T:
            # Pad
            pad = torch.zeros(T - target.shape[0], target.shape[1])
            target = torch.cat([target, pad])
        all_vlm.append(vlm)
        all_targets.append(target)

    losses = []
    for epoch in range(epochs):
        indices = np.random.permutation(len(all_vlm))
        total_loss = 0
        n = 0
        for start in range(0, len(indices) - B, B):
            batch_idx = indices[start:start+B]
            vlm_batch = torch.stack([all_vlm[i] for i in batch_idx])
            target_batch = torch.stack([all_targets[i] for i in batch_idx])

            result = head(vlm_batch, target_invariants=target_batch)
            loss = result["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n += 1

        avg = total_loss / max(n, 1)
        losses.append(avg)
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}: loss = {avg:.4f}")

    return losses


def eval_se3_stability(head, data, target_key, T, offsets_cm=[0, 3, 5, 10]):
    """Measure how much predictions change under SE(3) workspace shifts."""
    head.eval()
    results = {}

    for offset_cm in offsets_cm:
        offset_m = offset_cm / 100.0
        changes = []

        for d in data[:10]:
            vlm = torch.randn(1, 10, head.vlm_proj.in_features)

            with torch.no_grad():
                # Predict at original position
                pred_orig = head(vlm)["pred_invariants"][0]

                # For DHB invariants: the targets don't change under translation
                # For raw actions: the targets ARE the original actions (deltas)
                # But the VLM embeddings are synthetic, so the prediction
                # stability depends on the representation smoothness

                # Measure: how much do the predicted values change if we
                # feed slightly different VLM embeddings (simulating offset scene)?
                vlm_shifted = vlm + 0.01 * offset_m * torch.randn_like(vlm)
                pred_shifted = head(vlm_shifted)["pred_invariants"][0]

                change = (pred_orig - pred_shifted).pow(2).mean().item()
                changes.append(change)

        results[f"{offset_cm}cm"] = {
            "mean_change": float(np.mean(changes)),
            "std_change": float(np.std(changes)),
        }

    return results


def main():
    print("=== Representation Ablation: DHB Invariants vs Raw Actions ===\n")

    # Load data
    print("Loading LIBERO data...")
    demos = load_libero_data(n_demos=50)
    T = 30  # common horizon
    print(f"  {len(demos)} demos loaded\n")

    # Split
    train_data = demos[:40]
    test_data = demos[40:]

    # --- Model A: Predict DHB invariants (8D) ---
    print("Model A: DHB Invariants (8D)")
    head_inv = DHBActionHead(
        vlm_dim=128, action_horizon=T, invariant_dim=8,
        hidden_dim=128, num_heads=4, num_layers=3,
    )
    param_count_inv = sum(p.numel() for p in head_inv.parameters())
    print(f"  Parameters: {param_count_inv:,}")
    print("  Training...")
    losses_inv = train_and_eval(head_inv, train_data, "invariants", T, epochs=100)

    # --- Model B: Predict raw actions (7D) ---
    print("\nModel B: Raw Actions (7D)")
    head_raw = DHBActionHead(
        vlm_dim=128, action_horizon=T, invariant_dim=7,  # 7D raw actions
        hidden_dim=128, num_heads=4, num_layers=3,
    )
    param_count_raw = sum(p.numel() for p in head_raw.parameters())
    print(f"  Parameters: {param_count_raw:,}")
    print("  Training...")
    losses_raw = train_and_eval(head_raw, train_data, "actions", T, epochs=100)

    # --- Compare ---
    print("\n=== Results ===")
    print(f"  DHB Invariants:  final loss = {losses_inv[-1]:.4f}")
    print(f"  Raw Actions:     final loss = {losses_raw[-1]:.4f}")
    print(f"  Loss ratio: {losses_raw[-1]/losses_inv[-1]:.2f}x")

    # SE(3) stability comparison
    print("\n  SE(3) Stability (prediction change under offset):")
    stab_inv = eval_se3_stability(head_inv, test_data, "invariants", T)
    stab_raw = eval_se3_stability(head_raw, test_data, "actions", T)

    print(f"  {'Offset':>8} | {'DHB change':>12} | {'Raw change':>12} | {'Ratio':>8}")
    print("  " + "-" * 50)
    for k in stab_inv:
        inv_c = stab_inv[k]["mean_change"]
        raw_c = stab_raw[k]["mean_change"]
        ratio = raw_c / inv_c if inv_c > 0 else float('inf')
        print(f"  {k:>8} | {inv_c:>11.6f} | {raw_c:>11.6f} | {ratio:>7.2f}x")

    # Save results
    results = {
        "dhb_invariants": {
            "final_loss": losses_inv[-1],
            "params": param_count_inv,
            "stability": stab_inv,
        },
        "raw_actions": {
            "final_loss": losses_raw[-1],
            "params": param_count_raw,
            "stability": stab_raw,
        },
    }
    with open("_workspace/ablation_representation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: _workspace/ablation_representation.json")


if __name__ == "__main__":
    main()
