#!/usr/bin/env python3
"""
Step 3.4 (Rigorous): Train DHBActionHead on SPATIALLY DIVERSE observations.

Unlike train_dhb_head_e2e.py which only uses 0cm observations, this script
trains on the full DHB-augmented dataset (_workspace/lerobot_v3_dhb_task0/)
which contains 500 episodes with rendered images at ±5cm offset positions.

This makes the spatial robustness claim rigorous:
- Training: VLM sees diverse workspaces (0cm + ±5cm offset images)
- Testing: VLM should predict consistent invariants at UNSEEN offsets

Architecture:
  Rendered offset image → SmolVLA VLM (LoRA) → DHBActionHead → DHB invariants
  → decode from observed EE pose → SE(3) trajectory → embodik IK → success

The key test: if DHB invariants are truly SE(3)-invariant, then training on
augmented images should generalize to arbitrary offsets at test time.

Usage:
    MUJOCO_GL=egl ~/miniforge3/envs/smolvla/bin/python \
        _workspace/train_dhb_head_spatial.py --epochs 200 --out _workspace/dhb_head_spatial.pt
"""
import sys, os, json, time, argparse, gc
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, "/home/andypark/Projects/repos/LIBERO")

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class SpatialDHBDataset(Dataset):
    """
    Loads from the DHB-augmented LeRobot v3 dataset.
    Each episode has images rendered at an offset position and the
    corresponding DHB invariants of the trajectory.

    Since DHB invariants are SE(3)-invariant, the ground-truth invariants
    are IDENTICAL across all offset episodes from the same original demo.
    But the VLM input (images) changes with the offset.

    This forces the VLM to learn: "regardless of where the bowl is in the
    image, produce the same geometric invariants."
    """
    def __init__(self, dataset_dir: str, action_horizon: int = 100, max_episodes: int = None):
        import pandas as pd
        import glob
        from io import BytesIO
        from PIL import Image as PILImage

        from dhb_xr.encoder.dhb_dr import encode_dhb_dr
        from scipy.spatial.transform import Rotation

        print(f"Loading parquet files from {dataset_dir}...")
        parquet_files = sorted(glob.glob(f"{dataset_dir}/data/chunk-*/file-*.parquet"))
        if not parquet_files:
            parquet_files = sorted(glob.glob(f"{dataset_dir}/data/train-*.parquet"))
        print(f"  Found {len(parquet_files)} parquet files")

        self.action_horizon = action_horizon
        self.samples = []
        n_loaded = 0
        n_failed = 0
        max_ep = max_episodes if max_episodes else 999999
        episode_seen = set()

        def get_raw(img_data):
            return img_data['bytes'] if isinstance(img_data, dict) else img_data

        for pf in parquet_files:
            if len(episode_seen) >= max_ep:
                break
            try:
                df = pd.read_parquet(pf)
            except Exception:
                continue

            for ep_idx in sorted(df['episode_index'].unique()):
                if ep_idx in episode_seen or len(episode_seen) >= max_ep:
                    continue
                episode_seen.add(ep_idx)

                try:
                    ep_df = df[df['episode_index'] == ep_idx].sort_values('frame_index')
                    if len(ep_df) < 5:
                        continue

                    first = ep_df.iloc[0]
                    state = torch.tensor(first["observation.state"], dtype=torch.float32)

                    # Build EE trajectory from states
                    all_states = np.stack(ep_df["observation.state"].values)
                    ee_positions = all_states[:, :3]
                    aa_seq = all_states[:, 3:6]
                    rots = Rotation.from_rotvec(aa_seq)
                    q_xyzw = rots.as_quat()
                    quats_wxyz = np.column_stack([q_xyzw[:, 3], q_xyzw[:, 0],
                                                  q_xyzw[:, 1], q_xyzw[:, 2]])

                    enc = encode_dhb_dr(ee_positions, quats_wxyz,
                                        method="pos", use_default_initial_frames=True)
                    inv = np.concatenate([enc["linear_motion_invariants"],
                                          enc["angular_motion_invariants"]], axis=1)

                    T = action_horizon
                    inv_t = inv[:T].astype(np.float32)
                    if len(inv_t) < T:
                        inv_t = np.concatenate([inv_t,
                            np.zeros((T - len(inv_t), 8), dtype=np.float32)])

                    self.samples.append({
                        "img1_bytes": get_raw(first["observation.images.image"]),
                        "img2_bytes": get_raw(first["observation.images.image2"]),
                        "state": state,
                        "invariants": torch.from_numpy(inv_t),
                    })
                    n_loaded += 1

                except Exception as e:
                    n_failed += 1
                    if n_failed < 5:
                        print(f"  Episode {ep_idx} failed: {e}")

            del df
            gc.collect()
            if n_loaded % 50 == 0 and n_loaded > 0:
                print(f"  Loaded {n_loaded} episodes...")

        print(f"  Total: {n_loaded} episodes loaded, {n_failed} failed")

        # Per-channel normalization (computed once from invariants, no images needed)
        all_inv = np.stack([s["invariants"].numpy() for s in self.samples])  # (N, T, 8)
        self.inv_mean = all_inv.reshape(-1, 8).mean(axis=0).astype(np.float32)
        self.inv_std = all_inv.reshape(-1, 8).std(axis=0).astype(np.float32)
        self.inv_std = np.maximum(self.inv_std, 1e-6)
        print(f"  Invariant mean: {self.inv_mean}")
        print(f"  Invariant std:  {self.inv_std}")
        gc.collect()  # free the stacked array

        for s in self.samples:
            s["invariants"] = torch.from_numpy(
                (s["invariants"].numpy() - self.inv_mean) / self.inv_std
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from io import BytesIO
        from PIL import Image as PILImage
        s = self.samples[idx]

        def decode(raw):
            if isinstance(raw, bytes):
                img = PILImage.open(BytesIO(raw)).convert('RGB')
                arr = np.array(img).astype(np.float32) / 255.0
                return torch.from_numpy(arr).permute(2, 0, 1)
            # ndarray
            return torch.from_numpy(raw.astype(np.float32) / 255.0).permute(2, 0, 1)

        img1 = torch.flip(decode(s["img1_bytes"]), dims=[1, 2])
        img2 = torch.flip(decode(s["img2_bytes"]), dims=[1, 2])
        return img1, img2, s["state"], s["invariants"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="_workspace/lerobot_v3_dhb_task0")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_vlm", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--action_horizon", type=int, default=100)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--out", default="_workspace/dhb_head_spatial.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load SmolVLA
    print("Loading SmolVLA...")
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
    import lerobot.policies.smolvla.processor_smolvla
    from lerobot.policies.factory import make_pre_post_processors
    from peft import LoraConfig, get_peft_model

    model_id = "HuggingFaceVLA/smolvla_libero"
    policy = SmolVLAPolicy.from_pretrained(model_id).to(device).eval()
    preprocessor, _ = make_pre_post_processors(policy.config, pretrained_path=model_id)
    policy._preprocessor = preprocessor

    # Freeze + add LoRA
    for p in policy.model.parameters():
        p.requires_grad = False
    vlm_model = policy.model.vlm_with_expert.get_vlm_model().text_model
    lora_cfg = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank*2,
                          target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
    vlm_lora = get_peft_model(vlm_model, lora_cfg)
    vlm_lora.print_trainable_parameters()
    policy.model.vlm_with_expert.vlm.model.text_model = vlm_lora

    vlm_dim = policy.model.vlm_with_expert.vlm.config.text_config.hidden_size
    print(f"VLM hidden dim: {vlm_dim}")

    # Create DHBActionHead
    from dhb_xr.integration.vla.action_head import DHBActionHead
    head = DHBActionHead(vlm_dim=vlm_dim, action_horizon=args.action_horizon,
                         invariant_dim=8, hidden_dim=256, num_heads=8, num_layers=4).to(device)
    print(f"DHBActionHead: {sum(p.numel() for p in head.parameters()):,} params")

    # Load spatially diverse dataset
    dataset = SpatialDHBDataset(args.dataset, args.action_horizon, args.max_episodes)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        drop_last=True, num_workers=0)

    task_name = "pick up the black bowl between the plate and the ramekin and place it on the plate\n"

    # VLM forward
    def get_vlm_features(img1, img2, state_t):
        batch = {
            "observation.images.image": img1.to(device),
            "observation.images.image2": img2.to(device),
            "observation.state": state_t.to(device),
            "task": [task_name] * img1.shape[0],
        }
        batch = policy._preprocessor(batch)
        images, img_masks = policy.prepare_images(batch)
        st = policy.prepare_state(batch)
        lt = batch["observation.language.tokens"]
        lm = batch["observation.language.attention_mask"]
        prefix_embs, pp, pa = policy.model.embed_prefix(images, img_masks, lt, lm, state=st)
        ehs = policy.model.vlm_with_expert.expert_hidden_size
        cs = policy.model.config.chunk_size
        B = img1.shape[0]
        dummy = torch.zeros(B, cs, ehs, device=device, dtype=prefix_embs.dtype)
        dp = torch.ones(B, cs, dtype=torch.bool, device=device)
        da = torch.ones(B, cs, dtype=torch.bool, device=device)
        pm = torch.cat([pp, dp], dim=1)
        am = torch.cat([pa, da], dim=1)
        a2d = make_att_2d_masks(pm, am)
        pos_ids = torch.cumsum(pm, dim=1) - 1
        with torch.no_grad() if not head.training else torch.enable_grad():
            (vlm_out, _), _ = policy.model.vlm_with_expert.forward(
                attention_mask=a2d, position_ids=pos_ids, past_key_values=None,
                inputs_embeds=[prefix_embs, dummy], use_cache=False, fill_kv_cache=False)
        return vlm_out.float()

    # Optimizer
    lora_params = [p for p in policy.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": head.parameters(), "lr": args.lr},
        {"params": lora_params, "lr": args.lr_vlm},
    ], weight_decay=0.01)

    # Train
    print(f"\nTraining on {len(dataset)} spatially diverse episodes...")
    policy.eval()
    head.train()
    best_loss = float("inf")

    for epoch in range(args.epochs):
        total_loss = 0
        n = 0
        t0 = time.time()

        for img1, img2, state, target_inv in loader:
            vlm_feat = get_vlm_features(img1, img2, state)
            result = head(vlm_feat, target_invariants=target_inv.to(device))
            loss = result["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(head.parameters()) + lora_params, max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1

        avg = total_loss / max(n, 1)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{args.epochs}: loss={avg:.6f} ({time.time()-t0:.1f}s)")

        if avg < best_loss:
            best_loss = avg
            lora_state = {k: v for k, v in policy.model.state_dict().items()
                          if "lora" in k.lower()}
            torch.save({
                "dhb_head_state_dict": head.state_dict(),
                "vlm_lora_state_dict": lora_state,
                "epoch": epoch + 1,
                "loss": avg,
                "config": {"vlm_dim": vlm_dim, "action_horizon": args.action_horizon,
                            "invariant_dim": 8, "hidden_dim": 256, "num_heads": 8,
                            "num_layers": 4, "lora_rank": args.lora_rank,
                            "training": "spatial_augmented"},
                "inv_mean": dataset.inv_mean.tolist(),
                "inv_std": dataset.inv_std.tolist(),
            }, args.out)

    print(f"\nDone! Best loss: {best_loss:.6f}, checkpoint: {args.out}")


if __name__ == "__main__":
    main()
