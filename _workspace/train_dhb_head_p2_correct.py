#!/usr/bin/env python3
"""
CORRECT P2-augmented training for DHBActionHead spatial generalization.

Key: P2 observations get P2-ADAPTED targets (Fatrop-computed invariants
toward new plate position), not original targets.
"""
import sys, time, numpy as np, gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class P2CorrectDataset(Dataset):
    def __init__(self, dataset_dir, p2_cache, p2_targets, action_horizon=100):
        import pandas as pd, glob
        from io import BytesIO
        from PIL import Image as PILImage
        from dhb_xr.encoder.dhb_dr import encode_dhb_dr
        from scipy.spatial.transform import Rotation

        self.action_horizon = action_horizon
        self.samples = []

        # Load P2 targets (Fatrop-adapted)
        p2_tgt = np.load(p2_targets, allow_pickle=True)
        p2_invs = p2_tgt["p2_targets"]   # (50, T, 8) adapted invariants
        orig_invs = p2_tgt["orig_targets"]  # (50, T, 8) original invariants

        # Load P2 observations
        p2_cache_data = np.load(p2_cache, allow_pickle=True)

        # Load original observations from parquet
        pfiles = sorted(glob.glob(f"{dataset_dir}/data/chunk-*/file-*.parquet"))
        episode_seen = set()
        orig_samples = []
        for pf in pfiles:
            if len(episode_seen) >= 50: break
            try: df = pd.read_parquet(pf)
            except: continue
            for ep_idx in sorted(df['episode_index'].unique()):
                if ep_idx in episode_seen or len(episode_seen) >= 50: continue
                episode_seen.add(ep_idx)
                ep_df = df[df['episode_index'] == ep_idx].sort_values('frame_index')
                if len(ep_df) < 5: continue
                first = ep_df.iloc[0]
                img_data = first["observation.images.image"]
                if isinstance(img_data, dict):
                    img1_bytes = img_data['bytes']
                    img2_data = first["observation.images.image2"]
                    img2_bytes = img2_data['bytes'] if isinstance(img2_data, dict) else img2_data
                else:
                    img1_bytes = img_data
                    img2_bytes = first["observation.images.image2"]
                state = torch.tensor(first["observation.state"], dtype=torch.float32)
                orig_samples.append({"img1": img1_bytes, "img2": img2_bytes, "state": state})
            del df; gc.collect()

        T = action_horizon
        n_orig = len(orig_samples)
        n_p2 = 0  # will be set after loading

        def pad(inv, T):
            inv_t = inv[:T].astype(np.float32)
            if len(inv_t) < T:
                inv_t = np.concatenate([inv_t, np.zeros((T-len(inv_t), 8), dtype=np.float32)])
            return torch.from_numpy(inv_t)

        # Add original samples with original targets
        for i, s in enumerate(orig_samples):
            self.samples.append({
                "img1_bytes": s["img1"], "img2_bytes": s["img2"],
                "state": s["state"],
                "invariants": pad(orig_invs[i] if i < len(orig_invs) else np.zeros((T,8)), T),
                "source": "original"
            })

        # Add P2 samples with ADAPTED targets
        if "images_main" in p2_cache_data:
            imgs1_p2 = p2_cache_data["images_main"]  # (50, 256, 256, 3) uint8
            imgs2_p2 = p2_cache_data["images_eih"]
            states_p2 = p2_cache_data["states"]
            for i in range(len(imgs1_p2)):
                self.samples.append({
                    "img1_bytes": imgs1_p2[i],  # ndarray (256, 256, 3)
                    "img2_bytes": imgs2_p2[i],
                    "state": torch.from_numpy(states_p2[i].astype(np.float32)),
                    "invariants": pad(p2_invs[i] if i < len(p2_invs) else np.zeros((T,8)), T),
                    "source": "p2_adapted"
                })
        n_p2 = sum(1 for s in self.samples if s["source"] == "p2_adapted")

        # Per-channel normalization
        all_inv = np.stack([s["invariants"].numpy() for s in self.samples])
        self.inv_mean = all_inv.reshape(-1,8).mean(0).astype(np.float32)
        self.inv_std = np.maximum(all_inv.reshape(-1,8).std(0), 1e-6).astype(np.float32)
        for s in self.samples:
            s["invariants"] = torch.from_numpy((s["invariants"].numpy() - self.inv_mean) / self.inv_std)

        n_p = sum(1 for s in self.samples if s["source"] == "p2_adapted")
        n_o = sum(1 for s in self.samples if s["source"] == "original")
        print(f"  Dataset: {n_o} original + {n_p} P2-adapted = {len(self.samples)} total")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        from io import BytesIO
        from PIL import Image as PILImage
        s = self.samples[idx]
        def decode(raw):
            if isinstance(raw, bytes):
                arr = np.array(PILImage.open(BytesIO(raw)).convert('RGB')).astype(np.float32)/255.
                return torch.from_numpy(arr).permute(2,0,1)
            return torch.from_numpy(raw.astype(np.float32)/255.).permute(2,0,1) if hasattr(raw,'shape') else None
        img1 = torch.flip(decode(s["img1_bytes"]), dims=[1,2])
        img2 = torch.flip(decode(s["img2_bytes"]), dims=[1,2])
        return img1, img2, s["state"], s["invariants"]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="_workspace/lerobot_v3_dhb_task0")
    parser.add_argument("--p2-cache", default="_workspace/p2_obs_cache.npz")
    parser.add_argument("--p2-targets", default="_workspace/p2_aug_targets.npz")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out", default="_workspace/dhb_head_p2_correct.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load SmolVLA + LoRA
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
    import lerobot.policies.smolvla.processor_smolvla
    from lerobot.policies.factory import make_pre_post_processors
    from peft import LoraConfig, get_peft_model
    from dhb_xr.integration.vla.action_head import DHBActionHead

    model_id = "HuggingFaceVLA/smolvla_libero"
    policy = SmolVLAPolicy.from_pretrained(model_id).to(device).eval()
    preprocessor, _ = make_pre_post_processors(policy.config, pretrained_path=model_id)
    policy._preprocessor = preprocessor
    for p in policy.model.parameters(): p.requires_grad = False
    vlm_model = policy.model.vlm_with_expert.get_vlm_model().text_model
    lora_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"], lora_dropout=0.05, bias="none")
    vlm_lora = get_peft_model(vlm_model, lora_cfg)
    vlm_lora.print_trainable_parameters()
    policy.model.vlm_with_expert.vlm.model.text_model = vlm_lora

    vlm_dim = policy.model.vlm_with_expert.vlm.config.text_config.hidden_size
    head = DHBActionHead(vlm_dim=vlm_dim, action_horizon=100, invariant_dim=8,
                         hidden_dim=256, num_heads=8, num_layers=4).to(device)
    print(f"DHBActionHead: {sum(p.numel() for p in head.parameters()):,} params")

    print("\nLoading dataset...")
    dataset = P2CorrectDataset(args.dataset, args.p2_cache, args.p2_targets, action_horizon=100)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)

    TASK_NAME = "pick up the black bowl between the plate and the ramekin and place it on the plate\n"
    lora_params = [p for p in policy.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": head.parameters(), "lr": 3e-4},
                                    {"params": lora_params, "lr": 1e-5}], weight_decay=0.01)

    def get_vlm(img1, img2, state_t):
        batch = {"observation.images.image": img1.to(device),
                 "observation.images.image2": img2.to(device),
                 "observation.state": state_t.to(device), "task": [TASK_NAME]*img1.shape[0]}
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
        (vlm_out, _), _ = policy.model.vlm_with_expert.forward(
            attention_mask=a2d, position_ids=pos_ids, past_key_values=None,
            inputs_embeds=[prefix_embs, dummy], use_cache=False, fill_kv_cache=False)
        return vlm_out.float()

    print(f"\nTraining ({args.epochs} epochs, correct P2 targets)...")
    policy.eval(); head.train()
    best = float("inf")

    for epoch in range(args.epochs):
        total = n = 0
        for img1, img2, state, tgt in loader:
            vlm = get_vlm(img1, img2, state)
            loss = head(vlm, target_invariants=tgt.to(device))["loss"]
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(head.parameters())+lora_params, 1.0)
            optimizer.step()
            total += loss.item(); n += 1
        avg = total/max(n,1)
        if avg < best:
            best = avg
            lora_state = {k:v for k,v in policy.model.state_dict().items() if "lora" in k.lower()}
            torch.save({"dhb_head_state_dict": head.state_dict(), "vlm_lora_state_dict": lora_state,
                        "epoch": epoch+1, "loss": avg,
                        "config": {"vlm_dim": vlm_dim, "action_horizon": 100, "invariant_dim": 8,
                                   "hidden_dim": 256, "num_heads": 8, "num_layers": 4, "lora_rank": 8},
                        "inv_mean": dataset.inv_mean.tolist(), "inv_std": dataset.inv_std.tolist()}, args.out)
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{args.epochs}: loss={avg:.6f}")

    print(f"\nDone! Best: {best:.6f}, saved: {args.out}")

if __name__ == "__main__":
    main()
