#!/usr/bin/env python3
"""
Step 3.4 (Option A): End-to-end training of DHBActionHead with SmolVLA VLM.

Architecture:
    Image + Task → SmolVLA VLM (prefix-only, with LoRA) → (B, S, 2048)
    → DHBActionHead cross-attention → (B, T, 8) DHB invariants
    → MSE loss vs ground-truth invariants from LIBERO demos

Key difference from train_dhb_action_head.py:
    - VLM features are computed LIVE during training (not pre-cached)
    - LoRA adapters on VLM backbone allow gradients to flow back
    - DHBActionHead receives real visual features, not random/cached tensors

Usage:
    MUJOCO_GL=egl ~/miniforge3/envs/smolvla/bin/python \
        _workspace/train_dhb_head_e2e.py --epochs 50 --lr 3e-4

Requires: smolvla env with peft installed
    ~/miniforge3/envs/smolvla/bin/pip install peft
"""
import sys, os, json, time, argparse, gc, shutil
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

LIBERO_IMPORT_CANDIDATES = [
    "/home/andypark/Projects/repos/LIBERO",
    "/home/andypark/Projects/repos/LIBERO-PRO",
]
for _candidate in LIBERO_IMPORT_CANDIDATES:
    if Path(_candidate).exists():
        sys.path.insert(0, _candidate)
        break

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ── Dataset: pre-extract observations + DHB invariants from LIBERO ──────

class LiberoDHBDataset(Dataset):
    """
    Yields (obs_dict, dhb_invariants, task_name) for each demo.
    obs_dict contains images + state needed for SmolVLA prefix embedding.
    dhb_invariants are the ground-truth targets.
    """
    def __init__(self, task_id=0, max_demos=50, action_horizon=100):
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        from dhb_xr.encoder.dhb_dr import encode_dhb_dr
        from scipy.spatial.transform import Rotation
        import h5py

        suite = benchmark.get_benchmark_dict()["libero_spatial"]()
        task = suite.get_task(task_id)
        self.task_name = task.name.replace("_", " ") + "\n"
        bddl = os.path.join(get_libero_path("bddl_files"),
                            task.problem_folder, task.bddl_file)
        dpath = os.path.join(get_libero_path("datasets"), "libero_spatial",
                             f"{task.name}_demo.hdf5")

        # Create env for rendering observations
        env = OffScreenRenderEnv(**{
            "bddl_file_name": bddl,
            "camera_heights": 256, "camera_widths": 256,
            "camera_names": ["agentview", "robot0_eye_in_hand"],
        })
        env.seed(0)

        self.samples = []
        print(f"Loading {max_demos} demos from LIBERO task {task_id}...")

        with h5py.File(dpath) as f:
            for i, key in enumerate(sorted(f["data"].keys())):
                if i >= max_demos:
                    break
                d = f["data"][key]
                init_state = np.array(d["states"][0])
                ee_pos = np.array(d["obs"]["ee_pos"])

                # Load raw actions (7D: dx, dy, dz, dax, day, daz, gripper)
                raw_actions = np.array(d["actions"]).astype(np.float32)

                # Get orientations
                if "ee_states" in d["obs"]:
                    euler = np.array(d["obs"]["ee_states"])[:, 3:]
                    rots = Rotation.from_euler("xyz", euler)
                    q_xyzw = rots.as_quat()
                    quats = np.column_stack([q_xyzw[:, 3], q_xyzw[:, 0],
                                             q_xyzw[:, 1], q_xyzw[:, 2]])
                else:
                    quats = np.tile([1, 0, 0, 0], (len(ee_pos), 1))

                # Encode DHB invariants
                try:
                    enc = encode_dhb_dr(ee_pos, quats, method="pos",
                                        use_default_initial_frames=True)
                    inv = np.concatenate([enc["linear_motion_invariants"],
                                          enc["angular_motion_invariants"]], axis=1)
                except Exception:
                    continue

                # Truncate/pad to action_horizon
                T = action_horizon
                inv_t = inv[:T].astype(np.float32)
                if len(inv_t) < T:
                    inv_t = np.concatenate([inv_t,
                        np.zeros((T - len(inv_t), 8), dtype=np.float32)])

                # Truncate/pad raw actions to action_horizon
                act_t = raw_actions[:T].astype(np.float32)
                if len(act_t) < T:
                    act_t = np.concatenate([act_t,
                        np.zeros((T - len(act_t), 7), dtype=np.float32)])

                # Get initial observation from environment
                env.reset()
                obs = env.set_init_state(init_state)
                for _ in range(5):
                    obs, _, _, _ = env.step([0.0] * 7)

                # Extract images and state
                agentview = obs["agentview_image"].astype(np.uint8)
                eih = obs.get("robot0_eye_in_hand_image", agentview).astype(np.uint8)
                eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
                eef_quat = obs.get("robot0_eef_quat", np.array([0, 0, 0, 1.0]))
                aa = Rotation.from_quat(eef_quat).as_rotvec().astype(np.float32)
                gripper = obs.get("robot0_gripper_qpos", np.zeros(2))
                state = np.concatenate([eef_pos, aa, gripper]).astype(np.float32)

                self.samples.append({
                    "agentview": agentview,      # (256, 256, 3) uint8
                    "eye_in_hand": eih,           # (256, 256, 3) uint8
                    "state": state,               # (8,) float32
                    "invariants": inv_t,           # (T, 8) float32
                    "actions": act_t,              # (T, 7) float32
                })

                if (i + 1) % 10 == 0:
                    print(f"  Loaded {i+1} demos")

        env.close()
        print(f"  Total: {len(self.samples)} demos")

        # Compute per-channel normalization stats for invariants
        all_inv = np.stack([s["invariants"] for s in self.samples])  # (N, T, 8)
        self.inv_mean = all_inv.reshape(-1, 8).mean(axis=0).astype(np.float32)
        self.inv_std = all_inv.reshape(-1, 8).std(axis=0).astype(np.float32)
        self.inv_std = np.maximum(self.inv_std, 1e-6)  # avoid div by zero
        print(f"  Invariant stats:")
        print(f"    mean: {self.inv_mean}")
        print(f"    std:  {self.inv_std}")

        # Compute per-channel normalization stats for raw actions
        all_act = np.stack([s["actions"] for s in self.samples])  # (N, T, 7)
        self.act_mean = all_act.reshape(-1, 7).mean(axis=0).astype(np.float32)
        self.act_std = all_act.reshape(-1, 7).std(axis=0).astype(np.float32)
        self.act_std = np.maximum(self.act_std, 1e-6)
        print(f"  Action stats:")
        print(f"    mean: {self.act_mean}")
        print(f"    std:  {self.act_std}")

        # Normalize all stored invariants and actions
        for s in self.samples:
            s["invariants"] = (s["invariants"] - self.inv_mean) / self.inv_std
            s["actions"] = (s["actions"] - self.act_mean) / self.act_std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # Convert images to tensors: (3, H, W) float [0, 1]
        img1 = torch.from_numpy(s["agentview"]).permute(2, 0, 1).float() / 255.0
        img2 = torch.from_numpy(s["eye_in_hand"]).permute(2, 0, 1).float() / 255.0
        # Flip 180 degrees (SmolVLA convention for LIBERO)
        img1 = torch.flip(img1, dims=[1, 2])
        img2 = torch.flip(img2, dims=[1, 2])
        state = torch.from_numpy(s["state"])
        inv = torch.from_numpy(s["invariants"])  # already normalized
        act = torch.from_numpy(s["actions"])      # already normalized
        return img1, img2, state, inv, act


# ── VLM prefix feature extraction ──────────────────────────────────────

def get_vlm_prefix_features(policy, img1, img2, state, task_name, device):
    """
    Run SmolVLA VLM (prefix only) to get hidden states.

    Uses the policy's preprocessor to handle tokenization, then calls
    embed_prefix + vlm_with_expert.forward with suffix=None to get
    VLM-only hidden states.

    Returns: (B, prefix_len, 2048)
    """
    B = img1.shape[0]

    # Build batch in SmolVLA format
    batch = {
        "observation.images.image": img1.to(device),
        "observation.images.image2": img2.to(device),
        "observation.state": state.to(device),
        "task": [task_name] * B,
    }

    # Apply SmolVLA preprocessor (tokenizes text, normalizes state)
    batch = policy._preprocessor(batch)

    # Prepare images and state (same as SmolVLAPolicy.forward)
    images, img_masks = policy.prepare_images(batch)
    state_t = policy.prepare_state(batch)
    lang_tokens = batch["observation.language.tokens"]
    lang_masks = batch["observation.language.attention_mask"]

    # Embed prefix (images + language + state)
    prefix_embs, prefix_pad_masks, prefix_att_masks = \
        policy.model.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state_t)

    # Run VLM+expert forward with a dummy suffix to get VLM prefix output.
    # SmolVLA's interleaved architecture requires both prefix and suffix;
    # we create a minimal dummy suffix, then discard the suffix output.
    expert_hidden_size = policy.model.vlm_with_expert.expert_hidden_size
    chunk_size = policy.model.config.chunk_size  # 50
    dummy_suffix = torch.zeros(B, chunk_size, expert_hidden_size,
                               device=device, dtype=prefix_embs.dtype)
    dummy_suffix_pad = torch.ones(B, chunk_size, dtype=torch.bool, device=device)
    dummy_suffix_att = torch.ones(B, chunk_size, dtype=torch.bool, device=device)

    pad_masks = torch.cat([prefix_pad_masks, dummy_suffix_pad], dim=1)
    att_masks = torch.cat([prefix_att_masks, dummy_suffix_att], dim=1)
    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1

    (vlm_out, _suffix_out), _ = policy.model.vlm_with_expert.forward(
        attention_mask=att_2d_masks,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, dummy_suffix],
        use_cache=False,
        fill_kv_cache=False,
    )

    return vlm_out.float()  # (B, prefix_len, 2048), cast bf16 → fp32


def make_att_2d_masks(pad_masks, att_masks):
    """Replicate SmolVLA's attention mask construction for prefix-only."""
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks as _make
    return _make(pad_masks, att_masks)


# ── Main training loop ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_vlm", type=float, default=1e-5,
                        help="Learning rate for VLM LoRA adapters")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--action_horizon", type=int, default=100)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--max_demos", type=int, default=50)
    parser.add_argument("--out", default="_workspace/dhb_head_e2e_dual.pt")
    parser.add_argument("--no_lora", action="store_true",
                        help="Freeze VLM completely, only train DHBActionHead")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Config: epochs={args.epochs}, lr={args.lr}, lr_vlm={args.lr_vlm}, "
          f"batch_size={args.batch_size}, lora_rank={args.lora_rank}")

    # ── Load SmolVLA ────────────────────────────────────────────────────
    print("\nLoading SmolVLA...")
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    import lerobot.policies.smolvla.processor_smolvla  # register processor
    from lerobot.policies.factory import make_pre_post_processors

    model_id = "HuggingFaceVLA/smolvla_libero"
    policy = SmolVLAPolicy.from_pretrained(model_id).to(device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=model_id
    )
    policy._preprocessor = preprocessor

    # Freeze all VLM parameters
    for p in policy.model.parameters():
        p.requires_grad = False

    # ── Add LoRA to VLM backbone (optional) ─────────────────────────────
    if not args.no_lora:
        print("Adding LoRA adapters to VLM backbone...")
        from peft import LoraConfig, get_peft_model

        # Target the VLM text model's q_proj and v_proj
        vlm_model = policy.model.vlm_with_expert.get_vlm_model().text_model
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        vlm_model_lora = get_peft_model(vlm_model, lora_config)
        vlm_model_lora.print_trainable_parameters()
        # Replace in the model hierarchy
        policy.model.vlm_with_expert.vlm.model.text_model = vlm_model_lora
    else:
        print("VLM frozen (no LoRA)")

    # ── Create DHBActionHead ────────────────────────────────────────────
    print("\nCreating DHBActionHead...")
    from dhb_xr.integration.vla.action_head import DHBActionHead

    # Get actual VLM hidden size (960 for SmolVLM2-500M, not 2048)
    vlm_dim = policy.model.vlm_with_expert.vlm.config.text_config.hidden_size
    print(f"  VLM hidden size: {vlm_dim}")

    dhb_head = DHBActionHead(
        vlm_dim=vlm_dim,
        action_horizon=args.action_horizon,
        invariant_dim=8,
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        action_dim=7,
    ).to(device)

    param_count = sum(p.numel() for p in dhb_head.parameters())
    print(f"DHBActionHead parameters: {param_count:,} ({param_count/1e6:.1f}M)")

    # ── Load dataset ────────────────────────────────────────────────────
    print("\nLoading LIBERO dataset...")
    dataset = LiberoDHBDataset(
        task_id=0, max_demos=args.max_demos,
        action_horizon=args.action_horizon
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        drop_last=True, num_workers=0)

    # ── Optimizer: separate LR for VLM LoRA and DHB head ────────────────
    param_groups = [
        {"params": dhb_head.parameters(), "lr": args.lr},
    ]
    if not args.no_lora:
        vlm_lora_params = [p for p in policy.model.parameters() if p.requires_grad]
        if vlm_lora_params:
            param_groups.append({
                "params": vlm_lora_params,
                "lr": args.lr_vlm,
            })
            print(f"VLM LoRA trainable params: {sum(p.numel() for p in vlm_lora_params):,}")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

    # ── Training loop ───────────────────────────────────────────────────
    print(f"\nTraining ({args.epochs} epochs, {len(dataset)} demos, "
          f"batch_size={args.batch_size})...")

    task_name = dataset.task_name
    best_loss = float("inf")

    policy.eval()  # Keep policy in eval mode (BN, dropout) but LoRA adapters train
    dhb_head.train()

    for epoch in range(args.epochs):
        total_loss = 0.0
        total_loss_act = 0.0
        total_loss_inv = 0.0
        n_batches = 0
        t0 = time.time()

        for img1, img2, state, target_inv, target_act in loader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            state_t = state.to(device)
            target_inv = target_inv.to(device)
            target_act = target_act.to(device)

            # Forward: VLM prefix → DHBActionHead
            vlm_hidden = get_vlm_prefix_features(
                policy, img1, img2, state_t, task_name, device
            )

            # DHBActionHead forward with dual targets
            result = dhb_head(vlm_hidden, target_invariants=target_inv,
                              target_actions=target_act)
            loss = result["loss"]

            # Backward + step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(dhb_head.parameters()) +
                [p for p in policy.model.parameters() if p.requires_grad],
                max_norm=1.0
            )
            optimizer.step()

            total_loss += loss.item()
            total_loss_act += result.get("loss_act", torch.tensor(0.0)).item()
            total_loss_inv += result.get("loss_inv", torch.tensor(0.0)).item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_loss_act = total_loss_act / max(n_batches, 1)
        avg_loss_inv = total_loss_inv / max(n_batches, 1)
        elapsed = time.time() - t0

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{args.epochs}: "
                  f"loss={avg_loss:.6f} (act={avg_loss_act:.6f} inv={avg_loss_inv:.6f})  "
                  f"({elapsed:.1f}s/epoch)")

        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save best checkpoint
            checkpoint = {
                "dhb_head_state_dict": dhb_head.state_dict(),
                "epoch": epoch + 1,
                "loss": avg_loss,
                "config": {
                    "vlm_dim": vlm_dim,
                    "action_horizon": args.action_horizon,
                    "invariant_dim": 8,
                    "hidden_dim": 256,
                    "num_heads": 8,
                    "num_layers": 4,
                    "lora_rank": args.lora_rank if not args.no_lora else 0,
                    "action_dim": 7,
                },
                # Per-channel normalization stats (critical for eval decode)
                "inv_mean": dataset.inv_mean.tolist(),
                "inv_std": dataset.inv_std.tolist(),
                # Action normalization stats (for denormalization at eval)
                "act_mean": dataset.act_mean.tolist(),
                "act_std": dataset.act_std.tolist(),
            }
            # Save LoRA weights if applicable
            if not args.no_lora:
                lora_state = {k: v for k, v in policy.model.state_dict().items()
                              if "lora" in k.lower()}
                checkpoint["vlm_lora_state_dict"] = lora_state

            torch.save(checkpoint, args.out)

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\nTraining complete!")
    print(f"  Best loss: {best_loss:.6f}")
    print(f"  Checkpoint: {args.out}")
    print(f"  DHBActionHead: {param_count:,} params")
    if not args.no_lora:
        print(f"  VLM LoRA: rank={args.lora_rank}")

    # Save training summary
    summary = {
        "best_loss": best_loss,
        "epochs": args.epochs,
        "lr": args.lr,
        "lr_vlm": args.lr_vlm,
        "lora_rank": args.lora_rank if not args.no_lora else 0,
        "n_demos": len(dataset),
        "action_horizon": args.action_horizon,
        "checkpoint": args.out,
    }
    summary_path = args.out.replace(".pt", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
