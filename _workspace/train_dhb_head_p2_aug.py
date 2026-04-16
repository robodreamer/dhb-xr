#!/usr/bin/env python3
"""
P2-Augmented DHBActionHead Training.

Scientific question: If we train DHBActionHead on observations from P2-perturbed
environments (plate swapped to new position), will it generalize to predict correct
invariants in P2 test scenes?

Key insight:
  DHB invariants are SE(3)-invariant — the target invariants are IDENTICAL for a
  demo trajectory regardless of where the plate is. Only the VLM visual features
  differ between original and P2 scenes. Training on P2 observations teaches the
  VLM to produce consistent invariants under object swap perturbations.

Dataset design:
  - 50 original demos: images from original env, target invariants from trajectory
  - 50 P2-rendered episodes: images from P2 env (plate at new position), SAME
    target invariants (from same underlying demo trajectory — invariants don't change
    with object layout)
  = 100 total training episodes with 2x visual diversity

The VLM must learn: "regardless of where the plate is in the image, output the same
geometric invariants for this task" — which is exactly what SE(3) invariance guarantees.

Usage:
    # Render P2 observations first (libero env):
    MUJOCO_GL=egl ~/miniforge3/envs/libero/bin/python \
        _workspace/train_dhb_head_p2_aug.py --render-p2-only \
        --out-p2-cache _workspace/p2_obs_cache.npz

    # Then train (smolvla env):
    MUJOCO_GL=egl ~/miniforge3/envs/smolvla/bin/python \
        _workspace/train_dhb_head_p2_aug.py \
        --p2-cache _workspace/p2_obs_cache.npz \
        --epochs 200 --out _workspace/dhb_head_p2_aug.pt

    # Or render + train in one go (requires libero env with smolvla packages):
    MUJOCO_GL=egl ~/miniforge3/envs/libero/bin/python \
        _workspace/train_dhb_head_p2_aug.py --epochs 200 \
        --out _workspace/dhb_head_p2_aug.pt

Training time estimate:
  100 episodes, batch_size=2, 200 epochs:
  ~50 batches/epoch × 200 epochs = 10,000 steps
  ~1.5s/step on GPU → ~25 min total
"""
import sys, os, json, time, argparse, gc, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, "/home/andypark/Projects/repos/LIBERO")
sys.path.insert(0, "/home/andypark/Projects/repos/LIBERO-PRO")

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================================================================
# P2 observation renderer (runs in libero env)
# =========================================================================

def render_p2_observations(
    n_demos: int = 50,
    out_cache: str = "_workspace/p2_obs_cache.npz",
    bddl_p2: str = "/tmp/p2_perturbed.bddl",
    resolution: int = 256,
    seed: int = 42,
) -> str:
    """Render initial observations from the P2 perturbed environment.

    Uses env.reset() (not set_init_state) so the BDDL-defined P2 layout is
    preserved. The plate stays at box_region (Y~0.03), not the original
    plate_region (Y~0.2). Each of the n_demos resets gets mild random
    perturbation from the LIBERO seed mechanism.

    The invariants for these P2 observations come from the ORIGINAL demo
    trajectories (computed separately in P2AugmentedDHBDataset). This is
    the core of the approach: same invariants, different visual context.

    Args:
        n_demos: Number of P2 observations to render (should match n original demos)
        out_cache: Output .npz path
        bddl_p2: Path to perturbed BDDL file (plate swapped)
        resolution: Image resolution
        seed: Random seed for env

    Returns:
        Path to output cache file.
    """
    import random
    random.seed(seed)
    os.environ.setdefault("MUJOCO_GL", "egl")

    # Generate perturbed BDDL if it doesn't exist
    if not os.path.exists(bddl_p2):
        print(f"P2 BDDL not found at {bddl_p2}, generating...")
        from perturbation import BDDLParser, SwapPerturbator

        TASK_NAME = "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"
        TASK_SUITE = "libero_spatial"
        LIBERO_PRO_ROOT = Path("/home/andypark/Projects/repos/LIBERO-PRO")
        BDDL_ORIG = str(
            LIBERO_PRO_ROOT / "libero/libero/bddl_files/libero_spatial"
            / f"{TASK_NAME}.bddl"
        )
        OOD_CFG = str(LIBERO_PRO_ROOT / "libero_ood/ood_spatial_relation.yaml")

        with open(BDDL_ORIG) as f:
            content = f.read()
        parser = BDDLParser(content)
        swapper = SwapPerturbator(parser, OOD_CFG)
        perturbed_content = swapper.perturb(TASK_SUITE, TASK_NAME)

        with open(bddl_p2, "w") as f:
            f.write(perturbed_content)
        print(f"  Generated P2 BDDL: {bddl_p2}")

    from libero.libero.envs import OffScreenRenderEnv

    dataset_hdf5 = ("/home/andypark/Projects/data/libero/libero_spatial/"
                    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5")
    print(f"Rendering {n_demos} P2 observations...")
    print(f"  P2 BDDL: {bddl_p2}")
    print(f"  Dataset: {dataset_hdf5}")

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_p2,
        camera_heights=resolution,
        camera_widths=resolution,
        camera_names=["agentview", "robot0_eye_in_hand"],
        horizon=600,
    )
    env.seed(seed)

    images_main = []    # agentview, (H,W,3) uint8
    images_eih = []     # eye-in-hand, (H,W,3) uint8
    states_list = []    # EEF state, (8,) float32
    demo_indices = []   # which demo each rendering corresponds to

    def quat_to_axis_angle(quat_xyzw):
        """Convert quaternion (x,y,z,w) → axis-angle (3,)."""
        q = np.asarray(quat_xyzw, dtype=np.float64)
        w = np.clip(q[3], -1.0, 1.0)
        den = np.sqrt(max(1.0 - w * w, 0.0))
        if den > 1e-10:
            angle = 2.0 * np.arccos(w)
            axis = q[:3] / den
            return (axis * angle).astype(np.float32)
        return np.zeros(3, dtype=np.float32)

    # NOTE on set_init_state behavior:
    # LIBERO's set_init_state() calls sim.set_state_from_flattened(mujoco_state),
    # which restores the FULL MuJoCo state — including plate_1's position.
    # After set_init_state(demo_init), the plate goes back to its ORIGINAL position
    # (plate_region Y~0.2), not the P2 box_region (Y~0.03). This defeats the purpose.
    #
    # Therefore we use env.reset() only (Option A): the BDDL file places plate_1
    # in box_region (Y~0.03) and leaves the robot at the BDDL default joint config.
    # The robot pose will be slightly different from the demo-specific init, but:
    #   - The visual context (plate at P2 position) is correctly captured
    #   - The target invariants (from demo trajectory) are still valid targets
    #   - The VLM just needs to see "plate at Y~0.03" → predict the invariants
    #
    # We render n_demos frames — each is a fresh reset of the P2 env.
    # env.reset() adds small random perturbations (from env.seed), giving mild
    # visual diversity within the P2 layout.

    print(f"  Rendering {n_demos} P2 frames (env.reset() only, no set_init_state)...")
    for i in range(n_demos):
        # Fresh reset each time so the P2 BDDL places plate at box_region.
        obs = env.reset()

        # Stabilize physics
        for _ in range(3):
            obs, _, _, _ = env.step(np.zeros(7))

        # Capture initial observation
        agentview = obs["agentview_image"].astype(np.uint8)
        eih = obs.get("robot0_eye_in_hand_image", agentview).astype(np.uint8)

        eef_pos = obs.get("robot0_eef_pos", np.zeros(3)).astype(np.float32)
        eef_quat_xyzw = obs.get("robot0_eef_quat", np.array([0, 0, 0, 1.0]))
        eef_aa = quat_to_axis_angle(eef_quat_xyzw)
        gripper_qpos = obs.get("robot0_gripper_qpos", np.zeros(2)).astype(np.float32)
        state = np.concatenate([eef_pos, eef_aa, gripper_qpos]).astype(np.float32)

        images_main.append(agentview)
        images_eih.append(eih)
        states_list.append(state)
        # demo_idx: maps to the i-th original demo's invariants (mod 50 to cycle)
        demo_indices.append(i % 50)

        if (i + 1) % 10 == 0:
            print(f"  Rendered {i + 1}/{n_demos} P2 observations...")

    env.close()

    out_cache = str(out_cache)
    np.savez(
        out_cache,
        images_main=np.array(images_main),     # (N, H, W, 3) uint8
        images_eih=np.array(images_eih),       # (N, H, W, 3) uint8
        states=np.array(states_list),          # (N, 8) float32
        demo_indices=np.array(demo_indices),   # (N,) int
    )
    print(f"  Saved {len(images_main)} P2 observations to: {out_cache}")
    return out_cache


# =========================================================================
# Mixed dataset (original 50 + P2 50)
# =========================================================================

class P2AugmentedDHBDataset(Dataset):
    """Mixed dataset: original spatial demos + P2-perturbed observations.

    Both halves share the SAME target invariants (from the original demo
    trajectory) — only the visual observations differ.

    This design directly tests the hypothesis: a VLM trained on P2 visual
    inputs can predict the same invariants as for the original inputs.

    Dataset breakdown:
      - Samples 0..N_orig-1:   Original observations (from lerobot_v3_dhb_task0)
        * Specifically, episodes with offset=0 (the 50 original demos)
      - Samples N_orig..N-1:   P2 observations (from p2_obs_cache.npz)
        * Same 50 underlying demos, but plate is in swapped position

    Normalization: computed jointly over all 100 episodes so that
    original and P2 samples live in the same normalized space.
    """

    def __init__(
        self,
        dataset_dir: str,
        p2_cache_path: str,
        action_horizon: int = 30,
        max_orig_episodes: int = 50,
    ):
        import pandas as pd
        import glob
        from io import BytesIO
        from PIL import Image as PILImage
        from dhb_xr.encoder.dhb_dr import encode_dhb_dr
        from scipy.spatial.transform import Rotation

        self.action_horizon = action_horizon
        self.samples = []

        # ── Part 1: Load original observations ──────────────────────────
        print(f"[P2AugDataset] Loading original episodes from {dataset_dir}...")
        parquet_files = sorted(glob.glob(f"{dataset_dir}/data/chunk-*/file-*.parquet"))
        if not parquet_files:
            parquet_files = sorted(glob.glob(f"{dataset_dir}/data/train-*.parquet"))
        print(f"  Found {len(parquet_files)} parquet files")

        n_orig = 0
        n_failed = 0
        episode_seen = set()

        def get_raw(img_data):
            return img_data['bytes'] if isinstance(img_data, dict) else img_data

        # We want only the first max_orig_episodes unique episodes (the
        # original-offset demos — episode 0..49 in the 500-episode dataset
        # correspond to the 50 original demos at 0cm offset).
        # The dataset interleaves original+augmented, so we take the first
        # max_orig_episodes distinct episode indices encountered.
        for pf in parquet_files:
            if len(episode_seen) >= max_orig_episodes:
                break
            try:
                df = pd.read_parquet(pf)
            except Exception:
                continue

            for ep_idx in sorted(df['episode_index'].unique()):
                if ep_idx in episode_seen or len(episode_seen) >= max_orig_episodes:
                    continue
                episode_seen.add(ep_idx)

                try:
                    ep_df = df[df['episode_index'] == ep_idx].sort_values('frame_index')
                    if len(ep_df) < 5:
                        continue

                    first = ep_df.iloc[0]
                    state = torch.tensor(first["observation.state"], dtype=torch.float32)

                    # Build EE trajectory → DHB invariants
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
                        "source": "original",
                        "demo_idx": ep_idx,
                    })
                    n_orig += 1

                except Exception as e:
                    n_failed += 1
                    if n_failed < 5:
                        print(f"  Episode {ep_idx} failed: {e}")

            del df
            gc.collect()

        print(f"  Loaded {n_orig} original episodes ({n_failed} failed)")

        # ── Part 2: Load P2 observations ────────────────────────────────
        print(f"[P2AugDataset] Loading P2 observations from {p2_cache_path}...")
        p2 = np.load(p2_cache_path)
        p2_images_main = p2["images_main"]   # (N, H, W, 3) uint8
        p2_images_eih = p2["images_eih"]     # (N, H, W, 3) uint8
        p2_states = p2["states"]             # (N, 8) float32
        p2_demo_indices = p2["demo_indices"] # (N,) int

        # We need DHB invariants for each P2 episode.
        # The P2 observations correspond to demo_idx 0..N_p2-1.
        # We use the SAME invariants as the original demo (by demo index).
        # Build a lookup: demo_idx → invariants tensor
        demo_inv_lookup = {}
        for s in self.samples:
            demo_inv_lookup[s["demo_idx"]] = s["invariants"]

        n_p2 = 0
        n_p2_failed = 0
        for i in range(len(p2_images_main)):
            demo_idx = int(p2_demo_indices[i])
            if demo_idx not in demo_inv_lookup:
                # We didn't load this original demo — skip or recompute
                # (shouldn't happen if max_orig_episodes >= n_p2 demos)
                n_p2_failed += 1
                continue

            self.samples.append({
                "img1_array": p2_images_main[i],   # (H,W,3) uint8 ndarray
                "img2_array": p2_images_eih[i],    # (H,W,3) uint8 ndarray
                "state": torch.tensor(p2_states[i], dtype=torch.float32),
                # KEY: Same invariants as the original demo!
                "invariants": demo_inv_lookup[demo_idx].clone(),
                "source": "p2",
                "demo_idx": demo_idx,
            })
            n_p2 += 1

        print(f"  Loaded {n_p2} P2 episodes ({n_p2_failed} skipped)")
        print(f"  Total dataset: {len(self.samples)} episodes "
              f"({n_orig} original + {n_p2} P2)")

        # ── Normalization (computed jointly) ────────────────────────────
        all_inv = np.stack([s["invariants"].numpy() for s in self.samples])  # (N, T, 8)
        self.inv_mean = all_inv.reshape(-1, 8).mean(axis=0).astype(np.float32)
        self.inv_std = all_inv.reshape(-1, 8).std(axis=0).astype(np.float32)
        self.inv_std = np.maximum(self.inv_std, 1e-6)
        print(f"  Invariant mean: {self.inv_mean.round(4)}")
        print(f"  Invariant std:  {self.inv_std.round(4)}")

        for s in self.samples:
            s["invariants"] = torch.from_numpy(
                (s["invariants"].numpy() - self.inv_mean) / self.inv_std
            )
        gc.collect()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from io import BytesIO
        from PIL import Image as PILImage
        s = self.samples[idx]

        def decode_bytes(raw):
            if isinstance(raw, bytes):
                img = PILImage.open(BytesIO(raw)).convert('RGB')
                arr = np.array(img).astype(np.float32) / 255.0
                return torch.from_numpy(arr).permute(2, 0, 1)
            return torch.from_numpy(raw.astype(np.float32) / 255.0).permute(2, 0, 1)

        def decode_array(arr):
            return torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1)

        if s["source"] == "original":
            # Bytes from parquet; LIBERO renders upside-down → flip
            img1 = torch.flip(decode_bytes(s["img1_bytes"]), dims=[1, 2])
            img2 = torch.flip(decode_bytes(s["img2_bytes"]), dims=[1, 2])
        else:
            # P2 arrays from numpy; also flip (same rendering pipeline)
            img1 = torch.flip(decode_array(s["img1_array"]), dims=[1, 2])
            img2 = torch.flip(decode_array(s["img2_array"]), dims=[1, 2])

        return img1, img2, s["state"], s["invariants"]


# =========================================================================
# Training
# =========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="_workspace/lerobot_v3_dhb_task0",
                        help="Path to the spatial DHB LeRobot dataset")
    parser.add_argument("--p2-cache", default="_workspace/p2_obs_cache.npz",
                        help="Path to P2 observation cache (.npz)")
    parser.add_argument("--bddl-p2", default="/tmp/p2_perturbed.bddl",
                        help="Path to P2 perturbed BDDL file")
    parser.add_argument("--render-p2-only", action="store_true",
                        help="Only render P2 observations (no training)")
    parser.add_argument("--n-p2-demos", type=int, default=50,
                        help="Number of P2 observations to render")
    parser.add_argument("--max-orig-episodes", type=int, default=50,
                        help="Max original episodes to use from spatial dataset")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-vlm", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, default=30)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--out", default="_workspace/dhb_head_p2_aug.pt")
    parser.add_argument("--out-p2-cache", default="_workspace/p2_obs_cache.npz",
                        help="Output path for P2 cache (used with --render-p2-only)")
    args = parser.parse_args()

    # ── Step 1: Render P2 observations if needed ────────────────────────
    p2_cache = args.p2_cache if not args.render_p2_only else args.out_p2_cache

    if args.render_p2_only or not os.path.exists(p2_cache):
        print("=" * 60)
        print("Rendering P2 observations (requires libero env)...")
        print("=" * 60)
        rendered_path = render_p2_observations(
            n_demos=args.n_p2_demos,
            out_cache=p2_cache,
            bddl_p2=args.bddl_p2,
        )
        if args.render_p2_only:
            print(f"\nDone! P2 observations saved to: {rendered_path}")
            print("Now run training with:")
            print(f"  ~/miniforge3/envs/smolvla/bin/python "
                  f"_workspace/train_dhb_head_p2_aug.py "
                  f"--p2-cache {rendered_path} "
                  f"--epochs {args.epochs} --out {args.out}")
            return

    # ── Step 2: Train ────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

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

    # Freeze backbone + add LoRA to VLM text model
    for p in policy.model.parameters():
        p.requires_grad = False
    vlm_model = policy.model.vlm_with_expert.get_vlm_model().text_model
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    vlm_lora = get_peft_model(vlm_model, lora_cfg)
    vlm_lora.print_trainable_parameters()
    policy.model.vlm_with_expert.vlm.model.text_model = vlm_lora

    vlm_dim = policy.model.vlm_with_expert.vlm.config.text_config.hidden_size
    print(f"VLM hidden dim: {vlm_dim}")

    # Create DHBActionHead
    from dhb_xr.integration.vla.action_head import DHBActionHead
    head = DHBActionHead(
        vlm_dim=vlm_dim,
        action_horizon=args.action_horizon,
        invariant_dim=8,
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
    ).to(device)
    print(f"DHBActionHead: {sum(p.numel() for p in head.parameters()):,} params")

    # Load mixed dataset
    dataset = P2AugmentedDHBDataset(
        dataset_dir=args.dataset,
        p2_cache_path=p2_cache,
        action_horizon=args.action_horizon,
        max_orig_episodes=args.max_orig_episodes,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    task_name = "pick up the black bowl between the plate and the ramekin and place it on the plate\n"

    # VLM forward pass
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

    # Optimizer: head gets higher LR, LoRA gets lower LR
    lora_params = [p for p in policy.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": head.parameters(), "lr": args.lr},
        {"params": lora_params, "lr": args.lr_vlm},
    ], weight_decay=0.01)

    # Training loop
    n_orig = sum(1 for s in dataset.samples if s["source"] == "original")
    n_p2 = sum(1 for s in dataset.samples if s["source"] == "p2")
    print(f"\nTraining on {len(dataset)} episodes: {n_orig} original + {n_p2} P2")
    print(f"Epochs: {args.epochs}, batch_size: {args.batch_size}")
    print(f"Batches/epoch: {len(loader)}")

    policy.eval()
    head.train()
    best_loss = float("inf")

    for epoch in range(args.epochs):
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for img1, img2, state, target_inv in loader:
            vlm_feat = get_vlm_features(img1, img2, state)
            result = head(vlm_feat, target_invariants=target_inv.to(device))
            loss = result["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(head.parameters()) + lora_params, max_norm=1.0
            )
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{args.epochs}: "
                  f"loss={avg_loss:.6f} ({time.time()-t0:.1f}s)")

        if avg_loss < best_loss:
            best_loss = avg_loss
            lora_state = {k: v for k, v in policy.model.state_dict().items()
                          if "lora" in k.lower()}
            torch.save({
                "dhb_head_state_dict": head.state_dict(),
                "vlm_lora_state_dict": lora_state,
                "epoch": epoch + 1,
                "loss": avg_loss,
                "config": {
                    "vlm_dim": vlm_dim,
                    "action_horizon": args.action_horizon,
                    "invariant_dim": 8,
                    "hidden_dim": 256,
                    "num_heads": 8,
                    "num_layers": 4,
                    "lora_rank": args.lora_rank,
                    "training": "p2_augmented",
                    "n_orig_episodes": n_orig,
                    "n_p2_episodes": n_p2,
                },
                "inv_mean": dataset.inv_mean.tolist(),
                "inv_std": dataset.inv_std.tolist(),
            }, args.out)

    print(f"\nDone! Best loss: {best_loss:.6f}, checkpoint: {args.out}")


if __name__ == "__main__":
    main()
