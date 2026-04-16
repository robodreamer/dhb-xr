#!/usr/bin/env python3
"""
Control experiment: Train the SAME DHBActionHead architecture to predict
raw actions instead of DHB invariants. This isolates whether the eval gap
is due to (a) invariant decode pipeline or (b) prediction quality.

If this achieves non-zero success, the gap is in invariant decode.
If this also gets 0%, the gap is in the VLM→action prediction quality.

Usage:
    MUJOCO_GL=egl ~/miniforge3/envs/smolvla/bin/python \
        _workspace/train_direct_action_head.py --epochs 200
"""
import sys, os, json, time, argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, "/home/andypark/Projects/repos/LIBERO")

import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--action_horizon", type=int, default=30)
    parser.add_argument("--max_demos", type=int, default=50)
    parser.add_argument("--out", default="_workspace/direct_action_head.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load SmolVLA
    print("Loading SmolVLA...")
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
    import lerobot.policies.smolvla.processor_smolvla
    from lerobot.policies.factory import make_pre_post_processors

    model_id = "HuggingFaceVLA/smolvla_libero"
    policy = SmolVLAPolicy.from_pretrained(model_id).to(device).eval()
    preprocessor, _ = make_pre_post_processors(policy.config, pretrained_path=model_id)
    policy._preprocessor = preprocessor

    # Freeze VLM completely (no LoRA for this control experiment)
    for p in policy.model.parameters():
        p.requires_grad = False

    vlm_dim = policy.model.vlm_with_expert.vlm.config.text_config.hidden_size
    print(f"VLM dim: {vlm_dim}")

    # Create action head (same architecture, predicting 7D actions instead of 8D invariants)
    from dhb_xr.integration.vla.action_head import DHBActionHead
    action_dim = 7  # dx, dy, dz, dax, day, daz, gripper
    head = DHBActionHead(
        vlm_dim=vlm_dim,
        action_horizon=args.action_horizon,
        invariant_dim=action_dim,  # 7D raw actions
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
    ).to(device)
    print(f"Action head: {sum(p.numel() for p in head.parameters()):,} params")

    # Load LIBERO demos — extract observations + raw actions
    print("Loading LIBERO demos...")
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from scipy.spatial.transform import Rotation
    import h5py

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    task_name = task.name.replace("_", " ") + "\n"
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    dpath = os.path.join(get_libero_path("datasets"), "libero_spatial", f"{task.name}_demo.hdf5")

    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                             camera_names=["agentview", "robot0_eye_in_hand"])
    env.seed(0)

    samples = []
    T = args.action_horizon
    with h5py.File(dpath) as f:
        for i, key in enumerate(sorted(f["data"].keys())):
            if i >= args.max_demos:
                break
            d = f["data"][key]
            init_state = np.array(d["states"][0])
            actions = np.array(d["actions"])  # (L, 7)

            # Pad/truncate actions to T
            act_t = actions[:T].astype(np.float32)
            if len(act_t) < T:
                act_t = np.concatenate([act_t, np.zeros((T - len(act_t), 7), dtype=np.float32)])

            # Get observation
            env.reset()
            obs = env.set_init_state(init_state)
            for _ in range(5):
                obs, _, _, _ = env.step([0.0] * 7)

            agentview = obs["agentview_image"].astype(np.uint8)
            eih = obs["robot0_eye_in_hand_image"].astype(np.uint8)
            eef_pos = obs["robot0_eef_pos"]
            eef_quat = obs["robot0_eef_quat"]
            aa = Rotation.from_quat(eef_quat).as_rotvec().astype(np.float32)
            gripper = obs["robot0_gripper_qpos"]
            state = np.concatenate([eef_pos, aa, gripper]).astype(np.float32)

            samples.append({
                "agentview": agentview,
                "eye_in_hand": eih,
                "state": state,
                "actions": act_t,
            })
            if (i + 1) % 10 == 0:
                print(f"  Loaded {i+1}")

    env.close()
    print(f"  Total: {len(samples)} demos")

    # VLM feature extraction helper
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
        expert_hs = policy.model.vlm_with_expert.expert_hidden_size
        chunk = policy.model.config.chunk_size
        B = img1.shape[0]
        dummy = torch.zeros(B, chunk, expert_hs, device=device, dtype=prefix_embs.dtype)
        dp = torch.ones(B, chunk, dtype=torch.bool, device=device)
        da = torch.ones(B, chunk, dtype=torch.bool, device=device)
        pm = torch.cat([pp, dp], dim=1)
        am = torch.cat([pa, da], dim=1)
        a2d = make_att_2d_masks(pm, am)
        pos = torch.cumsum(pm, dim=1) - 1
        with torch.no_grad():
            (vlm_out, _), _ = policy.model.vlm_with_expert.forward(
                attention_mask=a2d, position_ids=pos, past_key_values=None,
                inputs_embeds=[prefix_embs, dummy], use_cache=False, fill_kv_cache=False)
        return vlm_out.float()

    # Train
    print(f"\nTraining ({args.epochs} epochs)...")
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    head.train()
    best_loss = float("inf")

    for epoch in range(args.epochs):
        indices = np.random.permutation(len(samples))
        total_loss = 0
        n = 0
        for start in range(0, len(indices) - args.batch_size, args.batch_size):
            batch_idx = indices[start:start + args.batch_size]
            imgs1, imgs2, states, targets = [], [], [], []
            for idx in batch_idx:
                s = samples[idx]
                img1 = torch.from_numpy(s["agentview"]).permute(2, 0, 1).float() / 255.0
                img2 = torch.from_numpy(s["eye_in_hand"]).permute(2, 0, 1).float() / 255.0
                imgs1.append(torch.flip(img1, dims=[1, 2]))
                imgs2.append(torch.flip(img2, dims=[1, 2]))
                states.append(torch.from_numpy(s["state"]))
                targets.append(torch.from_numpy(s["actions"]))

            img1_batch = torch.stack(imgs1)
            img2_batch = torch.stack(imgs2)
            state_batch = torch.stack(states)
            target_batch = torch.stack(targets).to(device)

            vlm_feat = get_vlm_features(img1_batch, img2_batch, state_batch)
            result = head(vlm_feat, target_invariants=target_batch)
            loss = result["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1

        avg = total_loss / max(n, 1)
        if avg < best_loss:
            best_loss = avg
            torch.save({
                "state_dict": head.state_dict(),
                "config": {"vlm_dim": vlm_dim, "action_dim": action_dim,
                           "action_horizon": T, "type": "direct_action"},
                "epoch": epoch + 1, "loss": avg,
            }, args.out)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: loss={avg:.6f}")

    print(f"\nDone! Best loss: {best_loss:.6f}")
    print(f"Checkpoint: {args.out}")


if __name__ == "__main__":
    main()
