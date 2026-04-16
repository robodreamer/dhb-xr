#!/usr/bin/env python3
"""
Step 3.5: Evaluate end-to-end trained DHBActionHead on LIBERO.

Loads the checkpoint from train_dhb_head_e2e.py, runs closed-loop
evaluation at 0cm, 3cm, 5cm offsets.

Pipeline per step:
    1. Observe → SmolVLA VLM prefix features (live, not cached)
    2. DHBActionHead → predict DHB invariants
    3. decode_dhb from current EE pose → SE(3) trajectory
    4. Execute via OSC delta controller

Usage:
    MUJOCO_GL=egl ~/miniforge3/envs/smolvla/bin/python \
        _workspace/eval_dhb_head_e2e.py \
        --checkpoint _workspace/dhb_head_e2e.pt \
        --offsets 0 3 5 --seeds 10
"""
import sys, os, json, time, argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, "/home/andypark/Projects/repos/LIBERO")

import torch


def load_model(checkpoint_path, device="cuda"):
    """Load SmolVLA + DHBActionHead from checkpoint."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    import lerobot.policies.smolvla.processor_smolvla
    from lerobot.policies.factory import make_pre_post_processors
    from dhb_xr.integration.vla.action_head import DHBActionHead

    print("Loading SmolVLA...")
    model_id = "HuggingFaceVLA/smolvla_libero"
    policy = SmolVLAPolicy.from_pretrained(model_id).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=model_id
    )
    policy._preprocessor = preprocessor

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]

    # Create DHBActionHead with saved config
    dhb_head = DHBActionHead(
        vlm_dim=config["vlm_dim"],
        action_horizon=config["action_horizon"],
        invariant_dim=config["invariant_dim"],
        hidden_dim=config["hidden_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        action_dim=config.get("action_dim", None),
    ).to(device)
    dhb_head.load_state_dict(ckpt["dhb_head_state_dict"])
    dhb_head.eval()

    # Load LoRA weights if present
    if "vlm_lora_state_dict" in ckpt and ckpt["vlm_lora_state_dict"]:
        print(f"Loading VLM LoRA weights (rank={config.get('lora_rank', '?')})...")
        from peft import LoraConfig, get_peft_model
        vlm_model = policy.model.vlm_with_expert.get_vlm_model().text_model
        lora_config = LoraConfig(
            r=config.get("lora_rank", 8),
            lora_alpha=config.get("lora_rank", 8) * 2,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
            bias="none",
        )
        vlm_model_lora = get_peft_model(vlm_model, lora_config)
        policy.model.vlm_with_expert.vlm.model.text_model = vlm_model_lora

        # Load LoRA state dict
        missing, unexpected = policy.model.load_state_dict(
            ckpt["vlm_lora_state_dict"], strict=False
        )
        print(f"  LoRA loaded: {len(ckpt['vlm_lora_state_dict'])} keys")

    print(f"  Training loss was: {ckpt.get('loss', '?'):.6f} (epoch {ckpt.get('epoch', '?')})")

    # Load normalization stats if available
    inv_mean = np.array(ckpt.get("inv_mean"), dtype=np.float32) if "inv_mean" in ckpt else None
    inv_std = np.array(ckpt.get("inv_std"), dtype=np.float32) if "inv_std" in ckpt else None
    if inv_mean is not None:
        print(f"  Inv normalization: mean={inv_mean}, std={inv_std}")
    else:
        print(f"  No invariant normalization stats in checkpoint")

    # Load action normalization stats (dual-head mode)
    act_mean = np.array(ckpt.get("act_mean"), dtype=np.float32) if "act_mean" in ckpt else None
    act_std = np.array(ckpt.get("act_std"), dtype=np.float32) if "act_std" in ckpt else None
    if act_mean is not None:
        print(f"  Act normalization: mean={act_mean}, std={act_std}")

    has_action_head = config.get("action_dim") is not None
    if has_action_head:
        print(f"  Dual-head mode: action_dim={config['action_dim']} (direct action prediction)")

    return policy, dhb_head, config, inv_mean, inv_std, act_mean, act_std


def get_vlm_features(policy, obs_dict, task_name, device):
    """Extract VLM prefix features from a single observation."""
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
    from scipy.spatial.transform import Rotation

    agentview = obs_dict["agentview_image"].astype(np.uint8)
    eih = obs_dict.get("robot0_eye_in_hand_image", agentview).astype(np.uint8)

    img1 = torch.from_numpy(agentview).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img2 = torch.from_numpy(eih).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img1 = torch.flip(img1, dims=[2, 3]).to(device)
    img2 = torch.flip(img2, dims=[2, 3]).to(device)

    eef_pos = obs_dict.get("robot0_eef_pos", np.zeros(3))
    eef_quat = obs_dict.get("robot0_eef_quat", np.array([0, 0, 0, 1.0]))
    aa = Rotation.from_quat(eef_quat).as_rotvec().astype(np.float32)
    gripper = obs_dict.get("robot0_gripper_qpos", np.zeros(2))
    state = np.concatenate([eef_pos, aa, gripper]).astype(np.float32)

    batch = {
        "observation.images.image": img1,
        "observation.images.image2": img2,
        "observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
        "task": [task_name],
    }
    batch = policy._preprocessor(batch)

    images, img_masks = policy.prepare_images(batch)
    state_t = policy.prepare_state(batch)
    lang_tokens = batch["observation.language.tokens"]
    lang_masks = batch["observation.language.attention_mask"]

    prefix_embs, prefix_pad_masks, prefix_att_masks = \
        policy.model.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state_t)

    # Create dummy suffix (SmolVLA requires both prefix + suffix)
    expert_hidden_size = policy.model.vlm_with_expert.expert_hidden_size
    chunk_size = policy.model.config.chunk_size
    dummy_suffix = torch.zeros(1, chunk_size, expert_hidden_size,
                               device=device, dtype=prefix_embs.dtype)
    dummy_suffix_pad = torch.ones(1, chunk_size, dtype=torch.bool, device=device)
    dummy_suffix_att = torch.ones(1, chunk_size, dtype=torch.bool, device=device)

    pad_masks = torch.cat([prefix_pad_masks, dummy_suffix_pad], dim=1)
    att_masks = torch.cat([prefix_att_masks, dummy_suffix_att], dim=1)
    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1

    with torch.no_grad():
        (vlm_out, _), _ = policy.model.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, dummy_suffix],
            use_cache=False,
            fill_kv_cache=False,
        )

    return vlm_out.float()  # (1, S, 2048), cast bf16 → fp32


def invariants_to_actions(pred_inv, ee_pos, ee_quat_xyzw, osc_pos_scale=80.0):
    """
    Decode DHB invariants back to delta actions.
    pred_inv: (T, 8) numpy array [4 linear + 4 angular invariants]
    Returns: (T-1, 7) delta actions [dx, dy, dz, dax, day, daz, grip]
    """
    from dhb_xr.decoder.dhb_dr import decode_dhb_dr
    from dhb_xr.core.types import EncodingMethod, DHBMethod
    from scipy.spatial.transform import Rotation

    # Split linear and angular invariants
    lin_inv = pred_inv[:, :4]
    ang_inv = pred_inv[:, 4:]

    # Initial pose for decoding (convert xyzw -> wxyz)
    q_wxyz = np.array([ee_quat_xyzw[3], ee_quat_xyzw[0],
                       ee_quat_xyzw[1], ee_quat_xyzw[2]])
    init_pose = {
        "position": ee_pos.copy(),
        "quaternion": q_wxyz,
    }

    try:
        result = decode_dhb_dr(
            lin_inv, ang_inv,
            initial_pose=init_pose,
            method=EncodingMethod.POSITION,
            dhb_method=DHBMethod.DOUBLE_REFLECTION,
            drop_padded=False,
        )
        positions = result["positions"]   # (T, 3) absolute trajectory
        quaternions = result["quaternions"]  # (T, 4) wxyz

        # Position deltas -> OSC position commands
        pos_deltas = np.diff(positions, axis=0)  # (T-1, 3)

        # Rotation deltas -> axis-angle OSC rotation commands
        ang_deltas = np.zeros((len(pos_deltas), 3))
        for i in range(len(pos_deltas)):
            q1 = quaternions[i]    # wxyz
            q2 = quaternions[i+1]  # wxyz
            r1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])  # xyzw
            r2 = Rotation.from_quat([q2[1], q2[2], q2[3], q2[0]])
            ang_deltas[i] = (r2 * r1.inv()).as_rotvec()

        # Build 7-DoF actions: [pos(3), rot(3), gripper(1)]
        actions = np.zeros((len(pos_deltas), 7))
        actions[:, :3] = pos_deltas * osc_pos_scale
        actions[:, 3:6] = ang_deltas * 10.0  # rotation scale (smaller)
        # Gripper: default to closed (-1) for grasping tasks
        actions[:, 6] = -1.0
        return actions
    except Exception as e:
        print(f"  decode_dhb failed: {e}")
        import traceback; traceback.print_exc()
        return np.zeros((pred_inv.shape[0], 7))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="_workspace/dhb_head_e2e.pt")
    parser.add_argument("--offsets", nargs="+", type=float, default=[0.0, 3.0, 5.0])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=120)
    parser.add_argument("--gain", type=float, default=2.0)
    parser.add_argument("--replan_every", type=int, default=10,
                        help="Re-predict invariants every N steps (closed-loop)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    policy, dhb_head, config, inv_mean, inv_std, act_mean, act_std = load_model(args.checkpoint, device)
    T = config["action_horizon"]
    has_action_head = config.get("action_dim") is not None

    # Setup LIBERO environment
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import h5py

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    task_name = task.name.replace("_", " ") + "\n"
    bddl = os.path.join(get_libero_path("bddl_files"),
                        task.problem_folder, task.bddl_file)
    dpath = os.path.join(get_libero_path("datasets"), "libero_spatial",
                         f"{task.name}_demo.hdf5")

    env = OffScreenRenderEnv(**{
        "bddl_file_name": bddl,
        "camera_heights": 256, "camera_widths": 256,
        "camera_names": ["agentview", "robot0_eye_in_hand"],
    })
    env.seed(0)

    # Load init states
    with h5py.File(dpath) as f:
        init_states = [np.array(f["data"][k]["states"][0])
                       for k in sorted(f["data"].keys())]

    # Run evaluation
    results = {"trials": [], "metadata": {
        "checkpoint": args.checkpoint,
        "offsets_cm": args.offsets,
        "seeds_per_offset": args.seeds,
        "action_horizon": T,
    }}

    for offset_cm in args.offsets:
        offset_m = offset_cm / 100.0
        successes = 0

        for seed in range(args.seeds):
            env.reset()
            init_state = init_states[seed % len(init_states)].copy()

            # Apply offset to bowl Y position (qpos[10])
            if offset_m != 0:
                init_state[11] += offset_m

            obs = env.set_init_state(init_state)

            # Use default controller gains (do NOT override — gain=2.0 breaks OSC)
            # The original eval scripts used gain overrides for JOINT_POSITION
            # controller, but DHBActionHead outputs OSC_POSE delta actions.

            # Settle
            for _ in range(5):
                obs, _, _, _ = env.step([0.0] * 7)

            # Closed-loop: re-predict every replan_every steps
            replan_every = args.replan_every
            reward = 0.0
            inv_norm = 0.0
            total_steps = 0

            for _ in range(0, args.max_steps, replan_every):
                # Predict from current observation
                vlm_feat = get_vlm_features(policy, obs, task_name, device)
                with torch.no_grad():
                    result = dhb_head.predict(vlm_feat)

                if has_action_head and "pred_actions" in result:
                    # ── Dual-head mode: direct action prediction ──
                    pred_actions = result["pred_actions"][0].cpu().numpy()

                    # Denormalize actions
                    if act_mean is not None and act_std is not None:
                        pred_actions = pred_actions * act_std + act_mean

                    inv_norm = float(np.linalg.norm(
                        result["pred_invariants"][0].cpu().numpy()))

                    # Execute first replan_every actions directly
                    for step_i in range(min(replan_every, len(pred_actions))):
                        act = pred_actions[step_i]
                        obs, r, done, info = env.step(act)
                        if r > reward:
                            reward = r
                        total_steps += 1
                        if done or total_steps >= args.max_steps:
                            break
                else:
                    # ── Legacy mode: invariant decode to actions ──
                    pred_inv = result["pred_invariants"][0].cpu().numpy()

                    # Denormalize if training used per-channel normalization
                    if inv_mean is not None and inv_std is not None:
                        pred_inv = pred_inv * inv_std + inv_mean

                    # Clamp magnitudes to non-negative
                    pred_inv[:, 0] = np.abs(pred_inv[:, 0])
                    pred_inv[:, 4] = np.abs(pred_inv[:, 4])

                    inv_norm = max(inv_norm, float(np.linalg.norm(pred_inv)))

                    # Decode invariants to actions from current pose
                    ee_pos = obs["robot0_eef_pos"]
                    ee_quat = obs["robot0_eef_quat"]
                    actions = invariants_to_actions(pred_inv, ee_pos, ee_quat)

                    if actions is None or len(actions) == 0:
                        break

                    # Execute first replan_every actions
                    for step_i in range(min(replan_every, len(actions))):
                        act = actions[step_i]
                        obs, r, done, info = env.step(act)
                        if r > reward:
                            reward = r
                        total_steps += 1
                        if done or total_steps >= args.max_steps:
                            break

                if done or total_steps >= args.max_steps:
                    break

            success = reward > 0.5
            if success:
                successes += 1

            results["trials"].append({
                "offset_cm": offset_cm,
                "seed": seed,
                "success": success,
                "reward": float(reward),
                "inv_norm": inv_norm,
            })

        rate = successes / args.seeds * 100
        print(f"  offset={offset_cm}cm: {successes}/{args.seeds} = {rate:.0f}%")

    env.close()

    # Save results
    out_path = args.checkpoint.replace(".pt", "_eval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
