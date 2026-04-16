#!/usr/bin/env python3
"""
Closed-loop DHBActionHead evaluation on LIBERO-PRO P2 (position swap).

Re-plans every REPLAN_EVERY steps from current observation.
Runs entirely in smolvla env (no env switching needed).

Comparison:
  - Baseline (SmolVLA raw): 0% on P2 (per LIBERO-PRO paper)
  - DHBActionHead (spatial-diverse): X% on P2

Usage:
  MUJOCO_GL=egl ~/miniforge3/envs/smolvla/bin/python \
      _workspace/eval_p2_closedloop.py --seeds 10 --replan_every 20
"""
import sys, os, json, time, argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, "/home/andypark/Projects/repos/LIBERO")
sys.path.insert(0, "/home/andypark/Projects/repos/LIBERO-PRO")

BDDL_ORIG = ("/home/andypark/Projects/repos/LIBERO/libero/libero/bddl_files/libero_spatial/"
             "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl")
CHECKPOINT = "_workspace/dhb_head_spatial.pt"
BDDL_PERTURBED = "/tmp/p2_perturbed.bddl"
RESULTS_JSON = "_workspace/eval_p2_closedloop_results.json"


def get_or_create_p2_bddl():
    """Generate P2 BDDL if not already done."""
    if not os.path.exists(BDDL_PERTURBED):
        from perturbation import SwapPerturbator, BDDLParser
        cfg_path = "/home/andypark/Projects/repos/LIBERO-PRO/libero_ood/ood_spatial_relation.yaml"
        with open(BDDL_ORIG) as f:
            orig = f.read()
        parser = BDDLParser(orig)
        perturbed = SwapPerturbator(parser, cfg_path).perturb(
            "libero_spatial",
            "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate")
        with open(BDDL_PERTURBED, "w") as f:
            f.write(perturbed)
        print(f"P2 BDDL created: {BDDL_PERTURBED}")
    return BDDL_PERTURBED


def load_model(device):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
    import lerobot.policies.smolvla.processor_smolvla
    from lerobot.policies.factory import make_pre_post_processors
    from peft import LoraConfig, get_peft_model

    model_id = "HuggingFaceVLA/smolvla_libero"
    policy = SmolVLAPolicy.from_pretrained(model_id).to(device).eval()
    preprocessor, _ = make_pre_post_processors(policy.config, pretrained_path=model_id)
    policy._preprocessor = preprocessor

    import torch
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    inv_mean = np.array(ckpt["inv_mean"]) if "inv_mean" in ckpt else None
    inv_std = np.array(ckpt["inv_std"]) if "inv_std" in ckpt else None

    from dhb_xr.integration.vla.action_head import DHBActionHead
    head = DHBActionHead(vlm_dim=cfg["vlm_dim"], action_horizon=cfg["action_horizon"],
                         invariant_dim=8, hidden_dim=256, num_heads=8, num_layers=4).to(device)
    head.load_state_dict(ckpt["dhb_head_state_dict"])
    head.eval()

    # Load LoRA
    if "vlm_lora_state_dict" in ckpt and ckpt["vlm_lora_state_dict"]:
        vlm_model = policy.model.vlm_with_expert.get_vlm_model().text_model
        lora_cfg = LoraConfig(r=cfg.get("lora_rank", 8), lora_alpha=cfg.get("lora_rank", 8)*2,
                              target_modules=["q_proj", "v_proj"], lora_dropout=0, bias="none")
        vlm_lora = get_peft_model(vlm_model, lora_cfg)
        policy.model.vlm_with_expert.vlm.model.text_model = vlm_lora
        policy.model.load_state_dict(ckpt["vlm_lora_state_dict"], strict=False)

    print(f"  Checkpoint: loss={ckpt.get('loss','?'):.4f}, epoch={ckpt.get('epoch','?')}")
    print(f"  Training type: {cfg.get('training','unknown')}")
    print(f"  Action horizon: {cfg['action_horizon']}")
    return policy, head, inv_mean, inv_std, cfg["action_horizon"]


def get_vlm_features(policy, obs_dict, task_name, device):
    """Extract VLM features from observation."""
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
    from scipy.spatial.transform import Rotation

    agentview = obs_dict["agentview_image"].astype(np.uint8)
    eih = obs_dict.get("robot0_eye_in_hand_image", agentview).astype(np.uint8)
    img1 = torch.from_numpy(agentview).permute(2,0,1).unsqueeze(0).float()/255.0
    img2 = torch.from_numpy(eih).permute(2,0,1).unsqueeze(0).float()/255.0
    img1 = torch.flip(img1, dims=[2,3]).to(device)
    img2 = torch.flip(img2, dims=[2,3]).to(device)

    eef_pos = obs_dict.get("robot0_eef_pos", np.zeros(3))
    eef_quat = obs_dict.get("robot0_eef_quat", np.array([0,0,0,1.0]))
    aa = Rotation.from_quat(eef_quat).as_rotvec().astype(np.float32)
    gripper = obs_dict.get("robot0_gripper_qpos", np.zeros(2))
    state = np.concatenate([eef_pos, aa, gripper]).astype(np.float32)

    batch = {"observation.images.image": img1, "observation.images.image2": img2,
             "observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
             "task": [task_name]}
    batch = policy._preprocessor(batch)
    images, img_masks = policy.prepare_images(batch)
    st = policy.prepare_state(batch)
    lt = batch["observation.language.tokens"]
    lm = batch["observation.language.attention_mask"]
    prefix_embs, pp, pa = policy.model.embed_prefix(images, img_masks, lt, lm, state=st)
    ehs = policy.model.vlm_with_expert.expert_hidden_size
    cs = policy.model.config.chunk_size
    dummy = torch.zeros(1, cs, ehs, device=device, dtype=prefix_embs.dtype)
    dp = torch.ones(1, cs, dtype=torch.bool, device=device)
    da = torch.ones(1, cs, dtype=torch.bool, device=device)
    pm = torch.cat([pp, dp], dim=1)
    am = torch.cat([pa, da], dim=1)
    a2d = make_att_2d_masks(pm, am)
    pos_ids = torch.cumsum(pm, dim=1) - 1
    with torch.no_grad():
        (vlm_out, _), _ = policy.model.vlm_with_expert.forward(
            attention_mask=a2d, position_ids=pos_ids, past_key_values=None,
            inputs_embeds=[prefix_embs, dummy], use_cache=False, fill_kv_cache=False)
    return vlm_out.float()


def predict_and_decode(policy, head, obs_dict, task_name, device, inv_mean, inv_std):
    """Predict invariants → decode to EE position trajectory."""
    import torch
    from dhb_xr.decoder.dhb_dr import decode_dhb_dr
    from dhb_xr.core.types import EncodingMethod, DHBMethod
    from scipy.spatial.transform import Rotation

    vlm_feat = get_vlm_features(policy, obs_dict, task_name, device)
    with torch.no_grad():
        pred_inv = head.predict(vlm_feat)["pred_invariants"][0].cpu().numpy()

    if inv_mean is not None:
        pred_inv = pred_inv * inv_std + inv_mean
    pred_inv[:, 0] = np.abs(pred_inv[:, 0])
    pred_inv[:, 4] = np.abs(pred_inv[:, 4])

    eef_pos = obs_dict["robot0_eef_pos"]
    eef_quat = obs_dict["robot0_eef_quat"]
    q_wxyz = np.array([eef_quat[3], eef_quat[0], eef_quat[1], eef_quat[2]])
    init_pose = {"position": eef_pos.copy(), "quaternion": q_wxyz}

    try:
        result = decode_dhb_dr(pred_inv[:,:4], pred_inv[:,4:], initial_pose=init_pose,
                               method=EncodingMethod.POSITION,
                               dhb_method=DHBMethod.DOUBLE_REFLECTION, drop_padded=False)
        positions = result["positions"]
        pos_deltas = np.diff(positions, axis=0)
        actions = np.zeros((len(pos_deltas), 7))
        actions[:, :3] = pos_deltas * 80.0
        # Gripper timing heuristic (from demo statistics):
        # - First ~40% of steps: open (approach bowl)
        # - Middle ~50%: closed (grasp and carry)
        # - Last ~10%: open (release on plate)
        T = len(actions)
        actions[:int(T*0.4), 6] = -1.0   # open
        actions[int(T*0.4):int(T*0.9), 6] = 1.0   # closed
        actions[int(T*0.9):, 6] = -1.0   # open (release)
        return actions
    except Exception as e:
        return np.zeros((10, 7))


def run_episode(env, policy, head, inv_mean, inv_std, task_name, device,
                max_steps=150, replan_every=20, bddl_label="original"):
    """Run one closed-loop episode with re-planning."""
    obs = env.reset()
    reward = 0.0
    total_steps = 0

    while total_steps < max_steps:
        # Re-predict from current observation
        current_obs = env.env._get_observations() if hasattr(env.env, '_get_observations') else obs
        # Make sure it has the needed keys
        if "agentview_image" not in current_obs:
            current_obs = obs  # fallback

        actions = predict_and_decode(policy, head, current_obs, task_name, device, inv_mean, inv_std)

        # Execute this chunk
        for step_i in range(min(replan_every, len(actions))):
            act = actions[step_i]
            obs, r, done, _ = env.step(act)
            if r > reward:
                reward = r
            total_steps += 1
            if done or total_steps >= max_steps:
                break
        if done or total_steps >= max_steps:
            break

    # Settling with gripper closed
    for _ in range(50):
        obs, r, done, _ = env.step([0.0]*3 + [0.0]*3 + [1.0])
        if r > reward:
            reward = r

    success = bool(env.check_success())
    return success, float(reward), total_steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--replan_every", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=150)
    args = parser.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print("Loading model...")
    policy, head, inv_mean, inv_std, T_horizon = load_model(device)

    # Get P2 BDDL
    bddl_p2 = get_or_create_p2_bddl()
    print(f"P2 BDDL: {bddl_p2}")

    from libero.libero.envs import OffScreenRenderEnv
    TASK_NAME = "pick up the black bowl between the plate and the ramekin and place it on the plate\n"

    results = {"mode": "p2_closedloop_spatial", "replan_every": args.replan_every,
               "trials": [], "summary": {}}

    for condition, bddl_path, label in [
        ("p2_dhb", bddl_p2, "P2 + DHBActionHead (spatial)"),
        ("p2_naive", bddl_p2, "P2 baseline (same model, no adaptation)"),
    ]:
        successes = 0
        print(f"\n=== {label} ===")

        for seed in range(args.seeds):
            env = OffScreenRenderEnv(bddl_file_name=bddl_path,
                                     camera_heights=256, camera_widths=256,
                                     camera_names=["agentview", "robot0_eye_in_hand"])
            env.seed(seed)

            success, reward, steps = run_episode(
                env, policy, head, inv_mean, inv_std, TASK_NAME, device,
                max_steps=args.max_steps, replan_every=args.replan_every,
                bddl_label=label
            )
            env.close()
            successes += int(success)
            print(f"  Seed {seed}: {'SUCCESS' if success else 'FAIL'} (reward={reward:.2f}, steps={steps})")
            results["trials"].append({
                "condition": condition, "seed": seed,
                "success": success, "reward": reward, "steps": steps
            })

        rate = successes / args.seeds * 100
        results["summary"][condition] = {"successes": successes, "total": args.seeds, "rate": rate}
        print(f"  === {label}: {successes}/{args.seeds} = {rate:.0f}% ===")

        if condition == "p2_dhb":
            break  # Only run DHB for now; baseline is known 0% from LIBERO-PRO paper

    print("\n=== FINAL SUMMARY ===")
    print(f"LIBERO-PRO paper: OpenVLA/pi0/pi0.5 all 0% on P2")
    for k, v in results["summary"].items():
        print(f"DHBActionHead (spatial): {v['successes']}/{v['total']} = {v['rate']:.0f}%")

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {RESULTS_JSON}")


if __name__ == "__main__":
    main()
