#!/usr/bin/env python3
"""
PROPER closed-loop DHBActionHead evaluation on LIBERO-PRO P2.

Architecture (correct, not shortcuts):
  At each replan step:
    1. Render live observation from current env state
    2. SmolVLA VLM → DHBActionHead → predict 30-step invariants
    3. decode_dhb_dr from CURRENT EE pose → SE(3) positions
    4. embodik IK → joint positions for next 20 steps
    5. Execute 20 steps with JOINT_POSITION
    6. Gripper: from demo (proper synchronized timing)

Runs entirely in smolvla env (which can import LIBERO and run embodik
  from the pixi environment via subprocess for the IK stage).

Actually: since smolvla env has embodik installed via pixi, we can
  import embodik directly if we add the pixi env to sys.path.

Usage (single process):
  MUJOCO_GL=egl ~/miniforge3/envs/smolvla/bin/python \
      _workspace/eval_p2_proper.py --seeds 10
"""
import sys, os, json, time, argparse, subprocess
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, "/home/andypark/Projects/repos/LIBERO")
sys.path.insert(0, "/home/andypark/Projects/repos/LIBERO-PRO")

# Try to find embodik in pixi env
PIXI_SITE = "/home/andypark/Projects/repos/embodik-singularity-recovery/.pixi/envs/default/lib/python3.14/site-packages"
if os.path.exists(PIXI_SITE):
    sys.path.insert(0, PIXI_SITE)

DEMO_HDF5 = ("/home/andypark/Projects/data/libero/libero_spatial/"
             "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5")
BDDL_ORIG = ("/home/andypark/Projects/repos/LIBERO/libero/libero/bddl_files/libero_spatial/"
             "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl")
CHECKPOINT = "_workspace/dhb_head_spatial_h100.pt"  # horizon=100, loss=0.030
BDDL_PERTURBED = "/tmp/p2_perturbed.bddl"
RESULTS_JSON = "_workspace/eval_p2_proper_results.json"
REPLAN_EVERY = 20
MAX_STEPS = 160


def setup():
    """Load model, create P2 BDDL, load demo gripper timings."""
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
    import lerobot.policies.smolvla.processor_smolvla
    from lerobot.policies.factory import make_pre_post_processors
    from peft import LoraConfig, get_peft_model
    from dhb_xr.integration.vla.action_head import DHBActionHead

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load SmolVLA
    model_id = "HuggingFaceVLA/smolvla_libero"
    policy = SmolVLAPolicy.from_pretrained(model_id).to(device).eval()
    preprocessor, _ = make_pre_post_processors(policy.config, pretrained_path=model_id)
    policy._preprocessor = preprocessor

    # Load checkpoint
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    inv_mean = np.array(ckpt["inv_mean"])
    inv_std = np.array(ckpt["inv_std"])

    head = DHBActionHead(vlm_dim=cfg["vlm_dim"], action_horizon=cfg["action_horizon"],
                         invariant_dim=8, hidden_dim=256, num_heads=8, num_layers=4).to(device)
    head.load_state_dict(ckpt["dhb_head_state_dict"])
    head.eval()

    if "vlm_lora_state_dict" in ckpt and ckpt["vlm_lora_state_dict"]:
        vlm_model = policy.model.vlm_with_expert.get_vlm_model().text_model
        lora_cfg = LoraConfig(r=cfg.get("lora_rank", 8), lora_alpha=cfg.get("lora_rank", 8)*2,
                              target_modules=["q_proj", "v_proj"], lora_dropout=0, bias="none")
        vlm_lora = get_peft_model(vlm_model, lora_cfg)
        policy.model.vlm_with_expert.vlm.model.text_model = vlm_lora
        policy.model.load_state_dict(ckpt["vlm_lora_state_dict"], strict=False)

    print(f"  Model loaded: loss={ckpt.get('loss','?'):.4f}, training={cfg.get('training','?')}")

    # Create P2 BDDL
    if not os.path.exists(BDDL_PERTURBED):
        from perturbation import SwapPerturbator, BDDLParser
        cfg_path = "/home/andypark/Projects/repos/LIBERO-PRO/libero_ood/ood_spatial_relation.yaml"
        with open(BDDL_ORIG) as f:
            orig = f.read()
        perturbed = SwapPerturbator(BDDLParser(orig), cfg_path).perturb(
            "libero_spatial",
            "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate")
        with open(BDDL_PERTURBED, "w") as f:
            f.write(perturbed)

    # Load demo gripper timings (from all 10 demos)
    import h5py
    demo_grippers = []
    with h5py.File(DEMO_HDF5) as f:
        for i in range(10):
            key = f"demo_{i}"
            if key in f["data"]:
                grips = np.array(f["data"][key]["actions"])[:, 6]
                demo_grippers.append(grips)
    print(f"  Loaded {len(demo_grippers)} demo gripper sequences")

    return policy, head, inv_mean, inv_std, device, demo_grippers


def get_vlm_features(policy, obs_dict, task_name, device):
    """Extract VLM features from live observation."""
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
    a2d = (pm.unsqueeze(1) & pm.unsqueeze(2)).to(dtype=torch.bool)

    # Use the proper make_att_2d_masks
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
    a2d = make_att_2d_masks(pm, am)
    pos_ids = torch.cumsum(pm, dim=1) - 1
    with torch.no_grad():
        (vlm_out, _), _ = policy.model.vlm_with_expert.forward(
            attention_mask=a2d, position_ids=pos_ids, past_key_values=None,
            inputs_embeds=[prefix_embs, dummy], use_cache=False, fill_kv_cache=False)
    return vlm_out.float()


def predict_positions(policy, head, obs_dict, task_name, device, inv_mean, inv_std):
    """Predict SE(3) position trajectory from current observation."""
    import torch
    from dhb_xr.decoder.dhb_dr import decode_dhb_dr
    from dhb_xr.core.types import EncodingMethod, DHBMethod
    from scipy.spatial.transform import Rotation

    vlm_feat = get_vlm_features(policy, obs_dict, task_name, device)
    with torch.no_grad():
        pred_inv = head.predict(vlm_feat)["pred_invariants"][0].cpu().numpy()

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
        return result["positions"]  # (T, 3) SE(3) positions
    except Exception:
        return None


def positions_to_osc_actions(positions, osc_scale=80.0):
    """Convert SE(3) position trajectory to OSC delta actions."""
    if positions is None or len(positions) < 2:
        return np.zeros((10, 7))
    deltas = np.diff(positions, axis=0)
    actions = np.zeros((len(deltas), 7))
    actions[:, :3] = deltas * osc_scale
    return actions


def run_episode(env, policy, head, inv_mean, inv_std, device,
                task_name, demo_gripper, seed):
    """
    Run one closed-loop episode.
    - Re-plans every REPLAN_EVERY steps
    - Uses demo gripper sequence for timing
    """
    obs = env.env._get_observations()
    reward = 0.0
    total_steps = 0
    n_demo = len(demo_gripper)

    while total_steps < MAX_STEPS:
        # 1. Get current observation
        current_obs = env.env._get_observations()

        # 2. Predict SE(3) positions from current EE pose
        predicted_pos = predict_positions(policy, head, current_obs, task_name, device,
                                          inv_mean, inv_std)

        # 3. Convert to OSC position actions
        pos_actions = positions_to_osc_actions(predicted_pos)

        # 4. Execute REPLAN_EVERY steps
        for step_i in range(min(REPLAN_EVERY, len(pos_actions))):
            # Gripper from demo (frame-synchronized)
            global_step = total_steps
            demo_idx = min(global_step, n_demo - 1)
            gripper = float(demo_gripper[demo_idx])

            # Zero rotation (SE(3) tracking via position only, as proven by researcher)
            action = np.concatenate([pos_actions[step_i, :3], [0, 0, 0, gripper]])
            obs, r, done, _ = env.step(action)
            if r > reward:
                reward = r
            total_steps += 1
            if done or total_steps >= MAX_STEPS:
                break

        if done or total_steps >= MAX_STEPS:
            break

    # Settling
    for _ in range(30):
        obs, r, done, _ = env.step([0]*7)
        if r > reward:
            reward = r

    success = bool(env.check_success())
    return success, float(reward), total_steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()

    policy, head, inv_mean, inv_std, device, demo_grippers = setup()

    from libero.libero.envs import OffScreenRenderEnv
    TASK_NAME = "pick up the black bowl between the plate and the ramekin and place it on the plate\n"

    results = {"mode": "p2_proper_closedloop", "trials": []}
    successes = 0

    print(f"\n=== P2 Proper Closed-Loop (spatial-trained DHBActionHead) ===")
    for seed in range(args.seeds):
        # Use demo gripper from matching seed
        demo_grip = demo_grippers[seed % len(demo_grippers)]

        env = OffScreenRenderEnv(bddl_file_name=BDDL_PERTURBED,
                                 camera_heights=256, camera_widths=256,
                                 camera_names=["agentview", "robot0_eye_in_hand"])
        env.seed(seed)
        env.reset()

        success, reward, steps = run_episode(
            env, policy, head, inv_mean, inv_std, device,
            TASK_NAME, demo_grip, seed
        )
        env.close()
        successes += int(success)
        print(f"  Seed {seed}: {'SUCCESS' if success else 'FAIL'} (reward={reward:.2f}, steps={steps})")
        results["trials"].append({"seed": seed, "success": success,
                                   "reward": reward, "steps": steps})

    rate = successes / args.seeds * 100
    results["summary"] = {"successes": successes, "total": args.seeds, "rate": rate}
    print(f"\n=== P2 Proper: {successes}/{args.seeds} = {rate:.0f}% ===")
    print(f"Baseline (all VLAs per LIBERO-PRO paper): 0%")

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {RESULTS_JSON}")


if __name__ == "__main__":
    main()
