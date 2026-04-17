#!/usr/bin/env python3
"""
Diagnostic script: compare original LIBERO actions vs DHB-decoded actions
vs model-predicted actions.

Answers:
  1. What do SmolVLA's original actions look like? (magnitude, range)
  2. Does encode->decode roundtrip preserve trajectory?
  3. What do model-predicted invariants decode to? (magnitude, range, direction)
  4. Is the OSC scale factor (80x) correct?

Usage:
    MUJOCO_GL=egl ~/miniforge3/envs/smolvla/bin/python _workspace/debug_eval_actions.py
"""
import sys, os
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, "/home/andypark/Projects/repos/LIBERO")


def print_action_stats(actions, label="Actions"):
    """Print per-column statistics for (T, 7) actions."""
    cols = ["dx", "dy", "dz", "dax", "day", "daz", "grip"]
    print(f"\n{'='*70}")
    print(f"  {label}  shape={actions.shape}")
    print(f"{'='*70}")
    print(f"  {'col':>5s}  {'mean':>10s}  {'std':>10s}  {'min':>10s}  {'max':>10s}  {'absmax':>10s}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for j in range(min(actions.shape[1], len(cols))):
        c = actions[:, j]
        print(f"  {cols[j]:>5s}  {c.mean():10.6f}  {c.std():10.6f}  "
              f"{c.min():10.6f}  {c.max():10.6f}  {np.abs(c).max():10.6f}")
    pos_mag = np.linalg.norm(actions[:, :3], axis=1)
    rot_mag = np.linalg.norm(actions[:, 3:6], axis=1)
    print(f"\n  pos_delta_norm: mean={pos_mag.mean():.6f}  max={pos_mag.max():.6f}")
    print(f"  rot_delta_norm: mean={rot_mag.mean():.6f}  max={rot_mag.max():.6f}")


def print_invariant_stats(inv, label="Invariants"):
    """Print per-column statistics for (T, 8) invariants."""
    cols = ["lin_m", "lin_e1", "lin_e2", "lin_e3", "ang_m", "ang_e1", "ang_e2", "ang_e3"]
    print(f"\n{'='*70}")
    print(f"  {label}  shape={inv.shape}")
    print(f"{'='*70}")
    print(f"  {'col':>7s}  {'mean':>10s}  {'std':>10s}  {'min':>10s}  {'max':>10s}  {'absmax':>10s}")
    print(f"  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for j in range(inv.shape[1]):
        c = inv[:, j]
        print(f"  {cols[j]:>7s}  {c.mean():10.6f}  {c.std():10.6f}  "
              f"{c.min():10.6f}  {c.max():10.6f}  {np.abs(c).max():10.6f}")


def main():
    import torch
    from dhb_xr.integration.vla.smolvla_adapter import SmolVLADHBAdapter
    import h5py
    from scipy.spatial.transform import Rotation
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from dhb_xr.encoder.dhb_dr import encode_dhb_dr
    from dhb_xr.decoder.dhb_dr import decode_dhb_dr
    from dhb_xr.core.types import EncodingMethod, DHBMethod

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 1. Load original LIBERO demo actions and EE trajectory ──────────
    print("\n" + "#"*70)
    print("# PART 1: Original LIBERO demo_0 actions & trajectory")
    print("#"*70)

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    task_name = task.name.replace("_", " ") + "\n"
    dpath = os.path.join(get_libero_path("datasets"), "libero_spatial",
                         f"{task.name}_demo.hdf5")
    bddl = os.path.join(get_libero_path("bddl_files"),
                        task.problem_folder, task.bddl_file)

    with h5py.File(dpath) as f:
        d = f["data"]["demo_0"]
        orig_actions = np.array(d["actions"])          # (T, 7)
        ee_pos = np.array(d["obs"]["ee_pos"])          # (T+1, 3)
        init_state = np.array(d["states"][0])

        # Check what obs keys exist
        print(f"\n  Dataset obs keys: {list(d['obs'].keys())}")
        print(f"  actions shape: {orig_actions.shape}")
        print(f"  ee_pos shape:  {ee_pos.shape}")
        print(f"  ee_pos[0]:     {ee_pos[0]}")
        print(f"  ee_pos[-1]:    {ee_pos[-1]}")
        print(f"  ee_pos range:  [{ee_pos.min(0)}, {ee_pos.max(0)}]")

        # Get orientations
        if "ee_states" in d["obs"]:
            euler = np.array(d["obs"]["ee_states"])[:, 3:]
            rots = Rotation.from_euler("xyz", euler)
            q_xyzw = rots.as_quat()
            quats_wxyz = np.column_stack([q_xyzw[:, 3], q_xyzw[:, 0],
                                          q_xyzw[:, 1], q_xyzw[:, 2]])
            print(f"  ee_states shape: {np.array(d['obs']['ee_states']).shape}")
        else:
            quats_wxyz = np.tile([1, 0, 0, 0], (len(ee_pos), 1)).astype(float)
            print("  WARNING: No ee_states found, using identity quaternions")

    print_action_stats(orig_actions, "Original LIBERO demo_0 actions")

    # Also compute position deltas from the EE trajectory directly
    pos_deltas_from_traj = np.diff(ee_pos, axis=0)
    print(f"\n  Position deltas from ee_pos trajectory:")
    print(f"    mean norm: {np.linalg.norm(pos_deltas_from_traj, axis=1).mean():.6f}")
    print(f"    max  norm: {np.linalg.norm(pos_deltas_from_traj, axis=1).max():.6f}")
    print(f"    These are RAW deltas in meters.")

    # Compare: original action[:3] vs pos_deltas
    print(f"\n  Ratio check: action[:3] vs pos_deltas_from_traj (first 10 steps):")
    for i in range(min(10, len(orig_actions), len(pos_deltas_from_traj))):
        act_norm = np.linalg.norm(orig_actions[i, :3])
        pd_norm = np.linalg.norm(pos_deltas_from_traj[i])
        ratio = act_norm / (pd_norm + 1e-10)
        print(f"    step {i:2d}: action_norm={act_norm:.6f}  pos_delta_norm={pd_norm:.6f}  ratio={ratio:.1f}")

    # ── 2. Encode -> Decode roundtrip ───────────────────────────────────
    print("\n" + "#"*70)
    print("# PART 2: Encode -> Decode roundtrip test")
    print("#"*70)

    enc = encode_dhb_dr(ee_pos, quats_wxyz, method="pos",
                        use_default_initial_frames=True)
    gt_inv = np.concatenate([enc["linear_motion_invariants"],
                              enc["angular_motion_invariants"]], axis=1)
    print(f"\n  Encoded invariants shape: {gt_inv.shape}")
    print_invariant_stats(gt_inv, "Ground-truth DHB invariants (demo_0)")

    # Decode back using default initial frames (identity)
    init_pose_default = {
        "position": ee_pos[0].copy(),
        "quaternion": quats_wxyz[0].copy(),
    }
    dec = decode_dhb_dr(
        enc["linear_motion_invariants"],
        enc["angular_motion_invariants"],
        initial_pose=init_pose_default,
        method=EncodingMethod.POSITION,
        dhb_method=DHBMethod.DOUBLE_REFLECTION,
        drop_padded=True,
    )
    dec_pos = dec["positions"]
    dec_quat = dec["quaternions"]

    # Compare original vs decoded positions
    min_len = min(len(ee_pos), len(dec_pos))
    pos_error = np.linalg.norm(ee_pos[:min_len] - dec_pos[:min_len], axis=1)
    print(f"\n  Roundtrip position error (mm):")
    print(f"    mean: {pos_error.mean()*1000:.3f}")
    print(f"    max:  {pos_error.max()*1000:.3f}")
    print(f"    Decoded trajectory length: {len(dec_pos)}")
    print(f"    Original trajectory length: {len(ee_pos)}")

    # Compute actions from decoded trajectory
    dec_pos_deltas = np.diff(dec_pos[:min_len], axis=0)
    orig_pos_deltas = np.diff(ee_pos[:min_len], axis=0)
    delta_error = np.linalg.norm(dec_pos_deltas - orig_pos_deltas[:len(dec_pos_deltas)], axis=1)
    print(f"\n  Roundtrip position-delta error (mm):")
    print(f"    mean: {delta_error.mean()*1000:.3f}")
    print(f"    max:  {delta_error.max()*1000:.3f}")

    # ── 3. What OSC scale gives the right action magnitude? ─────────────
    print("\n" + "#"*70)
    print("# PART 3: OSC scale factor analysis")
    print("#"*70)

    # The eval script uses: actions[:, :3] = pos_deltas * osc_pos_scale
    # So: osc_pos_scale = action_magnitude / pos_delta_magnitude
    # Let's check what the actual ratio is between LIBERO actions and EE deltas

    orig_act_pos_norms = np.linalg.norm(orig_actions[:, :3], axis=1)
    orig_traj_delta_norms = np.linalg.norm(pos_deltas_from_traj, axis=1)

    # Only compute ratio for non-tiny deltas (actions may be 1 longer than deltas)
    n_common = min(len(orig_act_pos_norms), len(orig_traj_delta_norms))
    act_norms_c = orig_act_pos_norms[:n_common]
    delta_norms_c = orig_traj_delta_norms[:n_common]
    mask = delta_norms_c > 1e-5
    if mask.sum() > 0:
        ratios = act_norms_c[mask] / delta_norms_c[mask]
        print(f"\n  Actual OSC scale (action_norm / pos_delta_norm) for non-tiny steps:")
        print(f"    median: {np.median(ratios):.1f}")
        print(f"    mean:   {np.mean(ratios):.1f}")
        print(f"    std:    {np.std(ratios):.1f}")
        print(f"    min:    {np.min(ratios):.1f}")
        print(f"    max:    {np.max(ratios):.1f}")
        print(f"    (eval script uses: 80.0)")

    # Also check rotation scale
    orig_act_rot_norms = np.linalg.norm(orig_actions[:, 3:6], axis=1)
    # Compute rotation deltas from quaternions
    rot_deltas = np.zeros((len(quats_wxyz)-1, 3))
    for i in range(len(quats_wxyz)-1):
        q1 = quats_wxyz[i]
        q2 = quats_wxyz[i+1]
        r1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])
        r2 = Rotation.from_quat([q2[1], q2[2], q2[3], q2[0]])
        rot_deltas[i] = (r2 * r1.inv()).as_rotvec()

    rot_delta_norms = np.linalg.norm(rot_deltas, axis=1)
    n_rot = min(len(orig_act_rot_norms), len(rot_delta_norms))
    rot_mask = rot_delta_norms[:n_rot] > 1e-5
    if rot_mask.sum() > 0:
        rot_ratios = orig_act_rot_norms[:n_rot][rot_mask] / rot_delta_norms[:n_rot][rot_mask]
        print(f"\n  Actual rotation scale (action_rot_norm / rot_delta_norm) for non-tiny steps:")
        print(f"    median: {np.median(rot_ratios):.1f}")
        print(f"    mean:   {np.mean(rot_ratios):.1f}")
        print(f"    (eval script uses: 10.0)")

    # ── 4. What the eval script's decode produces for GT invariants ──────
    print("\n" + "#"*70)
    print("# PART 4: Eval script decode of GT invariants -> actions")
    print("#"*70)

    # Simulate what eval_dhb_head_e2e.py's invariants_to_actions() does
    # but with ground-truth invariants
    from eval_dhb_head_e2e import invariants_to_actions

    # First, get initial EE pose from the environment (like eval does)
    env = OffScreenRenderEnv(**{
        "bddl_file_name": bddl,
        "camera_heights": 256, "camera_widths": 256,
        "camera_names": ["agentview", "robot0_eye_in_hand"],
    })
    env.seed(0)
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(5):
        obs, _, _, _ = env.step([0.0] * 7)

    ee_pos_env = obs["robot0_eef_pos"]
    ee_quat_env = obs["robot0_eef_quat"]  # xyzw
    print(f"\n  Env EE pos:  {ee_pos_env}")
    print(f"  Env EE quat: {ee_quat_env} (xyzw)")
    print(f"  HDF5 ee_pos[0]: {ee_pos[0]}")
    print(f"  Pos diff (env vs hdf5): {np.linalg.norm(ee_pos_env - ee_pos[0])*1000:.2f} mm")

    # Truncate invariants to action_horizon=30 (like training does)
    T = 30
    gt_inv_trunc = gt_inv[:T].astype(np.float32)
    if len(gt_inv_trunc) < T:
        gt_inv_trunc = np.concatenate([gt_inv_trunc,
            np.zeros((T - len(gt_inv_trunc), 8), dtype=np.float32)])

    # Decode GT invariants using eval's function (with OSC scale 80x)
    decoded_actions_80x = invariants_to_actions(gt_inv_trunc, ee_pos_env, ee_quat_env,
                                                 osc_pos_scale=80.0)
    print_action_stats(decoded_actions_80x, "GT invariants -> eval decode (scale=80)")

    # Also try with scale 1.0 to see raw deltas
    decoded_actions_1x = invariants_to_actions(gt_inv_trunc, ee_pos_env, ee_quat_env,
                                                osc_pos_scale=1.0)
    print_action_stats(decoded_actions_1x, "GT invariants -> eval decode (scale=1.0)")

    # Compare action magnitudes
    print(f"\n  COMPARISON: original vs decoded-from-GT (scale=80x)")
    orig_trunc = orig_actions[:len(decoded_actions_80x)]
    for j, col in enumerate(["dx", "dy", "dz", "dax", "day", "daz", "grip"]):
        o_absmax = np.abs(orig_trunc[:, j]).max()
        d_absmax = np.abs(decoded_actions_80x[:, j]).max()
        ratio = d_absmax / (o_absmax + 1e-10)
        print(f"    {col:>5s}: orig_absmax={o_absmax:.6f}  decoded_absmax={d_absmax:.6f}  ratio={ratio:.2f}x")

    # ── 5. Load model and predict ────────────────────────────────────────
    print("\n" + "#"*70)
    print("# PART 5: Model-predicted invariants vs ground truth")
    print("#"*70)

    ckpt_path = "_workspace/dhb_head_e2e_200ep.pt"
    print(f"\n  Loading checkpoint: {ckpt_path}")

    # Load model (reuse eval script's load_model)
    from eval_dhb_head_e2e import load_model, get_vlm_features

    policy, dhb_head, config, _, _, _, _ = load_model(ckpt_path, device)
    T_model = config["action_horizon"]
    print(f"  Action horizon: {T_model}")

    # Get VLM features for demo_0's initial observation
    vlm_feat = get_vlm_features(policy, obs, task_name, device)
    print(f"  VLM features shape: {vlm_feat.shape}")
    adapter = SmolVLADHBAdapter.from_dhb_head(dhb_head, vlm_dim=config["vlm_dim"], action_horizon=T_model)

    # Predict invariants
    with torch.no_grad():
        pred_inv = adapter.predict_invariants_from_observation(vlm_feat, obs)[0].cpu().numpy()  # (T, 8)

    print_invariant_stats(pred_inv, "Model-predicted invariants")
    print_invariant_stats(gt_inv_trunc, "Ground-truth invariants (truncated to T)")

    # Per-column correlation
    print(f"\n  Per-column correlation (pred vs GT):")
    for j in range(8):
        cols = ["lin_m", "lin_e1", "lin_e2", "lin_e3", "ang_m", "ang_e1", "ang_e2", "ang_e3"]
        corr = np.corrcoef(pred_inv[:, j], gt_inv_trunc[:, j])[0, 1]
        rmse = np.sqrt(np.mean((pred_inv[:, j] - gt_inv_trunc[:, j])**2))
        print(f"    {cols[j]:>7s}: r={corr:.3f}  rmse={rmse:.6f}  "
              f"gt_std={gt_inv_trunc[:, j].std():.6f}  pred_std={pred_inv[:, j].std():.6f}")

    # Decode predicted invariants to actions
    decoded_pred_80x = invariants_to_actions(pred_inv, ee_pos_env, ee_quat_env,
                                              osc_pos_scale=80.0)
    print_action_stats(decoded_pred_80x, "Predicted invariants -> decoded actions (scale=80)")

    decoded_pred_1x = invariants_to_actions(pred_inv, ee_pos_env, ee_quat_env,
                                             osc_pos_scale=1.0)
    print_action_stats(decoded_pred_1x, "Predicted invariants -> decoded actions (scale=1.0)")

    # Compare predicted decoded actions vs original
    print(f"\n  COMPARISON: original actions vs predicted decoded (scale=80)")
    pred_trunc = decoded_pred_80x[:len(orig_trunc)]
    for j, col in enumerate(["dx", "dy", "dz", "dax", "day", "daz", "grip"]):
        o_absmax = np.abs(orig_trunc[:, j]).max()
        p_absmax = np.abs(pred_trunc[:, j]).max() if len(pred_trunc) > 0 else 0
        ratio = p_absmax / (o_absmax + 1e-10)
        print(f"    {col:>5s}: orig_absmax={o_absmax:.6f}  pred_absmax={p_absmax:.6f}  ratio={ratio:.2f}x")

    # ── 6. Trajectory comparison ────────────────────────────────────────
    print("\n" + "#"*70)
    print("# PART 6: Trajectory drift analysis")
    print("#"*70)

    # Decode predicted invariants to full trajectory
    q_wxyz_env = np.array([ee_quat_env[3], ee_quat_env[0],
                           ee_quat_env[1], ee_quat_env[2]])

    pred_lin = pred_inv[:, :4]
    pred_ang = pred_inv[:, 4:]
    try:
        pred_dec = decode_dhb_dr(
            pred_lin, pred_ang,
            initial_pose={"position": ee_pos_env.copy(), "quaternion": q_wxyz_env},
            method=EncodingMethod.POSITION,
            dhb_method=DHBMethod.DOUBLE_REFLECTION,
            drop_padded=False,
        )
        pred_traj = pred_dec["positions"]

        gt_lin = gt_inv_trunc[:, :4]
        gt_ang = gt_inv_trunc[:, 4:]
        gt_dec = decode_dhb_dr(
            gt_lin, gt_ang,
            initial_pose={"position": ee_pos_env.copy(), "quaternion": q_wxyz_env},
            method=EncodingMethod.POSITION,
            dhb_method=DHBMethod.DOUBLE_REFLECTION,
            drop_padded=False,
        )
        gt_traj = gt_dec["positions"]

        min_t = min(len(pred_traj), len(gt_traj), len(ee_pos))
        print(f"\n  Trajectory lengths: pred={len(pred_traj)}, gt_dec={len(gt_traj)}, orig={len(ee_pos)}")

        print(f"\n  Position at key timesteps (meters):")
        print(f"  {'step':>4s}  {'orig_x':>8s} {'orig_y':>8s} {'orig_z':>8s}  |  "
              f"{'gt_x':>8s} {'gt_y':>8s} {'gt_z':>8s}  |  "
              f"{'pred_x':>8s} {'pred_y':>8s} {'pred_z':>8s}  |  "
              f"{'err_mm':>7s}")
        for step in [0, 5, 10, 15, 20, 25, min_t-1]:
            if step >= min_t:
                continue
            o = ee_pos[step]
            g = gt_traj[step]
            p = pred_traj[step]
            err = np.linalg.norm(p - o) * 1000
            print(f"  {step:4d}  {o[0]:8.4f} {o[1]:8.4f} {o[2]:8.4f}  |  "
                  f"{g[0]:8.4f} {g[1]:8.4f} {g[2]:8.4f}  |  "
                  f"{p[0]:8.4f} {p[1]:8.4f} {p[2]:8.4f}  |  "
                  f"{err:7.1f}")

        # Overall drift
        drift_pred = np.linalg.norm(pred_traj[:min_t] - ee_pos[:min_t], axis=1)
        drift_gt = np.linalg.norm(gt_traj[:min_t] - ee_pos[:min_t], axis=1)
        print(f"\n  Predicted trajectory drift from original (mm):")
        print(f"    mean={drift_pred.mean()*1000:.1f}  max={drift_pred.max()*1000:.1f}  final={drift_pred[-1]*1000:.1f}")
        print(f"  GT-decoded trajectory drift from original (mm):")
        print(f"    mean={drift_gt.mean()*1000:.1f}  max={drift_gt.max()*1000:.1f}  final={drift_gt[-1]*1000:.1f}")

    except Exception as e:
        print(f"  Trajectory decode failed: {e}")
        import traceback; traceback.print_exc()

    # ── 7. Gripper analysis ─────────────────────────────────────────────
    print("\n" + "#"*70)
    print("# PART 7: Gripper action analysis")
    print("#"*70)
    grip = orig_actions[:, 6]
    print(f"\n  Original gripper values:")
    print(f"    unique: {np.unique(np.round(grip, 2))}")
    print(f"    mean={grip.mean():.3f}  std={grip.std():.3f}")
    print(f"    First close (< -0.5) at step: ", end="")
    close_steps = np.where(grip < -0.5)[0]
    if len(close_steps) > 0:
        print(f"{close_steps[0]} (of {len(grip)} total)")
    else:
        print("never")
    print(f"    Eval uses constant -1.0 (always closed) -- this may be wrong")

    # ── 8. Open-loop vs closed-loop diagnosis ───────────────────────────
    print("\n" + "#"*70)
    print("# PART 8: Open-loop execution diagnosis")
    print("#"*70)
    print(f"\n  Current eval: predict ONCE from initial obs, execute all {T_model} steps")
    print(f"  Original demo length: {len(orig_actions)} steps")
    print(f"  Action horizon T: {T_model}")
    if T_model < len(orig_actions):
        print(f"  WARNING: T={T_model} < demo_len={len(orig_actions)} -> trajectory truncated!")
        print(f"  Missing {len(orig_actions) - T_model} steps of the demo")

    env.close()

    # ── SUMMARY ─────────────────────────────────────────────────────────
    print("\n" + "#"*70)
    print("# SUMMARY OF FINDINGS")
    print("#"*70)
    print("""
    Check the numbers above for:
    1. OSC scale: does median(action/delta) match 80? If not, fix it.
    2. Invariant prediction: is per-column r > 0.5? If not, model underfits.
    3. Decoded action magnitude: is ratio to original ~1.0? If >>1, scale too high.
    4. Trajectory drift: is final drift < 50mm? If not, decode is broken.
    5. Gripper: is constant -1.0 appropriate? Check when original demo closes.
    6. Truncation: does T cover the full demo? If not, increase T or use closed-loop.
    """)


if __name__ == "__main__":
    main()
