#!/usr/bin/env python3
"""
Generate CORRECT training targets for P2-augmented DHBActionHead training.

The key insight: for spatial generalization to P2 scenes, the training targets
must be the Fatrop-ADAPTED invariants (pointing to the P2 plate), NOT the
original demo invariants.

This script:
1. For each of the 50 demos, computes the Fatrop-adapted placement trajectory
   for the P2 scene (plate moved 17cm)
2. Encodes these adapted trajectories to DHB invariants
3. Saves as training targets for P2 observations

These targets + the P2 observations in p2_obs_cache.npz form the correct
training data for the DHBActionHead to learn P2 spatial generalization.

Usage:
    pixi run python _workspace/gen_p2_aug_targets.py
    # Then retrain:
    MUJOCO_GL=egl ~/miniforge3/envs/smolvla/bin/python \
        _workspace/train_dhb_head_p2_correct.py
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, "/home/andypark/Projects/repos/LIBERO")
sys.path.insert(0, "/home/andypark/Projects/repos/LIBERO-PRO")

DEMO_HDF5 = ("/home/andypark/Projects/data/libero/libero_spatial/"
             "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5")
N_DEMOS = 50
ACTION_HORIZON = 100

def main():
    import h5py
    from scipy.spatial.transform import Rotation
    from dhb_xr.encoder.dhb_dr import encode_dhb_dr
    from dhb_xr.optimization.fatrop_solver import generate_trajectory_fatrop

    print("Generating P2-adapted training targets...")
    print(f"  {N_DEMOS} demos × Fatrop adaptation → adapted invariants")

    # Get settled object positions for P2 env
    # Always define with seed=42 to get SAME cookies_1 swap as eval_libero_pro_p2.py
    # This ensures training targets match the eval environment (plate at Y=0.042)
    import random
    random.seed(42)
    from perturbation import BDDLParser, SwapPerturbator
    BDDL_ORIG = ("/home/andypark/Projects/repos/LIBERO/libero/libero/bddl_files/"
                 "libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl")
    OOD_CFG = "/home/andypark/Projects/repos/LIBERO-PRO/libero_ood/ood_spatial_relation.yaml"
    with open(BDDL_ORIG) as f:
        orig = f.read()
    pert = SwapPerturbator(BDDLParser(orig), OOD_CFG).perturb(
        "libero_spatial",
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate")
    BDDL_P2 = "/tmp/p2_perturbed.bddl"
    with open(BDDL_P2, "w") as f:
        f.write(pert)
    print(f"  Using P2 BDDL: {BDDL_P2} (cookies_1 swap, seed=42)")

    # Get settled positions
    from libero.libero.envs import OffScreenRenderEnv
    def get_settled(bddl, obj, settle=100):
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=64, camera_widths=64)
        env.seed(0); env.reset()
        for _ in range(settle): env.step(np.zeros(7))
        obs = env.env._get_observations()
        pos = obs.get(f"{obj}_pos", None)
        env.close()
        return pos.copy() if pos is not None else None

    orig_plate = get_settled(BDDL_ORIG, "plate_1")
    pert_plate = get_settled(BDDL_P2, "plate_1")
    orig_bowl = get_settled(BDDL_ORIG, "akita_black_bowl_1")
    pert_bowl = get_settled(BDDL_P2, "akita_black_bowl_1")
    print(f"  Orig plate: {orig_plate}")
    print(f"  P2 plate:   {pert_plate}")

    bowl_offset = pert_bowl - orig_bowl

    # Process each demo
    p2_targets = []
    orig_targets = []
    p2_failed = 0

    f = h5py.File(DEMO_HDF5)
    demo_keys = sorted(f["data"].keys())[:N_DEMOS]

    for i, key in enumerate(demo_keys):
        d = f["data"][key]
        rs = np.array(d["robot_states"])
        ee_pos = rs[:, 2:5].astype(np.float64)
        ee_quat = rs[:, 5:9].astype(np.float64)  # HDF5 stores wxyz directly (w at index 0)
        T = len(ee_pos)

        # Find phase boundaries
        actions = np.array(d["actions"])
        GRASP = T // 2
        for j in range(T):
            if actions[j, 6] > 0.5: GRASP = j; break
        RELEASE = T - 1
        for j in range(GRASP, T):
            if actions[j, 6] < 0.0: RELEASE = j; break

        # Original target invariants (full trajectory)
        try:
            enc = encode_dhb_dr(ee_pos, ee_quat, method="pos", use_default_initial_frames=True)
            orig_inv = np.concatenate([enc["linear_motion_invariants"],
                                        enc["angular_motion_invariants"]], axis=1)
            T_horizon = ACTION_HORIZON
            orig_t = orig_inv[:T_horizon].astype(np.float32)
            if len(orig_t) < T_horizon:
                orig_t = np.concatenate([orig_t, np.zeros((T_horizon-len(orig_t), 8), dtype=np.float32)])
            orig_targets.append(orig_t)
        except Exception as e:
            orig_targets.append(np.zeros((ACTION_HORIZON, 8), dtype=np.float32))

        # P2-adapted target: Fatrop-adapted placement trajectory
        try:
            place_pos = ee_pos[GRASP:RELEASE+1]
            place_quat = ee_quat[GRASP:RELEASE+1]
            eef_rel = place_pos[-1] - orig_plate
            new_goal = np.array([pert_plate[0]+eef_rel[0], pert_plate[1], pert_plate[2]+eef_rel[2]])
            new_start = place_pos[0] + bowl_offset

            res = generate_trajectory_fatrop(
                place_pos, place_quat,
                pose_target_init={"position": new_start, "quaternion": place_quat[0]},
                pose_target_final={"position": new_goal, "quaternion": place_quat[-1]},
                traj_length=min(len(place_pos), 50),
            )
            adapted_pos = res["positions"]
            n_place = len(adapted_pos)
            # Use Fatrop-adapted quaternions for full SE(3) adaptation.
            # The fix in fatrop_solver.py decodes optimized invariants through
            # decode_dhb_dr to get correct EE orientations (not the broken OCP
            # rotation state which only represented the linear frame).
            adapted_quat = res.get("quaternions", place_quat[:n_place])

            # Full P2 trajectory: pick phase (shifted) + adapted place phase
            p2_full_pos = np.concatenate([ee_pos[:GRASP] + bowl_offset, adapted_pos], axis=0)
            p2_full_quat = np.concatenate([ee_quat[:GRASP], adapted_quat], axis=0)

            # Encode to invariants
            enc_p2 = encode_dhb_dr(p2_full_pos, p2_full_quat, method="pos", use_default_initial_frames=True)
            p2_inv = np.concatenate([enc_p2["linear_motion_invariants"],
                                      enc_p2["angular_motion_invariants"]], axis=1)
            p2_t = p2_inv[:T_horizon].astype(np.float32)
            if len(p2_t) < T_horizon:
                p2_t = np.concatenate([p2_t, np.zeros((T_horizon-len(p2_t), 8), dtype=np.float32)])
            p2_targets.append(p2_t)
            print(f"  Demo {i}: OK (solver={res.get('solver','?')})")
        except Exception as e:
            print(f"  Demo {i}: Fatrop failed ({e}), using original target")
            p2_targets.append(orig_targets[-1].copy())
            p2_failed += 1

    f.close()

    # Save
    orig_arr = np.stack(orig_targets)  # (N, T, 8) original targets
    p2_arr = np.stack(p2_targets)      # (N, T, 8) P2-adapted targets
    print(f"\nOriginal targets: {orig_arr.shape}")
    print(f"P2-adapted targets: {p2_arr.shape}")
    print(f"Fatrop failures: {p2_failed}/{N_DEMOS}")

    out = "_workspace/p2_aug_targets.npz"
    np.savez(out, orig_targets=orig_arr, p2_targets=p2_arr,
             orig_plate=orig_plate, p2_plate=pert_plate, bowl_offset=bowl_offset)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
