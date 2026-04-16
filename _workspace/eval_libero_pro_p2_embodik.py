#!/usr/bin/env python3
"""
LIBERO-PRO P2 evaluation using embodik IK + JOINT_POSITION controller.

Improvement over eval_libero_pro_p2.py:
- Phase 1 (pick): Same original OSC P-controller
- Phase 2 (place): DHB-adapted trajectory → embodik IK → JOINT_POSITION

The embodik IK path has proven much better (100% on single-demo) vs OSC (70-80%).

Three-stage pipeline:
  Stage 1: Compute DHB-adapted trajectories (libero env)
  Stage 2: embodik IK on adapted trajectories (pixi env)
  Stage 3: Execute in perturbed LIBERO environment (libero env)

Usage:
  # Stage 1: compute adapted poses
  MUJOCO_GL=egl ~/miniforge3/envs/libero/bin/python \
      _workspace/eval_libero_pro_p2_embodik.py --stage adapt

  # Stage 2: IK
  cd ~/Projects/repos/embodik-singularity-recovery && \
      pixi run python ~/Projects/repos/dhb_xr/_workspace/eval_libero_pro_p2_embodik.py --stage ik

  # Stage 3: Execute
  MUJOCO_GL=egl ~/miniforge3/envs/libero/bin/python \
      _workspace/eval_libero_pro_p2_embodik.py --stage exec
"""
import sys, os, json, time, argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))  # for examples.integration
sys.path.insert(0, "/home/andypark/Projects/repos/LIBERO")
sys.path.insert(0, "/home/andypark/Projects/repos/LIBERO-PRO")

DEMO_HDF5 = ("/home/andypark/Projects/data/libero/libero_spatial/"
             "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5")
BDDL_ORIG = ("/home/andypark/Projects/repos/LIBERO/libero/libero/bddl_files/libero_spatial/"
             "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl")

ADAPTED_NPZ = "/tmp/p2_embodik_adapted.npz"
JOINTS_NPZ  = "/tmp/p2_embodik_joints.npz"
RESULTS_JSON = "_workspace/eval_libero_pro_p2_embodik_results.json"
N_DEMOS = 10  # Test on first 10 demos


def stage_adapt(args):
    """Compute P2 swap, DHB-adapt placement trajectory, save poses."""
    import h5py
    from libero.libero.envs import OffScreenRenderEnv
    from perturbation import SwapPerturbator, PerturbFlags
    from dhb_xr.encoder.dhb_dr import encode_dhb_dr
    from dhb_xr.decoder.dhb_dr import decode_dhb_dr
    from dhb_xr.core.types import EncodingMethod, DHBMethod
    from scipy.spatial.transform import Rotation
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("_dhb_demo",
        str(Path(__file__).parent.parent / "examples/integration/libero_pro_dhb_demo.py"))
    _mod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)
    adapt_trajectory = _mod.adapt_trajectory

    print("="*60)
    print("Stage 1: DHB P2 adaptation")
    print("="*60)

    # Apply P2 perturbation (swap spatial relations)
    cfg_path = "/home/andypark/Projects/repos/LIBERO-PRO/libero_ood/ood_spatial_relation.yaml"
    with open(BDDL_ORIG) as f:
        orig_bddl = f.read()
    from perturbation import BDDLParser
    parser = BDDLParser(orig_bddl)
    perturbed_bddl = SwapPerturbator(parser, cfg_path).perturb("libero_spatial",
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate")
    BDDL_PERTURBED = "/tmp/p2_perturbed.bddl"
    with open(BDDL_PERTURBED, "w") as f:
        f.write(perturbed_bddl)
    print(f"P2 BDDL written to {BDDL_PERTURBED}")

    # Get bowl and plate positions in both environments
    def get_object_pos(bddl, init_state, obj_name, apply_init_state=True):
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
        env.seed(0)
        env.reset()
        if apply_init_state:
            env.set_init_state(init_state)
        obs = env.env._get_observations()
        pos = obs.get(f"{obj_name}_pos", None)
        if pos is None:
            # Try sim qpos
            for name in obs.keys():
                if obj_name.split("_1")[0] in name and "pos" in name:
                    pos = obs[name]
                    break
        env.close()
        return np.array(pos) if pos is not None else None

    f = h5py.File(DEMO_HDF5, "r")
    all_adapted = {}

    for demo_idx in range(N_DEMOS):
        key = f"demo_{demo_idx}"
        if key not in f["data"]:
            continue
        d = f["data"][key]
        init_state = np.array(d["states"][0])
        actions = np.array(d["actions"])
        robot_states = np.array(d["robot_states"])
        # positions from robot_states (format: time, joint_states..., eef_pos(3), eef_quat(4))
        ee_pos = robot_states[:, 2:5].astype(np.float64)
        ee_quat_wxyz = robot_states[:, 5:9].astype(np.float64)
        ee_quat = ee_quat_wxyz[:, [1,2,3,0]]  # → xyzw

        gripper_states = np.array(d["obs"]["robot0_gripper_qpos"]) if "robot0_gripper_qpos" in d["obs"] else None
        T = len(actions)

        # Get object positions
        try:
            # CRITICAL: P2 env must NOT apply init_state (would override BDDL swap)
            orig_plate = get_object_pos(BDDL_ORIG, init_state, "plate_1", apply_init_state=True)
            pert_plate = get_object_pos(BDDL_PERTURBED, init_state, "plate_1", apply_init_state=False)
            orig_bowl  = get_object_pos(BDDL_ORIG, init_state, "akita_black_bowl_1", apply_init_state=True)
            pert_bowl  = get_object_pos(BDDL_PERTURBED, init_state, "akita_black_bowl_1", apply_init_state=False)
        except Exception as e:
            print(f"  Demo {demo_idx}: object pos failed: {e}")
            continue

        if orig_plate is None or pert_plate is None:
            print(f"  Demo {demo_idx}: plate pos not found, skipping")
            continue

        plate_offset = pert_plate - orig_plate
        bowl_offset = (pert_bowl - orig_bowl) if (pert_bowl is not None and orig_bowl is not None) else np.zeros(3)
        print(f"\n  Demo {demo_idx}: plate_offset={plate_offset}, bowl_offset={bowl_offset}")

        # Find grasp/release steps
        GRASP_STEP = T // 2
        RELEASE_STEP = T - 1
        for i in range(T):
            if actions[i, 6] > 0.5:
                GRASP_STEP = i
                break
        for i in range(GRASP_STEP, T):
            if actions[i, 6] < 0.0:
                RELEASE_STEP = i
                break

        # Phase 1: pick trajectory (use original EE positions, shifted by bowl offset)
        pick_pos = ee_pos[:GRASP_STEP+1].copy()
        pick_pos += bowl_offset  # apply bowl offset
        pick_quat = ee_quat[:GRASP_STEP+1].copy()

        # Phase 2: DHB-adapt placement trajectory
        place_pos = ee_pos[GRASP_STEP:RELEASE_STEP+1]
        place_quat = ee_quat[GRASP_STEP:RELEASE_STEP+1]

        # Placement goal: maintain same Z-height above table and same XY relative offset
        # The key insight: EEF should be ~same height above plate surface as in original demo
        # EEF Z at release: ~0.937 (original plate Z: 0.970) → EEF ~3cm below plate center
        # But perturbed plate is at same Z=0.970, so same Z-offset applies
        if orig_plate is not None:
            # Keep Z-offset from plate (EEF was Z_release - Z_plate = ~-0.033 below plate top)
            eef_z_rel = place_pos[-1][2] - orig_plate[2]  # typically -0.03 to 0
            # Keep X-offset from plate (small)
            eef_x_rel = place_pos[-1][0] - orig_plate[0]
            # NEW: place goal is directly above the perturbed plate, same offsets
            new_goal = np.array([
                pert_plate[0] + eef_x_rel,   # X: small offset from plate center
                pert_plate[1],                 # Y: directly at new plate position
                pert_plate[2] + eef_z_rel      # Z: same height relative to plate
            ])
            print(f"    Placement goal: orig={place_pos[-1]} → new={new_goal}")
        else:
            new_goal = place_pos[-1] + (pert_plate - orig_plate) if orig_plate is not None else place_pos[-1]

        try:
            adapted = adapt_trajectory(
                place_pos, place_quat,
                new_start_pos=place_pos[0] + bowl_offset,
                new_start_quat=place_quat[0],
                new_goal_pos=new_goal,
                new_goal_quat=place_quat[-1],
                traj_length=min(len(place_pos), 50),
            )
            adapted_place_pos = adapted["positions"]
            adapted_place_quat_wxyz = adapted.get("quaternions", np.tile([1,0,0,0], (len(adapted_place_pos),1)))
            # Convert quat from wxyz to xyzw for embodik
            adapted_place_quat_xyzw = np.column_stack([
                adapted_place_quat_wxyz[:,1], adapted_place_quat_wxyz[:,2],
                adapted_place_quat_wxyz[:,3], adapted_place_quat_wxyz[:,0]
            ])
            err_start = np.linalg.norm(adapted_place_pos[0] - (place_pos[0] + bowl_offset)) * 1000
            err_goal = np.linalg.norm(adapted_place_pos[-1] - new_goal) * 1000
            print(f"    DHB adapt: start_err={err_start:.1f}mm, goal_err={err_goal:.1f}mm, {len(adapted_place_pos)} steps")
        except Exception as e:
            print(f"    DHB adapt failed: {e}")
            continue

        # Full adapted trajectory: pick + place
        full_pos = np.concatenate([pick_pos, adapted_place_pos], axis=0)
        full_quat = np.concatenate([pick_quat, adapted_place_quat_xyzw], axis=0)

        # Gripper: close at GRASP_STEP, open at RELEASE_STEP
        gripper_cmd = np.ones(len(full_pos)) * -1.0  # open
        gripper_cmd[len(pick_pos):] = 1.0  # close during carry
        gripper_cmd[len(pick_pos) + len(adapted_place_pos) - 5:] = -1.0  # release at end

        all_adapted[f"demo{demo_idx}_pos"] = full_pos
        all_adapted[f"demo{demo_idx}_quat"] = full_quat
        all_adapted[f"demo{demo_idx}_grip"] = gripper_cmd
        all_adapted[f"demo{demo_idx}_init"] = init_state
        all_adapted[f"demo{demo_idx}_grasp_step"] = np.array([len(pick_pos)])

    f.close()

    np.savez(ADAPTED_NPZ, **all_adapted,
             bddl_perturbed=np.array([BDDL_PERTURBED]),
             n_demos=np.array([len([k for k in all_adapted if k.endswith("_pos")])]))
    print(f"\nSaved {len([k for k in all_adapted if k.endswith('_pos')])} adapted trajectories to {ADAPTED_NPZ}")


def stage_ik(args):
    """Solve IK for all adapted trajectories."""
    import embodik
    from robot_descriptions import panda_description

    print("="*60)
    print("Stage 2: embodik IK")
    print("="*60)

    data = np.load(ADAPTED_NPZ, allow_pickle=True)
    demo_keys = [k.replace("_pos","") for k in data.files if k.endswith("_pos") and k.startswith("demo")]

    robot = embodik.RobotModel(urdf_path=panda_description.URDF_PATH)
    solver = embodik.KinematicsSolver(robot)
    ik_opts = embodik.PositionIKOptions()
    ik_opts.position_tolerance = 0.001
    ik_opts.orientation_tolerance = 100.0
    ik_opts.max_iterations = 200

    all_joints = {}

    for dk in demo_keys:
        pos = data[f"{dk}_pos"]
        init_state = data[f"{dk}_init"]
        T = len(pos)

        # Get initial joint config from LIBERO init_state
        # LIBERO qpos format: [time(1), robot_joints(7), gripper(2), object_qpos...]
        q_init = np.zeros(robot.nq)
        if len(init_state) >= 8:
            q_init[:7] = init_state[1:8]  # robot joint positions
        robot.update_configuration(q_init)
        fk_t0 = np.array(robot.get_frame_pose("panda_hand_tcp").translation, dtype=float)
        base_offset = pos[0] - fk_t0
        print(f"\n  {dk}: base_offset_Z={base_offset[2]:.3f}")

        q_traj = np.zeros((T, robot.nq))
        q_current = q_init.copy()
        errors = []

        for t in range(T):
            target_pos = pos[t] - base_offset
            robot.update_configuration(q_current)
            cur_rot = np.array(robot.get_frame_pose("panda_hand_tcp").rotation, dtype=float)
            target_T = np.eye(4, order="F")
            target_T[:3,:3] = cur_rot
            target_T[:3,3] = target_pos

            result = solver.solve_position(q_current, target_T, "panda_hand_tcp", ik_opts)
            if result.status == embodik.SolverStatus.SUCCESS:
                q_solution = np.array(result.q_solution)
                q_traj[t] = q_solution
                q_current = q_solution.copy()
                robot.update_configuration(q_solution)
                fk = np.array(robot.get_frame_pose("panda_hand_tcp").translation, dtype=float)
                err = np.linalg.norm(fk + base_offset - pos[t]) * 1000
                errors.append(err)
            else:
                q_traj[t] = q_current.copy()
                errors.append(float("nan"))

        mean_err = np.nanmean(errors)
        n_fail = np.sum(np.isnan(errors))
        print(f"  {dk}: IK mean_err={mean_err:.2f}mm, failures={n_fail}/{T}")
        all_joints[f"{dk}_joints"] = q_traj
        all_joints[f"{dk}_base_offset"] = base_offset

    np.savez(JOINTS_NPZ, **all_joints,
             **{k: data[k] for k in data.files if k.endswith("_grip") or k.endswith("_init") or k == "bddl_perturbed" or k.endswith("_grasp_step")})
    print(f"\nSaved joint trajectories to {JOINTS_NPZ}")


def stage_exec(args):
    """Execute joint trajectories in LIBERO with JOINT_POSITION controller."""
    from libero.libero.envs import OffScreenRenderEnv

    print("="*60)
    print("Stage 3: LIBERO-PRO P2 execution (embodik IK)")
    print("="*60)

    data = np.load(JOINTS_NPZ, allow_pickle=True)
    BDDL_PERTURBED = str(data["bddl_perturbed"][0])
    demo_keys = [k.replace("_joints","") for k in data.files if k.endswith("_joints")]

    results = {"mode": "libero_pro_p2_embodik", "trials": [], "summary": {}}
    successes = 0
    total = 0
    output_max = 0.5  # rad
    kp = 150
    substeps = 3

    for dk in demo_keys:
        q_traj = data[f"{dk}_joints"]
        grip = data[f"{dk}_grip"]
        init_state = data[f"{dk}_init"]
        grasp_step = int(data[f"{dk}_grasp_step"][0]) if f"{dk}_grasp_step" in data else len(q_traj)//2

        env = OffScreenRenderEnv(bddl_file_name=BDDL_PERTURBED,
                                 controller="JOINT_POSITION",
                                 camera_heights=128, camera_widths=128)
        env.seed(0)
        obs = env.reset()
        # Let physics settle — P2 BDDL places objects at Z=0.97 (above table).
        # Need 200+ steps for objects to settle onto table surface (~Z=0.90).
        for _ in range(250):
            obs, _, _, _ = env.step(np.zeros(7))
        obs = env.env._get_observations()
        # Verify settled positions
        bowl_pos = obs.get("akita_black_bowl_1_pos", np.zeros(3))
        plate_pos = obs.get("plate_1_pos", np.zeros(3))
        print(f"  Settled: bowl={bowl_pos}, plate={plate_pos}")

        robot = env.env.robots[0]
        ctrl = robot.controller
        ctrl.output_max = ctrl.nums2array(output_max, len(ctrl.qpos_index))
        ctrl.output_min = ctrl.nums2array(-output_max, len(ctrl.qpos_index))
        ctrl.kp = ctrl.nums2array(kp, len(ctrl.qpos_index))
        ctrl.kd = 2 * np.sqrt(ctrl.kp)

        reward = 0.0
        for t in range(len(q_traj)):
            grip_idx = min(t, len(grip)-1)
            for _ in range(substeps):
                q_target = q_traj[t, :7]
                q_current = np.array(robot._joint_positions)
                delta = np.clip((q_target - q_current) / output_max, -1.0, 1.0)
                full_action = np.concatenate([delta, [grip[grip_idx]]])
                obs, r, done, _ = env.step(full_action)
                if r > reward:
                    reward = r

        # Settling
        for _ in range(100):
            q_target = q_traj[-1, :7]
            q_current = np.array(robot._joint_positions)
            delta = np.clip((q_target - q_current) / output_max, -1.0, 1.0)
            obs, r, done, _ = env.step(np.concatenate([delta, [1.0]]))
            if r > reward:
                reward = r

        success = env.check_success()
        env.close()
        successes += int(success)
        total += 1

        ee_final = obs.get("robot0_eef_pos", np.zeros(3))
        print(f"  {dk}: {'SUCCESS' if success else 'FAIL'} (reward={reward:.2f})")
        results["trials"].append({"demo": dk, "success": bool(success), "reward": float(reward)})

    rate = successes / max(total, 1) * 100
    results["summary"] = {"successes": successes, "total": total, "rate": rate}
    print(f"\n=== LIBERO-PRO P2 embodik IK: {successes}/{total} = {rate:.0f}% ===")

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {RESULTS_JSON}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["adapt", "ik", "exec"])
    args = parser.parse_args()

    if args.stage == "adapt":
        stage_adapt(args)
    elif args.stage == "ik":
        stage_ik(args)
    elif args.stage == "exec":
        stage_exec(args)


if __name__ == "__main__":
    main()
