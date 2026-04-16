"""
Record P2 failure videos for seeds 0 and 8.
Uses LIBERO OffScreenRenderEnv, replaying the IK-solved joint trajectories.
Saves annotated video to _workspace/p2_failure_seed{N}.mp4

Usage (in libero env):
    MUJOCO_GL=egl ~/miniforge3/envs/libero/bin/python _workspace/record_p2_failure_video.py
"""
import sys, os
import numpy as np
import cv2

sys.path.insert(0, '/home/andypark/Projects/repos/LIBERO')
sys.path.insert(0, '/home/andypark/Projects/repos/dhb_xr/src')

# ─── Load joint trajectories ──────────────────────────────────────────────
joints_path = '/tmp/dhb_spatial_p2_joints.npz'
poses_path  = '/tmp/dhb_spatial_p2_poses.npz'

joints_data = np.load(joints_path, allow_pickle=True)
poses_data  = np.load(poses_path,  allow_pickle=True)

BDDL_P2 = '/tmp/p2_perturbed.bddl'
OUT_DIR  = '/home/andypark/Projects/repos/dhb_xr/_workspace'
FPS      = 10
IMG_W, IMG_H = 720, 720

SEEDS_TO_RECORD = [0, 8]

def add_overlay(frame, timestep, ee_pos, gripper_val, seed_n, total_steps):
    """Add informational text overlay to a frame."""
    frame = frame.copy()
    h, w = frame.shape[:2]

    # Semi-transparent background for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    font      = cv2.FONT_HERSHEY_SIMPLEX
    font_sm   = cv2.FONT_HERSHEY_SIMPLEX
    color_w   = (255, 255, 255)
    color_r   = (80, 80, 255)   # BGR red
    color_g   = (80, 220, 80)   # BGR green

    # Header
    cv2.putText(frame, f'DHB-XR P2 Failure — Seed {seed_n}  [FAIL]',
                (10, 26), font, 0.75, color_r, 2)

    # Step counter
    cv2.putText(frame, f't = {timestep:03d}/{total_steps-1}',
                (10, 52), font_sm, 0.6, color_w, 1)

    # EE position
    cv2.putText(frame,
                f'EE: X={ee_pos[0]:+.3f}  Y={ee_pos[1]:+.3f}  Z={ee_pos[2]:+.3f}',
                (10, 74), font_sm, 0.55, color_w, 1)

    # Gripper state
    g_txt   = 'CLOSED' if gripper_val > 0 else 'open'
    g_color = color_r if gripper_val > 0 else color_w
    cv2.putText(frame, f'Gripper: {g_txt}', (10, 96), font_sm, 0.55, g_color, 1)

    # P2 plate target reminder (bottom bar)
    cv2.rectangle(frame, (0, h - 30), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, 'P2 target: X=0.083  Y=0.042  Z=0.902     Bowl pick: Y=0.21',
                (8, h - 10), font_sm, 0.45, (180, 180, 50), 1)

    return frame


def record_seed(seed_n):
    joints   = joints_data[f'seed{seed_n}_joints']    # (98, 9) — 7 joints + 2 gripper
    gripper  = joints_data[f'gripper_seed{seed_n}']   # variable length
    ee_traj  = poses_data[f'seed{seed_n}_pos']         # (98, 3)
    init_st  = joints_data[f'init_state_seed{seed_n}'] # (92,) flattened sim state

    T = len(joints)
    print(f"\n=== Recording seed {seed_n} ({T} steps) ===")

    out_path = os.path.join(OUT_DIR, f'p2_failure_seed{seed_n}.mp4')
    fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
    writer   = cv2.VideoWriter(out_path, fourcc, FPS, (IMG_W, IMG_H))

    try:
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as e:
        print(f"ERROR: Could not import OffScreenRenderEnv: {e}")
        writer.release()
        return None

    env = OffScreenRenderEnv(
        bddl_file_name=BDDL_P2,
        camera_names=['agentview'],
        camera_heights=IMG_H,
        camera_widths=IMG_W,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera='agentview',
    )
    env.seed(42)
    env.reset()

    # Restore the exact init state used during evaluation
    try:
        env.set_init_state(init_st)
        print(f"  Restored init state (len={len(init_st)})")
    except Exception as e:
        print(f"  Warning: could not restore init state: {e} — using default reset")
        env.reset()

    frames_written = 0
    for t in range(T):
        joint_action = joints[t].copy()   # (9,) — 7 joints + 2 fingers

        # Convert 9-dim joint+gripper to 7-dim action expected by LIBERO
        # LIBERO actions: [7 joint deltas OR positions, gripper command]
        # Joints[0:7] = joint positions, joints[7:9] = finger positions
        # Gripper command: use gripper array (1=close, -1=open)
        g_val = float(gripper[t]) if t < len(gripper) else 1.0
        action = np.zeros(7)
        action[:6] = joint_action[:6]
        action[6]  = g_val

        try:
            obs, reward, done, info = env.step(action)
        except Exception as e:
            print(f"  Step {t} error: {e}")
            break

        # Capture agentview image
        try:
            img = obs['agentview_image']  # (H, W, 3) uint8 RGB
            if img.shape[0] != IMG_H or img.shape[1] != IMG_W:
                img = cv2.resize(img, (IMG_W, IMG_H))
        except KeyError:
            # Try alternate keys
            for k in obs.keys():
                if 'image' in k.lower():
                    img = obs[k]
                    if img.shape[0] != IMG_H or img.shape[1] != IMG_W:
                        img = cv2.resize(img, (IMG_W, IMG_H))
                    break
            else:
                img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

        # LIBERO images may be flipped vertically
        img = np.flipud(img).astype(np.uint8)

        # Get current EE position
        ee_pos = ee_traj[t] if t < len(ee_traj) else ee_traj[-1]

        # Add overlay
        frame = add_overlay(img, t, ee_pos, g_val, seed_n, T)

        # Convert RGB → BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
        frames_written += 1

        if t % 20 == 0:
            print(f"  t={t:3d}/{T} EE_Y={ee_pos[1]:.4f} gripper={g_val:+.0f}")

    writer.release()
    env.close()
    print(f"  Saved {frames_written} frames → {out_path}")
    return out_path


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    results = {}
    for s in SEEDS_TO_RECORD:
        try:
            path = record_seed(s)
            results[s] = path or 'failed'
        except Exception as e:
            import traceback
            print(f"ERROR recording seed {s}: {e}")
            traceback.print_exc()
            results[s] = f'error: {e}'

    print("\n=== Recording Results ===")
    for s, p in results.items():
        print(f"  seed{s}: {p}")
