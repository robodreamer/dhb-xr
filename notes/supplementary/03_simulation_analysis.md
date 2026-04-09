# Simulation Analysis — DHB-XR CoRL 2026

Detailed investigation of LIBERO simulation integration and issues encountered.

---

## What Works

### Confirmed: LIBERO simulation runs correctly

```python
MUJOCO_GL=egl  # EGL headless rendering
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
env.reset()
env.set_init_state(init_state)  # CRITICAL: set demo's initial state
for action in demo_actions:
    obs, reward, done, _ = env.step(action)
```

- **naive=100% at 0cm** (confirmed in synchronous test with `set_init_state`)
- LIBERO environment creates in ~15s on this machine (slow initialization)
- EGL headless rendering works for video capture
- `agentview_rgb` obs key IS populated when using `set_init_state` correctly

### RGB frames from correct run

The working video files (17-41KB) in `output/videos/`:
- `naive_replay_0cm.mp4` — 18 frames, max pixel 240 (real content)
- `dhb_adaptation_0cm.mp4` — 20 frames, max pixel 239 (real content)
- `comparison_0cm_agentview.mp4` — 20 frames, side-by-side 256×528

These were captured when `agentview_rgb` was populated correctly.

---

## Root Cause: Why DHB Videos Show Only Gripper Moving

### OSC_POSE action space in LIBERO

LIBERO uses robosuite's `OSC_POSE` (Operational Space Control) controller.
This controller takes normalized action commands:

```
action = [Δx_norm, Δy_norm, Δz_norm, Δrx_norm, Δry_norm, Δrz_norm, gripper]
```

where the position deltas are **normalized to roughly [-1, 1]** by the controller's
internal gain/scaling. The actual mapping is:

```
EEF_velocity = action[:3] * controller_scale
```

where `controller_scale` is NOT constant — it varies per step depending on:
- Current joint configuration
- Distance to singularity
- Controller gain schedule
- Workspace boundary proximity

### Measured scaling (Task 0, LIBERO-Spatial)

```
Per-step scale statistics:
  min:    20.6×
  max:   577.1×
  median: 73.2×
  std:    93.3×
```

**Consequence**: Using a single scale factor (e.g., mean=83×) for DHB position
deltas results in the arm barely moving on some steps (scale too low) and
potentially moving too far on others.

### Why DHB actions appear to "only move the gripper"

When we compute:
```python
delta_m = dec['positions'][i+1] - dec['positions'][i]  # ~0.002m per step
apts[i,:3] = delta_m * 83  # = 0.17 — too small for steps needing 577×
```

The arm receives OSC commands that are 1/7 to 1/577 of what's needed.
The result: arm barely moves; gripper cycles through original open/close timing
at wrong position → "only gripper moving."

---

## Path to Fix

### Option 1: Waypoint PD controller (recommended)

Instead of replaying fixed OSC actions, use the adapted EEF trajectory as
waypoints and compute OSC actions via a PD controller:

```python
for target_pos in adapted_positions:
    current_pos = obs['robot0_eef_pos']
    error = target_pos - current_pos
    action[:3] = Kp * error  # proportional control
    action[3:6] = ... # orientation error
    obs, reward, done, _ = env.step(action)
```

This bypasses the action scaling issue entirely — the controller handles scaling internally.

**Requires**: Tuning `Kp`, handling orientation, proper convergence criterion.

### Option 2: Fatrop with proper action space

Use Fatrop to optimize directly in the OSC action space:
- Define the optimization problem in action space, not SE(3) space
- This requires knowing the forward kinematics mapping

More complex but more principled.

### Option 3: Perturb object in init_state

For the proper experiment (robot stays fixed, OBJECT moves):

```python
init_state = np.array(f['data'][dk]['states'][0])  # shape (92,)
# The LIBERO state vector encodes object positions
# Need to identify which indices correspond to object position
# Then: init_state[object_pos_idx:object_pos_idx+3] += offset
env.set_init_state(modified_init_state)
```

LIBERO state vector format (92-dim):
- Indices 0-6: robot joint positions
- Indices 7-13: robot joint velocities
- Indices 14-20: robot EEF position + quat
- Indices 21+: object states (position + quat + velocity)

Identifying exact indices requires inspecting `env.sim.data.qpos` mapping.

---

## Simulation Setup for Future Runs

Recommended setup script for proper DHB simulation evaluation:

```python
import os; os.environ['MUJOCO_GL'] = 'egl'  # Required for headless

# Correct setup:
env.reset()
env.set_init_state(init_state)  # Use demo's actual initial state
# Perturb object: env.sim.data.qpos[object_start_idx:object_start_idx+3] += offset
env.sim.forward()  # Apply state

# Then run with waypoint PD:
Kp = 10.0  # tune
for target_pos in adapted_traj:
    for step in range(max_steps_per_waypoint):
        curr = obs['robot0_eef_pos']
        err = target_pos - curr
        if np.linalg.norm(err) < 0.01: break
        action = np.zeros(7)
        action[:3] = np.clip(Kp * err, -1, 1)
        action[6] = original_gripper_at_this_step
        obs, r, done, _ = env.step(action)
```

---

## Current Status in Paper

- `tab:spatial_main` simulation rows: `--` (pending)
- Geometric rows: **0.0mm** at all offsets — valid primary evidence
- Timeline for fix: Week 3-4 (requires Fatrop installation + PD controller tuning)
