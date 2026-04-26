# StarVLA Qwen+DHB Direct vs EmbodiK Diagnostic

Date: 2026-04-17

## Question

After fixing the direct rollout action semantics, why does the end effector no longer flip 180 degrees, yet still barely move?

## Findings

### 1. The old direct rollout video was still invalid

Two rollout-side issues were present:

- The learned-policy rollout was still using absolute-action semantics instead of delta-action semantics.
- The rollout was unnormalizing model outputs with stale action statistics from the old absolute-action dataset, not the checkpoint's own delta-action run.

These are now fixed in [scripts/starvla_dhb_libero_rollout_smoke.py](/home/andypark/Projects/repos/dhb-xr/scripts/starvla_dhb_libero_rollout_smoke.py).

### 2. The corrected direct rollout no longer shows the same 180-degree wrist snap

The current direct rollout is now numerically sane:

- first action:
  - position delta: `[0.00148, 0.00027, -0.00284]`
  - rotation delta: `[0.00776, -0.01274, -0.00310]`
- last action:
  - position delta: `[0.00190, 0.00019, -0.00336]`
  - rotation delta: `[0.00752, -0.01213, -0.00309]`

So the remaining failure is no longer a gross action-contract bug.

### 3. The direct `predict_action()` path is under-moving in position

For the exact initial observation of the reference rollout:

- direct `predict_action()` mean position-delta norm: `0.00355 m`
- direct `predict_action()` mean rotation-delta norm: `0.01393 rad`

For the ground-truth `demo_0` from the same task:

- demo mean position-delta norm: `0.00878 m`
- demo 95th percentile position-delta norm: `0.01413 m`
- demo mean rotation-delta norm: `0.01004 rad`
- demo 95th percentile rotation-delta norm: `0.02166 rad`

Interpretation:

- position deltas from direct StarVLA are about `2.5x` smaller than the demo average
- rotation deltas are in the correct rough scale

This matches what the corrected rollout video shows: the robot no longer flips, but it barely translates enough to do the task.

### 4. The DHB `predict_trajectory()` path is producing a meaningful EE plan

On the same initial observation:

- trajectory waypoints: `6`
- mean inter-waypoint position step: `0.0225 m`
- total predicted displacement: `0.1092 m`

This is qualitatively different from the direct action head:

- direct action path: tiny local deltas
- DHB trajectory path: a structured multi-waypoint EE motion

This explains why the `embodik` bridge is still the more promising execution path for StarVLA+DHB.

## Conclusion

The current direct StarVLA rollout is no longer invalid, but it is not the right evaluation path for DHB right now.

- The 180-degree wrist snap was largely caused by rollout semantics and stale normalization.
- After fixing that, the direct controller path still under-moves in position.
- The DHB trajectory decoder is producing a meaningful EE plan, so the StarVLA->`embodik` route remains the better path for validating DHB behavior.

## Recommended next step

Prioritize:

1. StarVLA `predict_trajectory()`
2. `embodik` IK
3. `JOINT_POSITION` execution

De-prioritize raw direct LIBERO controller rollout as the main success signal until the direct policy head is better calibrated for delta translation.
