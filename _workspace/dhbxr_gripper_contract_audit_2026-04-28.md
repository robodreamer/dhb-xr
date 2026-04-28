# DHB-XR Gripper Contract And Strategy Audit

## Why This Audit Exists

The attached-transport continuation improved center reach but still failed full
pick-place. To avoid another ineffective training loop, this note checks the
failure against community VLA conventions and the latest closed-loop trace.

## Community-Code Sanity Check

OpenVLA and StarVLA both treat the gripper as a special action dimension rather
than as an ordinary normalized Cartesian delta. OpenVLA's LIBERO path predicts a
binary absolute gripper value in `[0, 1]`, normalizes it to `[-1, +1]`, and then
inverts it for LIBERO so execution is `-1 = open`, `+1 = close`. OpenVLA also
marks only the final gripper dimension as absolute and excludes it from the
normal action normalization mask.

VLA Foundry uses the same conceptual separation: action/proprio fields are
declared explicitly, relative pose fields are converted through a reference
pose, and gripper fields are debounced with hysteresis before execution. This
matches our desired direction: model outputs should remain generic, deployment
should not use object-distance gates, and gripper conversion should be a clean
environment contract.

TRI-LBM is not directly a LIBERO gripper fix, but its action-type-conditioned
normalizer is a useful reference for our longer-term token story. It suggests
that phase/type-specific action statistics can prevent multimodal chunks from
being collapsed into an average mode. For DHB-XR this maps naturally to
reach/grasp/transport/release token phases, but it is not the immediate blocker.

References:

- OpenVLA LIBERO eval:
  https://github.com/openvla/openvla/blob/main/experiments/robot/libero/run_libero_eval.py
- OpenVLA gripper utilities:
  https://github.com/openvla/openvla/blob/main/experiments/robot/robot_utils.py
- OpenVLA action-normalization contract:
  https://github.com/openvla/openvla/blob/main/prismatic/vla/datasets/rlds/oxe/materialize.py
- VLA Foundry gripper debounce and field adapter:
  https://github.com/TRI-ML/vla_foundry/blob/main/vla_foundry/inference/robotics/data_adapter.py
- TRI-LBM action classifier/normalizer:
  https://github.com/lucidrains/TRI-LBM

## Current Trace Findings

Best attached-transport continuation trace:

- Checkpoint:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_attached_transport_policytrace_guarded_fromclosedtransport300/checkpoints/steps_100_pytorch_model.pt`
- Trace:
  `_workspace/paper_eval/2026-04-28_best_step100_gripper_contract_trace_sweepmatch.json`
- Demo geometry probe:
  `_workspace/paper_eval/2026-04-28_demo_grasp_geometry_probe_task0_first3.json`

The gripper sign and actuator contract are not the bug. At close step `99`, the
policy commands `+1.0`; observed gripper qpos falls from about `0.0341` to
`0.0069` by step `101`, which confirms physical close is happening.

The actual failure is grasp-affordance geometry. The rollout closes at
EE-minus-bowl roughly `[-8.8, -15.3, +29.2] mm`, which is near the bowl center
in XY. Successful demos close from a side-grasp pose: the first three demos
show physical close mean around `[-13.6, -45.1, +26.7] mm`, and the earlier
50-demo probe reported the same side-affordance pattern. After the rollout
closes near center, the bowl is pushed/slid until EE-bowl XY is about `46 mm`,
but it is not securely grasp-attached.

This explains the apparent contradiction in previous metrics. The current run
can report excellent minimum bowl-center XY (`13.3 mm`) while still failing the
real grasp. Bowl-center proximity is therefore the wrong primary full-task
metric for this task.

## Updated Decision

Do not continue tuning attached post-contact transport from the same branch
until the side-grasp contact gate is passed. The attached replay positives are
useful for the big augmentation story, but they assume a grasp-attached state.
The live policy has not yet produced that state reliably.

The next breakthrough target is side-affordance grasp acquisition:

1. Train/evaluate the pre-close and close transition against demo
   EE-minus-object affordance vectors, not bowl-center distance.
2. Require a pass gate that includes close-vector error, observed closed qpos,
   post-close object/EE coupling, and object-target progress.
3. Only after physical coupling is reliable should attached-object transport
   positives be used to train target-reaching transport and release.
4. Keep rollout deployment object-free. Simulator geometry is valid for offline
   labels, probes, and scoring only.

## Claim Impact

This audit prevents an ineffective loop: the problem is not more epochs on the
current attached-positive branch, not gripper polarity, and not another scalar
open/close threshold sweep. The manuscript story should say that DHB-XR now has
the infrastructure for simulator-assisted augmented transport tokens, but the
current embodied gap is side-affordance physical grasp formation before those
transport tokens become meaningful in live closed-loop rollout.
