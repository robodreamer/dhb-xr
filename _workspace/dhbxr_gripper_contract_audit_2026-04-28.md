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
- Affordance score:
  `_workspace/paper_eval/2026-04-28_best_step100_gripper_contract_affordance_score.json`

The gripper sign and actuator contract are not the bug. At close step `99`, the
policy commands `+1.0`; observed gripper qpos falls from about `0.0341` to
`0.0069` by step `101`, which confirms physical close is happening.

The command-close event is not the same as physical close. The first close
command happens near the bowl center: EE-minus-bowl is roughly
`[-8.8, -15.3, +29.2] mm`. However, by the time the gripper is physically
closed, the bowl has moved into a side-grasp geometry:
`[-9.0, -44.5, +26.6] mm`. The affordance scorer reports only `4.6 mm` error
to the first-three-demo physical close mean of `[-13.6, -45.1, +26.7] mm`.

This corrects the failure diagnosis. The current run can form a plausible
physical side-grasp, but it does not turn that grasp into target-directed
transport. After close, best bowl-plate XY remains about `125 mm`, final
bowl-plate XY is about `130 mm`, and there is no reopen.

This explains the apparent contradiction in previous metrics. Minimum
bowl-center XY (`13.3 mm`) is not sufficient, but the active blocker is now
post-close target transport and release after a physically plausible close, not
gripper polarity or side-affordance discovery.

## Updated Decision

Do not continue tuning attached post-contact transport from the same branch
without changing the transport supervision. The branch already passes the
side-grasp close-affordance gate in the reproduced trace, but its learned
post-close motion does not reduce bowl-plate XY enough.

The next breakthrough target is target-directed held-object transport:

1. Keep scoring the close transition against demo EE-minus-object affordance
   vectors, not bowl-center distance, because command-close and physical-close
   can differ.
2. Require a pass gate that includes physical close-vector error, observed
   closed qpos, post-close object/EE coupling, object-target progress, and
   release proximity.
3. Replace the current 3-D attached transport target with stronger
   held-object/target conditioning, such as object-to-EE plus target-to-EE
   features or phase-conditioned transport tokens.
4. Keep rollout deployment object-free. Simulator geometry is valid for offline
   labels, probes, and scoring only.

## Claim Impact

This audit prevents an ineffective loop: the problem is not gripper polarity,
not another scalar open/close threshold sweep, and not simply "get closer to the
bowl center." The manuscript story should say that DHB-XR now has the
infrastructure for simulator-assisted augmented transport tokens, and the
current embodied gap is target-directed transport/release after a physically
plausible side-grasp close.
