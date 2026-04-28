# DHB-XR Paper Result Pack 2026-04-28

## Purpose

This note is the current paper-facing source of truth. It consolidates the
validated evidence, the safe claims, the artifact paths, and the remaining gap
for the DHB-XR story. It should be preferred over older exploratory notes when
writing the draft, captions, or project-page text.

The current result is strong enough for a clean prototype story, but not for a
claim that full pick-place is solved. The safest paper arc is:

1. DHB-XR provides a geometric action interface for low-data VLA adaptation.
2. Continuous DHB action execution now gives stable, smooth, non-zero-offset
   reach behavior under the corrected closed-loop protocol.
3. Under a matched short-budget StarVLA/Qwen setup, DHB produces a more useful
   structured trajectory signal than the regular QwenOFT action head, with an
   explicit IK-quality caveat.
4. Learned contact and release tokens are mechanically trainable, and the
   best contact rollout reaches the demonstrated side-grasp geometry, but the
   current full task still fails at held-object lift/carry/lower/release.
5. Discrete invariant tokens are promising as compression/analysis tokens, but
   continuous DHB remains the validated streaming-control path today.

## Claim Status

| Claim | Current status | Evidence | Paper wording |
| --- | --- | --- | --- |
| Continuous DHB action can run in a closed-loop VLA rollout without the old rotation snap or arm-freeze failures. | Supported. | Corrected direct-action reach sweep gets `5/5` reach successes and `0` IK failures. | Safe to state as an embodied prototype result. |
| DHB action improves spatial offset reaching under the current evaluation contract. | Supported for reach-only task-0 diagnostics. | The canonical sweep covers `0,+3,-3 cm` offsets and succeeds on all five episode/offset cases. | Safe if scoped to reach/adaptation, not full pick-place. |
| DHB is better than a regular action head. | Partially supported. | Matched task-0 paper-safe comparison: `QwenDHB` gets `1/3` at `0 cm` and `1/3` at `3 cm`; matched `QwenOFT` gets `0/3` and `0/3`. | Say "bounded prototype advantage under this evaluator"; include DHB's `140` IK failures versus QwenOFT's `0`. |
| DHB contact/gripper tokens can replace demo-time replay. | Partially supported. | Learned side-grasp close reaches `9.2 mm` physical close-vector error in a `+3 cm` rollout; release/contact-phase heads learn offline. | Safe as "mechanically trainable"; not safe as solved integrated manipulation. |
| Integrated full pick-place is solved. | Not supported. | Best transport branches close and move the bowl, but fail placement/release. Object-token/adaptor branches are negative. | Present as limitation and next-step target. |
| Discrete invariant tokens are ready for streaming policy control. | Not supported yet. | Offline discrete sweep shows collapsed token use and non-monotonic reconstruction. | Use as a future compression/interface direction, not the current execution path. |

## Canonical Results

### Corrected Reach And Offset Adaptation

Canonical artifact:

`_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionxyz_from2000/live_reach_step400_calib_y135_z115_5case_benchmark_raw_max140.json`

Protocol:

- Task suite: `libero_spatial`, task `0`
- Checkpoint: `dhb_starvla_qwen_fresh_actionxyz_from2000/checkpoints/steps_400_pytorch_model.pt`
- Execution: `--trajectory-source direct_action`
- Calibration: `--direct-action-xyz-scale 1.0 1.35 1.15`
- Init/images: benchmark init, raw LIBERO images
- Horizon: `--max-policy-steps 140`

Result:

| Case | Success step | Min bowl XY | Min bowl XYZ |
| --- | ---: | ---: | ---: |
| Episode `0`, `0 cm` | `80` | `46.3 mm` | `92.0 mm` |
| Episode `0`, `+3 cm` | `91` | `47.8 mm` | `70.9 mm` |
| Episode `0`, `-3 cm` | `80` | `23.1 mm` | `82.9 mm` |
| Episode `1`, `+3 cm` | `96` | `48.5 mm` | `61.9 mm` |
| Episode `1`, `-3 cm` | `79` | `30.3 mm` | `85.4 mm` |

Aggregate: `5/5` reach success, average minimum bowl XY `39.2 mm`, average
minimum bowl XYZ `78.6 mm`, and `0` IK failures.

Representative videos:

- `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionxyz_from2000/reach_step400_calib_y135_z115_ep0_offset-3.mp4`
- `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionxyz_from2000/reach_step400_calib_y135_z115_ep0_offset3.mp4`
- `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionxyz_from2000/reach_step400_calib_y135_z115_ep1_offset3.mp4`

Draft claim:

> Under the corrected benchmark-init closed-loop protocol, the continuous
> DHB-XR action head reaches the task object across `0 cm`, `+3 cm`, and
> `-3 cm` spatial offsets with `5/5` reach success and no IK failures. This
> separates the remaining manipulation failure from basic spatial adaptation,
> rotation handling, and arm-motion stability.

### Matched QwenDHB Versus QwenOFT Control

Canonical note:

`_workspace/paper_eval/2026-04-25_matched_task0/summary.md`

Protocol:

- Models: `QwenDHB` and matched `QwenOFT`
- Base VLM: `Qwen2.5-VL-3B-Instruct`
- Training budget: short-budget `steps_200` checkpoints
- Episodes: `0,1,10`
- Offsets: `0 cm`, `3 cm`
- Execution: shared `predict_trajectory() -> embodik -> LIBERO JOINT_POSITION`

Result:

| Model | `0 cm` final success | `3 cm` final success | IK failures |
| --- | ---: | ---: | ---: |
| `QwenDHB` | `1/3` | `1/3` | `140` |
| `QwenOFT` | `0/3` | `0/3` | `0` |

Interpretation:

The result supports a bounded prototype task-outcome advantage for DHB under a
matched short-budget evaluator. It does not support a blanket claim that DHB is
better on execution quality, because the regular-head control has cleaner IK.

Draft claim:

> In a matched short-budget StarVLA/Qwen comparison on task 0, continuous DHB
> obtains one final task success at both tested offsets, while the matched
> regular QwenOFT action head obtains none. The comparison is intentionally
> scoped: DHB improves the structured trajectory signal and final task outcome
> in this prototype setting, but late-trajectory IK feasibility remains an
> explicit execution-quality gap.

### Learned Contact And Gripper Timing

Canonical artifacts:

- `_workspace/paper_eval/2026-04-27_bestposv2_thr050_ep1_offset3_video_physical_score.json`
- `_workspace/paper_eval/2026-04-27_bestposv2_closedtransport_actiononly_physical_scores.json`
- `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_closecontact_sidegrasp_bestposv2_fromrolloutpos400/bestposv2_thr050_ep1_offset3.mp4`

Best contact result:

- Offset case: episode `1`, `+3 cm`
- Command close: step `99`
- Physical close: step `101`
- Demonstrated side-grasp reference: mean EE-minus-bowl
  `[-1.5, -49.9, +26.3] mm`
- Physical close vector: `[-9.3, -45.3, +25.1] mm`
- Physical close-vector error: `9.2 mm`
- Outcome: no final task success; transport/release still fail

Closed-transport action-only follow-up:

- Physical close-vector error remains strong at `10.6 mm`
- IK failures drop to `0`
- Outcome still has no final placement/release

Interpretation:

This is the clearest evidence that learned gripper/contact control is not
merely replaying demo time. The policy can reach the side-grasp affordance
under a non-zero offset. The blocker is what happens after the object should be
held: the current action/token targets do not reliably lift, carry, lower, and
release the bowl at the plate.

Draft claim:

> Replacing fixed demo-time gripper replay with learned contact supervision
> substantially improves event alignment: in the best non-zero-offset rollout,
> the physical close occurs within `9.2 mm` of the demonstrated side-grasp
> affordance. However, this contact success does not yet translate into full
> pick-place success, exposing held-object transport as the current bottleneck.

### Release And Contact-Phase Token Validation

Canonical artifacts:

- `_workspace/paper_eval/2026-04-28_contact_phase_offline_probe_phasehead_releasebal8_step200_original_manifest.json`
- `_workspace/paper_eval/2026-04-28_contactphase3_balanced600_offline_probe_step200_full.json`
- `_workspace/paper_eval/2026-04-28_contactphase3_balanced600_offline_probe_step600_full.json`

Head-only release-balanced calibration on the original manifest:

- Rows evaluated: `99`
- Supervised steps: `769`
- Release steps: `88`
- Target-release open-probability mean: `0.918`
- Release open rate at threshold `0.5`: `94.3%`
- No-event false open rate at threshold `0.5`: `7.9%`
- No-event false open rate at threshold `0.7`: `5.4%`

Three-class contact-phase calibration:

- Step `200`: phase accuracy `60.8%`, release recall `91.8%`, close recall
  `50.9%`, no-event recall `61.9%`; false-open background is still high.
- Step `600`: phase accuracy improves to `72.2%`, but release recall collapses
  to `0.0%`.

Interpretation:

The integrated gripper/contact token is learnable, but current live rollouts
do not reach a release-ready object-target state. Do not spend more time on
threshold tuning until held-object transport reaches the plate.

Draft claim:

> Offline contact-phase probes show that release/open tokens can be learned
> from generic manifest labels without runtime object-distance heuristics. The
> remaining failure is not the absence of a gripper signal; it is that the live
> policy does not bring the held object to a release-ready state.

### Object-Pair Transport Diagnostics

Canonical artifacts:

- `_workspace/paper_eval/2026-04-28_objectpair6_lifttransport_object_token_probe.json`
- `_workspace/paper_eval/2026-04-28_objectpair6_dagger_gated_object_token_probe.json`
- `_workspace/paper_eval/2026-04-28_closedobjdelta6_ep1p3_component_debug_sweep.json`

Object-token offline probe:

| Run | Best checkpoint | Object-minus-target XY MAE |
| --- | ---: | ---: |
| 6-D lifttransport object token | `steps_100` | `32.0 mm` |
| 6-D dagger-gated object token | `steps_150` | `72.6 mm` |

Direct closed object-delta adapter:

| Step | Success | Close | Min bowl XY | IK failures |
| ---: | ---: | ---: | ---: | ---: |
| `50` | `0/1` | `1/1` | `34.3 mm` | `29` |
| `100` | `0/1` | `1/1` | `13.3 mm` | `30` |
| `150` | `0/1` | `1/1` | `25.4 mm` | `38` |

The adapter branch gets close but creates IK-heavy, target-incorrect behavior.
It should be closed. The right next transport attempt is not another free
residual adapter; it is stricter successful post-grasp data with bounded
lift/carry/lower/release targets and a gate that object-token XY MAE is in the
`20-30 mm` range before any live sweep.

### Discrete DHB Token Status

Canonical artifact:

`_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/discrete_offline_sweep_step200.json`

Current offline status:

- Samples: `24`
- Average token unique count: `1.0`
- Collapsed sample fraction: `1.0`
- Full continuous-vs-discrete invariant MAE: `0.135`
- Full position MAE: `186.4 mm`
- Endpoint drift: `673.8 mm`
- Monotonicity checks: all false

Interpretation:

Discrete invariant tokens are not yet validated as the streaming-control path.
They remain useful for the manuscript as a principled compression and language
interface direction, but the current embodied evidence should use continuous
DHB action.

Draft claim:

> Continuous DHB actions are the current deployment path. Discrete invariant
> tokens remain a representation goal for compression, symbolic planning, and
> language-aligned action abstraction, but the present discrete tokenizer is
> not yet accurate enough for closed-loop execution.

## Recommended Paper Figures And Videos

| Asset | Content | Use |
| --- | --- | --- |
| Figure 1 | Teleop/UMI data plus simulated/world-model spatial augmentation feeding DHB-XR invariant actions. | Main motivation and big-picture story. |
| Figure 2 | Qwen VLM observation encoder plus DHB continuous action/contact-token head plus streaming execution loop. | Method. |
| Figure 3 | Corrected reach sweep table and three non-zero-offset videos. | Clean embodied result. |
| Figure 4 | Matched QwenDHB versus QwenOFT task-0 comparison with IK caveat. | Controlled evidence for DHB advantage. |
| Figure 5 | Contact/phase/object-token diagnostics. | Shows what is learned and where full-task failure remains. |
| Supplement | Failure clips for transport/release and object-adapter negative branches. | Makes the limitation credible instead of hidden. |

Project-page video priority:

1. Show `reach_step400_calib_y135_z115_ep1_offset3.mp4` as the clean
   non-zero-offset reach success.
2. Show `bestposv2_thr050_ep1_offset3.mp4` as learned side-grasp contact, with
   caption explicitly saying that final transport/release is not solved.
3. If using QwenOFT videos, place them beside the matched summary table and
   label them as the short-budget regular-head control.
4. Avoid presenting any current transport video as a solved pick-place result.

## Main Story Draft

DHB-XR should be framed as a geometric action interface for scaling VLA data.
Instead of asking a VLA to regress environment-specific absolute or raw delta
actions, DHB-XR exposes an action representation that factors motion geometry
from the current robot/world frame. This makes the same small teleoperated
dataset more reusable under spatial offsets and creates a natural target for
simulation or world-model augmentation: generate spatially diverse observations
and pair them with spatially adapted DHB trajectories.

The present embodied result validates the first layer of that story. A
continuous DHB action head, driven by Qwen visual-language features and the
current proprioceptive state/history, executes stable closed-loop reaching
across non-zero offsets under benchmark initialization. The result is not just
an offline reconstruction metric; it is a real rollout with no IK failures and
successful delayed reaching beyond the original demonstration horizon.

The current full pick-place failure is also informative. Learned contact and
release tokens can be trained from generic gripper/event labels, and the best
non-zero-offset contact rollout reaches the demonstrated side-grasp geometry.
However, the policy does not yet learn robust held-object lift/carry/lower and
release. This identifies the next data/representation gap: post-contact target
transport needs successful reset/replay or simulator-assisted positives and
bounded phase-specific action targets, not runtime object-distance heuristics
or more scalar threshold sweeps.

## Safe Abstract Sentence

> DHB-XR introduces a geometric action interface for VLA policies that improves
> low-data spatial adaptation by expressing robot motion through continuous
> invariant action tokens; in a StarVLA/Qwen prototype, DHB-XR achieves stable
> non-zero-offset closed-loop reaching and a bounded matched-head advantage
> over a regular action head, while exposing held-object transport as the
> remaining barrier to full pick-place.

## Limitations Paragraph

The current experiments are intentionally scoped. The strongest embodied result
is robust offset reaching, not full pick-place. Learned contact and release
tokens are validated offline and partially in rollout, but held-object
transport/release remains unsolved. The matched QwenDHB versus QwenOFT result
shows a prototype task-outcome advantage for DHB under a short-budget
evaluator, but DHB still has late-trajectory IK failures that must be reduced
before claiming broad execution-quality superiority. Discrete DHB tokens remain
a promising compression and planning interface, but continuous DHB is the
validated deployment path in the current system.

## Next Hard-Gated Experiment

Do not start another generic warm-start sweep. The next worthwhile experiment
must satisfy all gates below:

1. Generate successful post-grasp positive data with the bowl attached or
   genuinely coupled, including lift, carry, lower, and release-tail segments.
2. Keep contact phase separate from continuous object-pair geometry.
3. Use bounded phase/action-type-specific targets rather than free residual
   object-delta adapters.
4. Require object-token or object-target XY MAE near `20-30 mm` on supervised
   post-contact rows before running live policy rollouts.
5. Preserve the corrected reach gate: `5/5` reach success, average bowl XY near
   `38-40 mm`, and `0` IK failures.
6. Run a matched regular-head control before making any transport-superiority
   claim.

## Stop List

Do not spend more unattended time on:

- Runtime object-distance gripper gates, except as diagnostics or offline label
  generation.
- More residual scale sweeps on the same attached-transport manifest.
- More contact threshold/class-weight sweeps without a transport-to-target
  improvement.
- Free closed object-delta adapters when the object token is still above
  `30 mm` XY MAE.
- Presenting the current full-task videos as successful pick-place.
