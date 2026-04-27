# DHB-XR Unattended Benchmark Runbook

## 0. Paper Claim Closure Run Order

Run order is now claim-driven:

1. Freeze the continuous `predict_trajectory() -> embodik` evaluator.
2. Run the matched `QwenDHB` and `QwenOFT` comparison through that exact
   evaluator.
3. Save success metrics, IK metrics, object-relative placement offsets,
   smoothness/path-efficiency metrics, runtime/memory metrics, and videos.
4. Only after the matched comparison is stable, run seed-size and augmentation
   studies.
5. Keep discrete-token policy work offline until token prediction and decoded
   endpoint drift pass the gates below.

Do not mix old rollout-specific heuristics into paper numbers. Any evaluator
knob used for one model must be used for the matched control unless the note
explicitly labels the run as diagnostic-only.

### Current Live Reach Diagnostic Contract

For StarVLA live `predict_trajectory() -> embodik` reach diagnostics, use this
contract when validating non-zero offsets:

```bash
PYTHONPATH=vendor/starVLA:src:${PYTHONPATH:-} WANDB_MODE=disabled MUJOCO_GL=egl .pixi-home/bin/pixi run python scripts/starvla_dhb_live_checkpoint_sweep.py \
  --config_yaml configs/libero_dhbxr_qwen_fresh_actionxyz.yaml \
  --run-dir _workspace/starvla_runs/dhb_starvla_qwen_fresh_actionxyz_from2000 \
  --checkpoint-step 400 \
  --task_suite_name libero_spatial --task_id 0 \
  --case 0:0 --case 0:3 --case 0:-3 --case 1:3 --case 1:-3 \
  --rollout-seed 70 \
  --replan_every 4 --max-first-offset-mm 35 --max-step-mm 20 \
  --smooth-window 3 --blend-overlap-alpha 0.6 --blend-mode constant \
  --commit-prefix-steps 0 --adaptive-commit-threshold-mm 0 --adaptive-commit-max-steps 0 \
  --delta-continuation-alpha 0 --delta-continuation-steps 0 \
  --gripper-source demo --init-state-source benchmark --image-orientation raw \
  --trajectory-source direct_action --direct-action-xyz-scale 1.0 1.35 1.15 \
  --direct-action-xyz-bias-mm 0 0 0 --state-history-update-rate policy \
  --max-policy-steps 140 --settle-steps 20 \
  --reach-only --reach-threshold-xy-m 0.05 --reach-threshold-z-m 0.08 --reach-stop-on-success
```

Gate:

- Pass: `5/5` reach successes, `0` IK failures, and no object-specific gripper
  gates in the policy path.
- Always set and report `--rollout-seed` for claim-level live rollouts. The
  live evaluator records both `config_seed` and `rollout_seed`; old single-run
  gripper traces without this field are diagnostics, not final evidence.
- Current pass artifact:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionxyz_from2000/live_reach_step400_calib_y135_z115_5case_benchmark_raw_max140.json`.
- Do not reuse the older demo-init/raw-image artifact as the paper-safe live
  reach protocol. It remains provenance only.

### Current Full-Task Learned-Gripper Diagnostic Contract

For full pick-place diagnostics under non-zero offsets, do not use the rollout
default horizon. The default `--max-policy-steps 0` preserves the nominal demo
length, which can stop the rollout before delayed offset-adapted close/release
events. The rollout scripts now warn and record `demo_length_cap_warning` when
this happens. For current task-0 learned-gripper diagnostics, use an explicit
long horizon:

```bash
PYTHONPATH=vendor/starVLA:src:${PYTHONPATH:-} WANDB_MODE=disabled MUJOCO_GL=egl ./.pixi-home/bin/pixi run python scripts/starvla_dhb_live_checkpoint_sweep.py \
  --config_yaml configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_currentready_objectrelease.yaml \
  --run-dir _workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_currentready_objectrelease_fromtransport200 \
  --checkpoint-step 300 \
  --task_suite_name libero_spatial --task_id 0 --case 1:3 \
  --rollout-seed 70 \
  --replan_every 4 --max-first-offset-mm 35 --max-step-mm 20 \
  --smooth-window 3 --blend-overlap-alpha 0.6 --blend-mode ramp \
  --commit-prefix-steps 0 --gripper-source model \
  --model-gripper-decode-source contact_gripper_state \
  --init-state-source benchmark --image-orientation raw \
  --trajectory-source direct_action --direct-action-xyz-scale 0.85 1.55 1.15 \
  --direct-action-xyz-bias-mm 0 0 0 \
  --closed-direct-action-xyz-scale 0.85 1.55 1.15 \
  --closed-direct-action-xyz-bias-mm 0 0 0 \
  --state-history-update-rate policy --state-gripper-source commanded \
  --disable-gripper-gates \
  --gripper-close-threshold 0.55 --gripper-open-threshold 0.40 \
  --gripper-transition-debounce-steps 2 \
  --max-policy-steps 240 --settle-steps 60
```

Gate:

- Pass for full-task claim: reward success, true `disable_gripper_gates`, close
  affordance pass, release/placement pass, and no severe IK collapse.
- Partial only: learned close/reopen without object gates but final
  bowl-plate XY remains above the scorer gate. Current object-release
  `steps_300` is in this category: close `82`, reopen `153`, minimum bowl XY
  `1.8 mm`, but final target XY `116.1 mm` and `11` IK failures.
- Negative: direct `contact_phase` decode on this branch. It closes/reopens
  too early and is not the active deployment path.

### Frozen Continuous Evaluator Contract

Primary paper path:

1. Generate Stage-1 trajectory chunks with `scripts/starvla_dhb_embodik_stage1.py`
   for `QwenDHB`.
2. Generate matched Stage-1 trajectory chunks with
   `scripts/starvla_oft_embodik_stage1.py` for `QwenOFT`.
3. Pack episode outputs with `scripts/starvla_dhb_embodik_pack.py`.
4. Run shared IK and sim with `_workspace/eval_dhb_head_embodik.py`.

Use `--poses-npz` and `--joints-npz` when running IK/sim so paper runs do not
depend on global `/tmp/dhb_embodik_poses.npz` or `/tmp/dhb_embodik_joints.npz`
scratch files.

Freeze these knobs for the first paper-safe pass:

- task suite: `libero_spatial`
- task ID: `0`
- episodes: `0,1,10`
- offsets: `0 cm`, `3 cm`
- replan: `4`
- first-offset clamp: `35 mm`
- per-step clamp: `20 mm`
- smoothing window: `3`
- blend overlap alpha: `0.6`
- blend mode: `constant`
- commit prefix: `0`
- delta continuation: disabled for paper-success numbers
- gripper source: `demo`, with the same thresholds and debounce for both heads
- downstream evaluator: same IK, same JOINT_POSITION sim, same default-attempt
  reporting

Report high-gain retry separately from default-attempt success. Do not merge
retry-assisted success into the headline table.

Current paper-safe task-0 artifact:

`_workspace/paper_eval/2026-04-25_matched_task0/summary.md`

### Required DHB Training Contract

The next fresh `QwenDHB` runs must enable chunk-local invariant supervision:

```yaml
datasets:
  vla_data:
    state_history_window: 4
    recompute_dhb_invariants_from_actions: true

framework:
  action_model:
    state_history_window: 4
```

Do not use old checkpoints trained on static per-timestep DHB rows for clean
paper claims. Those rows decode with `10-73 mm` error when sliced as action
chunks because DHB-DR frame initialization is chunk-start dependent. The fixed
dataloader recomputes each invariant target from the sampled raw delta-action
window and current state; the reconstruction smoke is now `0.0 mm` on the same
task-0 bases.

## Scope

Run this after the frozen-OAT and token-warmup diagnostics. The old
`short2`/`short3` path is obsolete: those runs fixed a target-shape bug but were
still dominated by collapsed token labels. The current valid path starts from
the standalone `oat_tokenizer_gate1` checkpoint, uses the normalized tokenizer
path for GT labels, and treats embodied discrete rollout as gated on offline
token-policy evidence.

Primary run:

```bash
RUN_DIR=_workspace/starvla_runs/dhb_starvla_qwen_discrete_tokenwarm1
CFG=$RUN_DIR/config.full.yaml
```

Current known artifacts:

- `$RUN_DIR/checkpoints/steps_100_pytorch_model.pt`
- `$RUN_DIR/checkpoints/steps_200_pytorch_model.pt`
- `$RUN_DIR/config.full.yaml`
- `$RUN_DIR/config.full.yaml`
- `$RUN_DIR/summary.jsonl`

Current key checkpoint:

- `$RUN_DIR/checkpoints/steps_1000_pytorch_model.pt`

Longer follow-up run:

```bash
RUN_DIR2=_workspace/starvla_runs/dhb_starvla_qwen_discrete_tokenwarm2
CFG2=$RUN_DIR2/config.full.yaml
```

Current `tokenwarm2` checkpoints:

- `$RUN_DIR2/checkpoints/steps_1000_pytorch_model.pt`
- `$RUN_DIR2/checkpoints/steps_2000_pytorch_model.pt`
- `$RUN_DIR2/checkpoints/steps_3000_pytorch_model.pt`

Current valid controls:

- frozen-OAT collapsed control:
  `_workspace/starvla_runs/dhb_starvla_qwen_discrete_frozen_oat1/checkpoints/steps_300_pytorch_model.pt`
- standalone tokenizer gate:
  `_workspace/starvla_runs/oat_tokenizer_gate1/oat_tokenizer.pt`

## 1. Standalone OAT Tokenizer Gate

Run the tokenizer without the VLA backbone first. This tests whether DHB
invariant chunks can produce non-collapsed discrete tokens before spending GPU
time on StarVLA or embodied rollout.

```bash
OUT=_workspace/starvla_runs/oat_tokenizer_gate1
mkdir -p "$OUT"

WANDB_MODE=disabled ./.pixi-home/bin/pixi run python scripts/starvla_dhb_pretrain_oat_tokenizer.py \
  --config_yaml configs/libero_dhbxr_qwen_discrete_supervised.yaml \
  --dataset_mode train \
  --max_train_chunks 1024 \
  --max_eval_chunks 256 \
  --train_steps 1000 \
  --batch_size 64 \
  --learning_rate 1e-3 \
  --normalize \
  --lambda_register_diversity 0.01 \
  --min_unique_tokens 8 \
  --min_effective_tokens 4 \
  --min_per_position_unique 2 \
  --min_reconstruction_r2 0 \
  --save_json "$OUT/oat_tokenizer_pretrain_eval.json" \
  --checkpoint_out "$OUT/oat_tokenizer.pt"
```

Gate:

- Pass: JSON `gate_passed` is true, `metrics.token_usage.collapsed_sample_fraction <= 0.25`, `metrics.token_usage.unique_token_count >= 8`, `metrics.token_usage.effective_num_tokens >= 4`, every token position has at least `2` unique IDs, and full reconstruction `R2 >= 0`.
- Strong pass: same as pass plus `metrics.token_usage.per_sample_unique_mean >= 2.0`, full reconstruction MAE is stable or improving by the final training log, and prefix `k=8` is no worse than smaller prefixes.
- Fail: all tokens remain ID `7`, `collapsed_sample_fraction > 0.25`, or reconstruction is finite only because the decoder learned an average trajectory independent of token ID.

Stop condition:

- If this standalone gate fails, do not run more StarVLA discrete fine-tuning or embodied discrete rollout. Adjust tokenizer normalization, FSQ range, or anti-collapse loss first.

Current result:

- `_workspace/starvla_runs/oat_tokenizer_gate1/oat_tokenizer_pretrain_eval.json`
  passed this gate with `550` unique token IDs, effective token count `306.13`,
  collapsed sample fraction `0.0`, and reconstruction `R2 = 0.2941`.
- The tokenizer is no longer the blocker. The next unattended task is token
  policy validation and overfit diagnostics, not another standalone-tokenizer
  retry.

Additional codec gate:

```bash
RUN_DIR=_workspace/starvla_runs/dhb_starvla_qwen_discrete_frozen_oat1

WANDB_MODE=disabled ./.pixi-home/bin/pixi run python scripts/starvla_dhb_discrete_offline_sweep.py \
  --config_yaml "$RUN_DIR/config.full.yaml" \
  --checkpoint "$RUN_DIR/checkpoints/steps_300_pytorch_model.pt" \
  --save_json "$RUN_DIR/discrete_offline_sweep_step300_normfix_gtcodec.json" \
  --max_samples 24 \
  --holdout_mod 10 \
  --holdout_fold 0 \
  --token_prefix_ks 1,2,4,8 \
  --include_full \
  --evaluate_gt_codec
```

Current teacher-forced GT-token codec result:

- `gt_codec_inv_mae = 0.055538`
- `gt_codec_pos_mae_m = 0.004207`
- `gt_codec_endpoint_drift_m = 0.016274`

This validates the discrete invariant codec, not direct token-policy execution.

## 2. Confirm Training Completion

```bash
tail -n 20 _workspace/starvla_runs/dhb_starvla_qwen_discrete_supervised_short2/summary.jsonl
find _workspace/starvla_runs/dhb_starvla_qwen_discrete_supervised_short2/checkpoints -maxdepth 1 -type f | sort
```

Gate:

- Pass: `steps_300_pytorch_model.pt` is present, or training clearly stopped intentionally and at least `steps_100_pytorch_model.pt` plus `steps_200_pytorch_model.pt` are present.
- Fail: no new checkpoint beyond the already observed `steps_200_pytorch_model.pt`, missing `$RUN_DIR/config.full.yaml`, or the JSONL tail contains NaN/inf losses or a crash.

## 3. Offline Discrete Token Sweeps

Run this after any token-policy training checkpoint. Use the corrected offline
sweep with normalized GT-token derivation, sequence-collapse metrics, and
teacher-forced GT codec enabled. This command intentionally skips missing
checkpoints.

```bash
RUN_DIR=_workspace/starvla_runs/dhb_starvla_qwen_discrete_tokenwarm1
CFG=$RUN_DIR/config.full.yaml

for ITEM in \
  "step250:$RUN_DIR/checkpoints/steps_250_pytorch_model.pt" \
  "step500:$RUN_DIR/checkpoints/steps_500_pytorch_model.pt" \
  "step750:$RUN_DIR/checkpoints/steps_750_pytorch_model.pt" \
  "step1000:$RUN_DIR/checkpoints/steps_1000_pytorch_model.pt" \
  "final:$RUN_DIR/checkpoints/final_pytorch_model.pt" \
  "final:$RUN_DIR/final_model/pytorch_model.pt"
do
  TAG=${ITEM%%:*}
  CKPT=${ITEM#*:}
  if [ ! -f "$CKPT" ]; then
    echo "skip missing $TAG $CKPT"
    continue
  fi
  WANDB_MODE=disabled ./.pixi-home/bin/pixi run python scripts/starvla_dhb_discrete_offline_sweep.py \
    --config_yaml "$CFG" \
    --checkpoint "$CKPT" \
    --save_json "$RUN_DIR/discrete_offline_sweep_${TAG}.json" \
    --max_samples 24 \
    --holdout_mod 10 \
    --holdout_fold 0 \
    --token_prefix_ks 1,2,4,8 \
    --include_full \
    --evaluate_gt_codec
done
```

Artifacts:

- `$RUN_DIR/discrete_offline_sweep_step100.json`
- `$RUN_DIR/discrete_offline_sweep_step200.json`
- `$RUN_DIR/discrete_offline_sweep_step300.json`, if `steps_300_pytorch_model.pt` exists
- `$RUN_DIR/discrete_offline_sweep_final.json`, if a final checkpoint exists

Each JSON contains:

- `continuous.avg_gt_inv_mae`
- `continuous.avg_token_unique_count`
- `continuous.gt_avg_token_unique_count`
- `continuous.gt_majority_token_accuracy_baseline`
- `continuous.collapsed_sample_fraction`
- `continuous.gt_collapsed_sample_fraction`
- predicted and GT token histograms
- `prefix_aggregates.{1,2,4,8}`
- `full_aggregate`
- `monotonicity.*`

## 4. Continuous Baseline Comparison

Use the frozen-OAT collapsed control and the token-warmup candidate as the first
comparison. The strongest current evidence is the improvement from a fixed
token sequence toward a partially varying token policy.

```bash
BASE=_workspace/starvla_runs/dhb_starvla_qwen_discrete_frozen_oat1/discrete_offline_sweep_step300_normfix_gtcodec.json
RUN_DIR=_workspace/starvla_runs/dhb_starvla_qwen_discrete_tokenwarm1

python3 -c 'import json, pathlib
base=json.loads(pathlib.Path("'"$BASE"'").read_text())
print("baseline", "'"$BASE"'")
print("  continuous", base["continuous"])
print("  full", base["full_aggregate"])
for p in sorted(pathlib.Path("'"$RUN_DIR"'").glob("discrete_offline_sweep_*.json")):
    d=json.loads(p.read_text())
    print(p)
    print("  continuous", d["continuous"])
    print("  full", d["full_aggregate"])
    print("  monotonicity", d["monotonicity"])'
```

Baseline values already observed for the frozen-OAT collapsed control:

- `sample_count = 24`
- `continuous.avg_gt_inv_mae = 0.198139`
- `continuous.unique_token_sequence_count = 1`
- `continuous.repeated_token_sequence_fraction = 0.958333`
- `full_aggregate.cont_disc_pos_mae_m = 0.56722`
- `full_aggregate.endpoint_drift_m = 1.614899`

Current `tokenwarm1/steps_1000` candidate:

- `continuous.avg_gt_inv_mae = 0.103244`
- `continuous.unique_token_sequence_count = 5`
- `continuous.repeated_token_sequence_fraction = 0.791667`
- `continuous.avg_token_accuracy = 0.052083`
- `continuous.gt_majority_token_accuracy_baseline = 0.03125`
- `full_aggregate.cont_disc_pos_mae_m = 0.046055`
- `full_aggregate.endpoint_drift_m = 0.180139`
- `gt_codec_aggregate.gt_codec_pos_mae_m = 0.004207`

Longer token-warmup ranking:

| Checkpoint | Unique token sequences | Pred global IDs | Token accuracy / majority | Full pos MAE | Full endpoint drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| `tokenwarm1/steps_1000` | `5 / 24` | `10` | `0.052083 / 0.03125` | `0.046055 m` | `0.180139 m` |
| `tokenwarm2/steps_1000` | `8 / 24` | `15` | `0.083333 / 0.03125` | `0.123912 m` | `0.479050 m` |
| `tokenwarm2/steps_2000` | `11 / 24` | `20` | `0.072917 / 0.03125` | `0.082015 m` | `0.295894 m` |
| `tokenwarm2/steps_3000` | `9 / 24` | `20` | `0.067708 / 0.03125` | `0.049830 m` | `0.189869 m` |

Checkpoint decision:

- Use `tokenwarm1/steps_1000` as the current executable-discrete offline
  reference because it has the lowest full endpoint drift.
- Use `tokenwarm2/steps_2000` as the token-diversity diagnostic because it has
  the highest unique-sequence count, but do not use it for embodied rollout.
- Use `tokenwarm2/steps_3000` only as a secondary candidate if the goal is to
  inspect whether higher diversity can coexist with near-tokenwarm1 decode
  fidelity.

Offline token gate:

- Pass: choose the best checkpoint only if predicted and GT `collapsed_sample_fraction <= 0.25`, predicted and GT average token unique counts are at least `2.0`, `avg_token_accuracy` beats `gt_majority_token_accuracy_baseline`, `full_aggregate.cont_disc_pos_mae_m <= 0.12`, `full_aggregate.endpoint_drift_m <= 0.40`, and at least two monotonicity flags are `true`.
- Strong pass: same as above plus `continuous.avg_gt_inv_mae <= 0.12`, so the supervised token objective did not materially regress the continuous trajectory head.
- Fail: `collapsed_sample_fraction >= 0.5`, `avg_token_unique_count < 2.0`, full decode is worse than the baseline full drift, or continuous `avg_gt_inv_mae > 0.14`.
- Stop condition: if every available checkpoint fails the offline token gate, do not run embodied smoke; preserve the JSONs and retrain or inspect token supervision.

Current gate result: no token-native checkpoint passes the full gate yet because
trajectory-fidelity and monotonicity remain mixed. The right next step is a
reduced-cardinality/factorized token target or a token-only warmup that freezes
the continuous trajectory branch, not an embodied discrete rollout.

Trainer note: the validated tokenwarm runs used `train_starvla.py`, which
optimizes only `action_loss` and logs component tensors detached. The generic
`train_unified.py` path has been patched so returned metrics such as
`loss_inv`, `loss_oat_token`, and token-accuracy tensors are not accidentally
backpropagated as separate objectives.

Reduced-codebook note:

- `_workspace/starvla_runs/oat_tokenizer_gate_fsq4x4x4x4` uses a 256-way
  codebook with 8 registers. It keeps `R2 = 0.244400`, `218` unique IDs, and
  effective token count `110.861`, but fails because the last register has only
  one unique token.
- `_workspace/starvla_runs/oat_tokenizer_gate_fsq4x4x4x4_k7` uses the same
  256-way codebook with 7 registers. It keeps `R2 = 0.209199`, `220` unique
  IDs, and effective token count `100.434`, but again the final register is
  static.
- Do not train a policy directly on these artifacts as-is. Use them to design a
  trimmed/factorized token target that excludes static tail registers from CE,
  or test a slightly larger codebook that restores per-position diversity.

Reduced-codebook policy follow-up:

- `_workspace/starvla_runs/dhb_starvla_qwen_discrete_fsq4_tokenwarm1`
  completed 1000 steps from `tokenwarm1/steps_1000`.
- The warm-start loader skipped old OAT and token-branch tensors when codebook
  shape changed, preserving the configured 256-way frozen tokenizer.
- Offline sweep:
  `_workspace/starvla_runs/dhb_starvla_qwen_discrete_fsq4_tokenwarm1/discrete_offline_sweep_step1000_normfix_gtcodec.json`
- Regression smoke:
  `_workspace/starvla_runs/dhb_starvla_qwen_discrete_fsq4_tokenwarm1/predict_smoke_step1000_discrete.json`

Reduced policy metrics:

| Metric | Value |
| --- | ---: |
| predicted unique token sequences | `8 / 24` |
| predicted global token IDs | `13` |
| token accuracy / majority | `0.104167 / 0.067708` |
| predicted per-position unique | `[2,3,3,2,2,3,3,2]` |
| full position MAE | `0.097732 m` |
| full endpoint drift | `0.357411 m` |
| GT-token codec position MAE | `0.003903 m` |

This passes the loose offline token gate but not the stronger endpoint-fidelity
comparison against `tokenwarm1/steps_1000`. Use it as the next token-learning
baseline, not as an embodied rollout target unless explicitly running a
diagnostic smoke.

Token-only follow-up:

- `_workspace/starvla_runs/dhb_starvla_qwen_discrete_fsq4_tokenonly1`
  completed 1000 steps from `fsq4_tokenwarm1/steps_1000`.
- Qwen and the continuous trajectory-producing branch were frozen; only the
  token objective was optimized.
- Offline sweep:
  `_workspace/starvla_runs/dhb_starvla_qwen_discrete_fsq4_tokenonly1/discrete_offline_sweep_step1000_normfix_gtcodec.json`
- Regression smoke:
  `_workspace/starvla_runs/dhb_starvla_qwen_discrete_fsq4_tokenonly1/predict_smoke_step1000_discrete.json`

Token-only metrics:

| Metric | Value |
| --- | ---: |
| predicted unique token sequences | `8 / 24` |
| predicted global token IDs | `13` |
| token accuracy / majority | `0.114583 / 0.067708` |
| full position MAE | `0.098059 m` |
| full endpoint drift | `0.358955 m` |
| GT-token codec position MAE | `0.003903 m` |

Conclusion: same-objective token CE is not the missing piece. The next unattended
token run should change target structure with factorized or trimmed token CE
rather than extending this run.

## 5. Token-Branch Overfit Probe

Run this before committing to a long token-policy training campaign when the
offline sweep shows partial collapse. The probe caches Qwen hidden states and
optimizes only the token branch against frozen OAT-derived labels.

```bash
RUN_DIR=_workspace/starvla_runs/dhb_starvla_qwen_discrete_tokenwarm1

WANDB_MODE=disabled ./.pixi-home/bin/pixi run env \
  PYTHONPATH=vendor/starVLA:src \
  STARVLA_USE_DEEPSPEED=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=0 \
  python scripts/starvla_dhb_token_overfit_probe.py \
    --config_yaml "$RUN_DIR/config.full.yaml" \
    --checkpoint "$RUN_DIR/checkpoints/steps_1000_pytorch_model.pt" \
    --num_examples 4 \
    --train_steps 300 \
    --learning_rate 0.0005 \
    --log_interval 50 \
    --save_json "$RUN_DIR/token_overfit_probe_4ex_step300.json"
```

Current result:

- passed
- final token accuracy: `1.0`
- final sequence accuracy: `1.0`
- final token loss: `0.000123`
- predicted global token IDs: `30`
- target global token IDs: `30`

Interpretation: the token branch can memorize diverse OAT labels. Continue
longer token-warmup training; do not redesign the token branch unless held-out
offline metrics saturate.

## 6. Next Small Embodied Smoke

Only run this after the offline token gate passes and the live embodied script
has an explicit discrete-token inference flag. Until that wiring is present,
use `scripts/starvla_dhb_predict_smoke.py` as the regression check and do not
claim closed-loop discrete-token execution.

Set `BEST_STEP` to the passing checkpoint, preferring the lowest full `endpoint_drift_m` among passing JSONs:

```bash
RUN_DIR=_workspace/starvla_runs/dhb_starvla_qwen_discrete_tokenwarm1
BEST_STEP=1000
BEST_CKPT=$RUN_DIR/checkpoints/steps_${BEST_STEP}_pytorch_model.pt

WANDB_MODE=disabled ./.pixi-home/bin/pixi run python scripts/starvla_dhb_predict_smoke.py \
  --config_yaml "$RUN_DIR/config.full.yaml" \
  --checkpoint "$BEST_CKPT" \
  --use-discrete-tokens \
  --emit-oat-tokens \
  --save_json "$RUN_DIR/predict_smoke_step${BEST_STEP}_discrete.json"
```

If live discrete wiring is added, then run a two-case embodied smoke:

```bash
RUN_DIR=_workspace/starvla_runs/dhb_starvla_qwen_discrete_tokenwarm1
BEST_STEP=1000

WANDB_MODE=disabled MUJOCO_GL=egl ./.pixi-home/bin/pixi run python scripts/starvla_dhb_live_checkpoint_sweep.py \
  --config_yaml "$RUN_DIR/config.full.yaml" \
  --run-dir "$RUN_DIR" \
  --task_suite_name libero_spatial \
  --task_id 0 \
  --case 0:0 \
  --case 1:3 \
  --replan_every 4 \
  --max-first-offset-mm 35 \
  --max-step-mm 20 \
  --smooth-window 3 \
  --blend-overlap-alpha 0.6 \
  --blend-mode constant \
  --gripper-source demo \
  --init-state-source demo \
  --settle-steps 60 \
  --output-json "$RUN_DIR/live_checkpoint_sweep/tokenwarm1_small_smoke_summary.json" \
  --output-md "$RUN_DIR/live_checkpoint_sweep/tokenwarm1_small_smoke_summary.md"
```

Note: `starvla_dhb_live_checkpoint_sweep.py` evaluates all checkpoints present in `$RUN_DIR/checkpoints`, not only `BEST_CKPT`. Keep `BEST_CKPT` set for notes and post-hoc interpretation.

Embodied smoke artifacts:

- `$RUN_DIR/live_checkpoint_sweep/tokenwarm1_small_smoke_summary.json`
- `$RUN_DIR/live_checkpoint_sweep/tokenwarm1_small_smoke_summary.md`
- Per-step case summaries under `$RUN_DIR/live_checkpoint_sweep/step_<N>/`

Embodied smoke gate:

- Pass: selected checkpoint has at least one `close_count`, or improves average min bowl distance against the continuous shorttrain summaries without increasing `avg_ik_fail_total` materially.
- Strong pass: selected checkpoint gets at least `1/2` success on the two-case smoke.
- Fail: `avg_ik_fail_total` is high, `avg_hold_fraction` is near the old held-still failure mode, both cases fail to close, or distances regress versus the existing summaries under `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/live_checkpoint_sweep/summary.md`.

## 7. If The Small Smoke Passes

Run the broader embodied sweep only after the two-case smoke passes:

```bash
RUN_DIR=_workspace/starvla_runs/dhb_starvla_qwen_discrete_tokenwarm1

WANDB_MODE=disabled MUJOCO_GL=egl ./.pixi-home/bin/pixi run python scripts/starvla_dhb_live_checkpoint_sweep.py \
  --config_yaml "$RUN_DIR/config.full.yaml" \
  --run-dir "$RUN_DIR" \
  --task_suite_name libero_spatial \
  --task_id 0 \
  --case 0:0 \
  --case 1:0 \
  --case 2:0 \
  --case 0:3 \
  --case 1:3 \
  --case 2:3 \
  --replan_every 4 \
  --max-first-offset-mm 35 \
  --max-step-mm 20 \
  --smooth-window 3 \
  --blend-overlap-alpha 0.6 \
  --blend-mode constant \
  --gripper-source demo \
  --init-state-source demo \
  --settle-steps 60 \
  --output-json "$RUN_DIR/live_checkpoint_sweep/tokenwarm1_full_sweep_summary.json" \
  --output-md "$RUN_DIR/live_checkpoint_sweep/tokenwarm1_full_sweep_summary.md"
```

Full-sweep artifacts:

- `$RUN_DIR/live_checkpoint_sweep/tokenwarm1_full_sweep_summary.json`
- `$RUN_DIR/live_checkpoint_sweep/tokenwarm1_full_sweep_summary.md`
- Per-step case summaries under `$RUN_DIR/live_checkpoint_sweep/step_<N>/`

Report the final result against these existing references:

- Offline token baseline: `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/discrete_offline_sweep_step200.json`
- Continuous live checkpoint baseline: `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/live_checkpoint_sweep/summary.md`
- QwenOFT matched control note: `_workspace/starvla_qwenoft_control_baseline.md`
