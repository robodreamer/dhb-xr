# DHB-XR Discrete Token Collapse Diagnosis

## Current Finding

`short2` and `short3` fixed the old `short1` supervised-token target
shape/mismatch issue. Offline sweeps now run against the corrected contract.

The remaining failure is more basic: GT OAT tokens and predicted OAT tokens both
collapse to all token ID `7`. As a result, `avg_token_accuracy=1.0` is
meaningless because the model is matching a degenerate target.

## Interpretation

This is not evidence that a token-native DHB policy works. It is evidence that
the standalone tokenizer/target path has an anti-collapse bug or missing
validation gate.

The continuous DHB path remains the executable policy story. Discrete invariant
tokens remain a scaling/compression hypothesis pending anti-collapse validation.

## Do Not Do Yet

Do not run embodied discrete rollout until standalone tokenizer gates pass.
Live rollout would only test a known-degenerate token stream.

## Required Gates

- Standalone tokenizer reconstruction beats the mean prior (`R2 >= 0`).
- GT OAT token targets use more than one token ID, with at least `8` global
  unique IDs for the first gate.
- Predicted OAT tokens use more than one token ID and beat the GT majority-token
  baseline.
- Effective token count is at least `4`.
- Each register position has at least `2` unique IDs.
- `avg_token_accuracy` is reported with token histograms.
- Prefix/full decode metrics are interpretable and not dominated by a constant
  token stream.
- Endpoint drift is competitive with the continuous checkpoint before
  `predict_trajectory() -> embodik` rollout.

## Likely Debug Surface

- OAT token binning range or clipping.
- Token target construction after export.
- Batch/sequence dimension handling around token labels.
- Decode assumptions that map valid continuous variation to one token ID.
- Metric reporting that hides collapse behind perfect accuracy.

## Current Implementation Response

- `scripts/starvla_dhb_pretrain_oat_tokenizer.py` pretrains/evaluates the OAT
  tokenizer without the VLA backbone and writes strict gate diagnostics.
- `OATTokenizer` now exposes token-collapse diagnostics and a differentiable
  register-diversity loss.
- `DHBActionHead` exposes `lambda_oat_register_diversity` so future StarVLA
  discrete retrains can apply the same anti-collapse pressure.
- `scripts/starvla_dhb_discrete_offline_sweep.py` now reports both predicted
  and derived-GT token histograms, GT collapse rate, and a majority-token
  baseline for interpreting token accuracy.

## First Positive Gate Result

`_workspace/starvla_runs/oat_tokenizer_gate1` passed the standalone tokenizer
gate:

- reconstruction `R2 = 0.2941`
- `unique_token_count = 550`
- `effective_num_tokens = 306.13`
- `collapsed_sample_fraction = 0.0`

This means DHB invariant chunks can produce diverse OAT tokens when the
tokenizer is trained as its own module with normalization and explicit gates.
The remaining bug is the StarVLA integration contract: labels and decode should
come from this frozen tokenizer source rather than a moving in-model tokenizer
that can self-label collapsed outputs.

## Frozen OAT Integration Update

The StarVLA action head now loads the frozen `oat_tokenizer_gate1` checkpoint
and its normalization stats. The offline sweep was also corrected to derive GT
tokens through the backend normalization path; direct `oat.tokenize(raw_inv)` is
not a valid comparison for normalized checkpoints.

Teacher-forced GT-token reconstruction passes the execution-grade codec check:

- `gt_codec_inv_mae = 0.055538`
- `gt_codec_pos_mae_m = 0.004207`
- `gt_codec_endpoint_drift_m = 0.016274`

The first frozen-OAT supervised run still collapsed cross-sample predictions:

- `dhb_starvla_qwen_discrete_frozen_oat1/steps_300`
- predicted unique token sequences: `1 / 24`
- per-position predicted unique counts: `[1, 1, 1, 1, 1, 1, 1, 1]`
- full endpoint drift: `1.614899 m`

The token-warmup run is the first partial policy-side recovery:

- `dhb_starvla_qwen_discrete_tokenwarm1/steps_1000`
- predicted unique token sequences: `5 / 24`
- per-position predicted unique counts: `[3, 3, 1, 2, 1, 1, 1, 2]`
- token accuracy beats the GT majority baseline: `0.052083 > 0.03125`
- full continuous-vs-discrete position MAE: `0.046055 m`
- full endpoint drift: `0.180139 m`

Remaining diagnosis: the tokenizer is no longer the blocker. The direct token
policy is learning, but the 1000-way FSQ classification problem is still
high-cardinality and partially collapsed. Next work should use offline gates,
token overfit probes, longer training, smaller/structured codebooks, or
per-position/latent-factor losses before making any headline embodied discrete
claim.

## Token-Branch Overfit Probe

`scripts/starvla_dhb_token_overfit_probe.py` now caches a tiny fixed set of
Qwen hidden states and trains only the OAT token branch against frozen
OAT-derived labels. This isolates token-head memorization from VLM inference,
dataloader noise, and embodied rollout.

The 4-example probe from
`dhb_starvla_qwen_discrete_tokenwarm1/steps_1000` passed:

- artifact:
  `_workspace/starvla_runs/dhb_starvla_qwen_discrete_tokenwarm1/token_overfit_probe_4ex_step300.json`
- initial token accuracy: `0.09375`
- initial sequence accuracy: `0.0`
- final token accuracy: `1.0`
- final sequence accuracy: `1.0`
- final token loss: `0.000123`
- predicted unique sequences: `4`
- predicted global token IDs: `30`
- target global token IDs: `30`

Interpretation: the direct token branch can represent and memorize diverse
token labels when the examples are fixed. The remaining failure is therefore
not a hard architectural impossibility in the token decoder. The next justified
step is longer token-warmup training and held-out offline sweeps; if held-out
collapse persists, then the next likely blocker is token-space cardinality or
insufficient data/optimization pressure rather than a decode/normalization bug.

## Longer Token-Warmup Diagnosis

`dhb_starvla_qwen_discrete_tokenwarm2` completed through `steps_3000` and was
swept with the corrected normalized GT-token path.

| Checkpoint | Unique token sequences | Pred global IDs | Token accuracy / majority | Full pos MAE | Full endpoint drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| `tokenwarm1/steps_1000` | `5 / 24` | `10` | `0.052083 / 0.03125` | `0.046055 m` | `0.180139 m` |
| `tokenwarm2/steps_1000` | `8 / 24` | `15` | `0.083333 / 0.03125` | `0.123912 m` | `0.479050 m` |
| `tokenwarm2/steps_2000` | `11 / 24` | `20` | `0.072917 / 0.03125` | `0.082015 m` | `0.295894 m` |
| `tokenwarm2/steps_3000` | `9 / 24` | `20` | `0.067708 / 0.03125` | `0.049830 m` | `0.189869 m` |

Diagnosis:

- Token collapse is no longer the primary failure. The policy emits multiple
  non-collapsed sequences, and token accuracy consistently beats the majority
  baseline.
- The remaining blocker is alignment between token prediction and executable
  trajectory fidelity. Better token diversity does not monotonically reduce
  full decoded endpoint drift.
- The GT side remains much more diverse than the policy side: the held-out GT
  labels use `136` global IDs and `24 / 24` unique token sequences, while the
  best policy checkpoint uses only `20` global IDs and `11 / 24` unique
  sequences.
- The GT-token codec stays strong across all sweeps:
  `0.004207 m` position MAE and `0.016274 m` endpoint drift.

Current conclusion: the discrete-token representation is valid, but the direct
policy target is still too hard or too weakly conditioned for the current data
budget. The next most defensible change is not another embodied rollout or
longer same-objective training; it is a token-learning intervention that reduces
effective label cardinality or isolates token CE optimization from continuous
trajectory regression.

## Reduced-Codebook Probe

The first reduced-cardinality check used a 256-way FSQ codebook
(`fsq_levels=[4,4,4,4]`) instead of the current 1000-way frozen OAT target.

| Probe | Registers | R2 | Unique IDs | Effective IDs | Per-position unique | Gate |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `fsq4x4x4x4` | `8` | `0.244400` | `218` | `110.861` | `[101,101,96,108,58,56,59,1]` | fail |
| `fsq4x4x4x4_k7` | `7` | `0.209199` | `220` | `100.434` | `[118,110,66,82,112,60,1]` | fail |

Diagnosis: this is not global collapse. The smaller codebook keeps broad global
usage and positive reconstruction R2, but the final register becomes static.
That suggests the next target design should be factorized or trimmed by
position, not simply "same OAT objective with fewer global IDs."

## Reduced-Codebook Policy Result

The first reduced-codebook StarVLA token policy run is:

`_workspace/starvla_runs/dhb_starvla_qwen_discrete_fsq4_tokenwarm1`

Offline sweep result at `steps_1000`:

- predicted unique token sequences: `8 / 24`
- predicted global token IDs: `13`
- token accuracy beats majority baseline: `0.104167 > 0.067708`
- predicted per-position unique counts: `[2, 3, 3, 2, 2, 3, 3, 2]`
- full continuous-vs-discrete position MAE: `0.097732 m`
- full endpoint drift: `0.357411 m`
- GT-token codec position MAE: `0.003903 m`
- GT-token codec endpoint drift: `0.015620 m`

Diagnosis update: reducing the token space from 1000-way to 256-way helps the
policy learn non-collapsed token structure. The remaining problem is not token
collapse; it is recovering executable endpoint fidelity. The next reduced-token
run tested this directly by freezing the continuous trajectory branch and
optimizing token CE only:

- `_workspace/starvla_runs/dhb_starvla_qwen_discrete_fsq4_tokenonly1`
- token accuracy improved slightly to `0.114583 > 0.067708`
- unique predicted token sequences stayed at `8 / 24`
- full endpoint drift stayed effectively unchanged at `0.358955 m`
- predict smoke passed:
  `_workspace/starvla_runs/dhb_starvla_qwen_discrete_fsq4_tokenonly1/predict_smoke_step1000_discrete.json`

Updated diagnosis: more same-objective token CE is not enough. The next
intervention should change the target structure itself, most likely by
factorizing token prediction by register or trimming low-information/static
tail registers from the token CE objective.
