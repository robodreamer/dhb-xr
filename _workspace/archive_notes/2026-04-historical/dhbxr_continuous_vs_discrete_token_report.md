# DHB-XR Continuous vs Discrete Token Report

## Purpose

This note separates what the current discrete-token work proves from what it
does not prove. The main goal is to keep the invariant-token scaling hypothesis
useful while avoiding overclaiming from collapsed token targets or predictions.

## Current Status

The discrete-token infrastructure is now complete enough to evaluate:

- streaming DHB token pipeline tests are present
- discrete DHB inference is implemented
- token trajectory decode examples exist
- LIBERO export can include supervised DHB token targets
- held-out offline sweep exists at
  `scripts/starvla_dhb_discrete_offline_sweep.py`

`short2` and `short3` fixed the old `short1` token-target shape/mismatch issue.
However, the corrected offline sweeps still show GT and predicted OAT token
collapse to all ID `7`. Therefore `avg_token_accuracy=1.0` is a degenerate
match and should not be treated as policy quality.

Recent commits supporting this path:

- `5ed05c2` added streaming DHB token pipeline tests
- `a379e76` aligned examples with the streaming token path
- `be647e4` added discrete token trajectory decode examples
- `7fca7c6` added the discrete DHB inference path
- `625bef0` added supervised DHB token export and the offline sweep

## Continuous Checkpoint Discrete Readout

The current evaluated checkpoint is:

`_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/checkpoints/steps_200_pytorch_model.pt`

This checkpoint was trained primarily as a continuous-invariant policy, not a
token-native supervised policy. The held-out offline sweep showed:

| Metric | Result |
| --- | ---: |
| sample count | `24` |
| avg token unique count | `1.0` |
| collapsed sample fraction | `1.0` |
| continuous avg GT invariant MAE | `0.108868` |
| best prefix by position MAE | `k=4` |
| `k=4` continuous-vs-discrete position MAE | `0.101175 m` |
| `k=4` endpoint drift | `0.347557 m` |
| full `k=8` position MAE | `0.186358 m` |
| full `k=8` endpoint drift | `0.673839 m` |

Interpretation: the discrete path runs, but this checkpoint collapses to one
token id per sample. More tokens do not monotonically improve the reconstruction,
and the full-token decode is worse than the `k=4` prefix on position drift.

## Single-Sample Continuous vs Discrete Diagnostic

For the same checkpoint:

| Comparison | Invariant MAE | Position MAE |
| --- | ---: | ---: |
| continuous vs discrete full | `0.1368` | `0.1732 m` |
| continuous vs discrete prefix-4 | `0.1324` | `0.1053 m` |
| discrete full vs prefix-4 | `0.0669` | `0.2170 m` |

The prefix-4 decode is closer to the continuous trajectory than the full
discrete decode on this sample. That is a useful debugging signal, but not a
success result.

## Correct Interpretation

The continuous invariant head is currently the executable policy path. The
discrete-token path is an infrastructure milestone and an analysis tool, not yet
a validated control policy.

The invariant-token story remains strong at the representation level:

- DHB magnitude tokenization can be SE(3)-stable
- magnitude tokens are highly compressible
- OAT-style prefix decoding is conceptually aligned with streaming action
  generation

But the current StarVLA token targets and predictions do not yet realize that
story because OAT tokens collapse to one ID.

## Contaminated Run Exclusion

Treat this run as contaminated and exclude it from claims:

`_workspace/starvla_runs/dhb_starvla_qwen_discrete_supervised_short1`

Reason: the run used malformed token target supervision. Its outputs should not
be used as evidence for or against supervised discrete DHB policies.

The corrected pending run is:

`_workspace/starvla_runs/dhb_starvla_qwen_discrete_supervised_short2`

`short2` and `short3` have the corrected supervised-token contract. They are
diagnostics for token-native behavior, not evidence for it, because both still
collapse to all ID `7`.

## Claim Discipline

Safe now:

- The code path can export, train against, infer, and sweep discrete DHB token
  trajectories.
- A continuous-invariant checkpoint can be probed through the discrete decoder.
- The current continuous checkpoint is not token-native and shows token collapse.
- `short2` and `short3` resolved the old target shape/mismatch bug.

Not safe yet:

- Discrete DHB tokens improve closed-loop success.
- Full-token decoding is better than prefixes.
- The learned policy has discovered diverse invariant token structure.
- `avg_token_accuracy=1.0` means anything when GT and predicted tokens are all
  ID `7`.

## Next Gate

Do not run embodied discrete rollout until standalone tokenizer gates pass.
Minimum gate:

- standalone tokenizer full reconstruction `R2 >= 0`
- GT OAT tokens are not all ID `7`
- predicted OAT tokens are not all ID `7`
- GT and predicted token histograms both beat their majority-token baseline
- global unique token count is at least `8`
- effective token count is at least `4`
- each token position has at least `2` unique IDs
- prefix metrics roughly monotonic or at least not worse at `k=8`
- endpoint drift competitive with the continuous checkpoint
- only then run integrated `predict_trajectory() -> embodik` validation

The next fix should target tokenizer supervision, binning, target construction,
or OAT decode assumptions before spending time on live control.

## 2026-04-24 Standalone Gate Update

A new standalone tokenizer gate exists at:

`scripts/starvla_dhb_pretrain_oat_tokenizer.py`

The script loads DHB invariant chunks without the VLA backbone, supports
normalization and register-diversity loss, and writes reconstruction, prefix,
token histogram, entropy, and collapse metrics. A two-step CPU smoke confirmed
dataset loading and artifact writing. With strict defaults, that smoke fails as
intended because reconstruction `R2` is negative and token diversity is below
the gate. This prevents the earlier failure mode where finite reconstruction or
perfect collapsed-token accuracy looked like success.

A substantive standalone run passed the gate:

`_workspace/starvla_runs/oat_tokenizer_gate1/oat_tokenizer_pretrain_eval.json`

Key metrics:

- reconstruction `R2 = 0.2941`
- reconstruction `MAE = 0.06094`
- `unique_token_count = 550`
- `effective_num_tokens = 306.13`
- `collapsed_sample_fraction = 0.0`
- per-position unique counts:
  `[175, 169, 141, 155, 141, 130, 131, 16]`

This is a positive discrete-token result, but it is not yet a policy result.
The next required step is to make StarVLA use this frozen tokenizer source for
labels and decoding, then rerun the offline supervised-token sweep.

## 2026-04-24 Frozen OAT Policy Update

The frozen tokenizer integration is now validated, and the old offline sweep
normalization bug is fixed. GT token labels are derived through the same
normalization path used during training.

Teacher-forced GT-token reconstruction is strong:

| Metric | Value |
| --- | ---: |
| GT-token invariant MAE | `0.055538` |
| GT-token position MAE | `0.004207 m` |
| GT-token endpoint drift | `0.016274 m` |

This supports the representation/codec claim: DHB invariant chunks can be
encoded into discrete OAT tokens and decoded back into executable trajectories
with millimeter-scale position error on held-out demo chunks.

Direct token-policy learning is improved but not yet final. The 300-step
frozen-OAT policy collapsed to one predicted sequence. A new 1000-step token
warmup run with frozen OAT, explicit token telemetry, observation-conditioned
token-query context, and `train_with_discrete_oat_tokens=false` produced:

| Metric | Step 300 frozen OAT | Step 1000 token warmup |
| --- | ---: | ---: |
| predicted unique sequences | `1 / 24` | `5 / 24` |
| predicted repeated-sequence fraction | `0.958333` | `0.791667` |
| token accuracy vs majority | `0.036458 / 0.03125` | `0.052083 / 0.03125` |
| full continuous-vs-discrete position MAE | `0.56722 m` | `0.046055 m` |
| full endpoint drift | `1.614899 m` | `0.180139 m` |

Interpretation: the token-native policy is no longer a total fixed-sequence
failure, and the decoded token trajectory is now close enough to justify more
offline/limited embodied smoke testing. It is still not a success claim: several
token positions remain collapsed, predicted diversity is far below GT diversity,
and exact token accuracy is weak.

## 2026-04-24 Token-Branch Overfit Check

A fixed-set overfit probe was added:

`scripts/starvla_dhb_token_overfit_probe.py`

It caches Qwen hidden states, freezes everything except the OAT token branch,
and trains against frozen OAT-derived token labels. This directly tests whether
the token branch can memorize diverse labels before spending more full VLA
training time.

Result:

`_workspace/starvla_runs/dhb_starvla_qwen_discrete_tokenwarm1/token_overfit_probe_4ex_step300.json`

| Metric | Initial | Final |
| --- | ---: | ---: |
| token CE | `4.794156` | `0.000123` |
| token accuracy | `0.09375` | `1.0` |
| sequence accuracy | `0.0` | `1.0` |
| predicted unique sequences | `3` | `4` |
| predicted global token IDs | `9` | `30` |
| target global token IDs | `30` | `30` |

Interpretation: the token branch is capable of fitting diverse OAT labels, so
the current held-out weakness is not caused by a broken token decoder or
normalization path. Longer token-warmup training is now a justified next
experiment; if it saturates below the offline gate, the likely next move is a
smaller or factorized token space rather than more execution-side tuning.

## 2026-04-24 Longer Token-Warmup Result

The follow-up run is:

`_workspace/starvla_runs/dhb_starvla_qwen_discrete_tokenwarm2`

It continues from `tokenwarm1/steps_1000` for a total budget of 3000 additional
steps with corrected normalized GT-token labels and the same frozen OAT codec.
The completed checkpoint ranking is:

| Checkpoint | Continuous inv MAE | Unique token sequences | Pred global IDs | Token accuracy / majority | Full pos MAE | Full endpoint drift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `frozen_oat1/steps_300` | `0.198139` | `1 / 24` | `n/a` | `0.036458 / 0.03125` | `0.567220 m` | `1.614899 m` |
| `tokenwarm1/steps_1000` | `0.103244` | `5 / 24` | `10` | `0.052083 / 0.03125` | `0.046055 m` | `0.180139 m` |
| `tokenwarm2/steps_1000` | `0.102612` | `8 / 24` | `15` | `0.083333 / 0.03125` | `0.123912 m` | `0.479050 m` |
| `tokenwarm2/steps_2000` | `0.091144` | `11 / 24` | `20` | `0.072917 / 0.03125` | `0.082015 m` | `0.295894 m` |
| `tokenwarm2/steps_3000` | `0.124719` | `9 / 24` | `20` | `0.067708 / 0.03125` | `0.049830 m` | `0.189869 m` |

Interpretation:

- `tokenwarm2` improves predicted token diversity and token accuracy versus the
  original collapsed frozen-OAT policy.
- The best discrete executable trajectory fidelity is still
  `tokenwarm1/steps_1000`, not the longest-trained checkpoint.
- `tokenwarm2/steps_2000` gives the best token diversity and lowest continuous
  invariant MAE, but its full discrete decode drifts too far for embodied
  rollout.
- `tokenwarm2/steps_3000` partly recovers full decode fidelity while preserving
  more token diversity than `tokenwarm1`, but it does not beat the current
  endpoint-drift baseline.

Claim discipline:

- Safe: the GT-token codec is strong, and direct token-policy training now
  produces non-collapsed held-out token sequences.
- Safe: longer token training improves token diversity over the collapsed
  frozen-OAT policy, and the token branch can memorize diverse labels in a
  controlled overfit probe.
- Not safe: the direct discrete-token policy is better than the continuous DHB
  policy or ready for closed-loop task-success claims.

Next structural direction: treat exact 1000-way token classification as too
high-cardinality for the current low-data warmup. The most credible next
discrete-token experiment is a smaller or factorized token target, or a
token-only full-data warmup with the continuous head frozen so token learning
cannot trade away trajectory fidelity.

## 2026-04-24 Reduced-Codebook Probe

Two standalone 256-way OAT probes were run to check whether a smaller token
space can preserve the representation claim before training a new StarVLA token
policy.

| Probe | Registers | Gate | R2 | MAE | Unique IDs | Effective IDs | Per-position unique |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `fsq4x4x4x4` | `8` | fail | `0.244400` | `0.062272` | `218` | `110.861` | `[101,101,96,108,58,56,59,1]` |
| `fsq4x4x4x4_k7` | `7` | fail | `0.209199` | `0.064328` | `220` | `100.434` | `[118,110,66,82,112,60,1]` |

This is a partial positive result: reconstruction and global usage stay healthy
with a smaller 256-way codebook, but the final register becomes static. It
should not replace `oat_tokenizer_gate1` yet. The useful next variant is a
trimmed/factorized policy target that excludes static tail registers from token
CE, or a slightly larger codebook that restores final-register diversity.

## 2026-04-24 Reduced-Codebook Policy Probe

The first 256-way StarVLA policy warmup is:

`_workspace/starvla_runs/dhb_starvla_qwen_discrete_fsq4_tokenwarm1`

It was warm-started from `tokenwarm1/steps_1000`, while preserving the new
configured frozen OAT tokenizer and reinitializing the incompatible token
branch.

| Checkpoint | Inv MAE | Unique seq | Global pred IDs | Token acc / majority | Full pos MAE | Full endpoint drift | GT codec pos MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tokenwarm1/steps_1000` | `0.103244` | `5 / 24` | `10` | `0.052083 / 0.03125` | `0.046055 m` | `0.180139 m` | `0.004207 m` |
| `fsq4_tokenwarm1/steps_1000` | `0.100575` | `8 / 24` | `13` | `0.104167 / 0.067708` | `0.097732 m` | `0.357411 m` | `0.003903 m` |
| `fsq4_tokenonly1/steps_1000` | `0.100575` | `8 / 24` | `13` | `0.114583 / 0.067708` | `0.098059 m` | `0.358955 m` | `0.003903 m` |

Interpretation: reduced tokens substantially improve the token-learning signal
and pass the loose offline token gate, but they still lose to the old
`tokenwarm1` checkpoint on decoded endpoint drift. This is a token-learning
breakthrough, not yet an execution result.

Token-only follow-up: freezing the continuous trajectory branch and optimizing
only token CE slightly improves token accuracy, but endpoint drift is unchanged
to slightly worse. This argues against spending more time on the same CE-only
objective and toward factorized/trimmed token targets that remove
low-information registers from the supervised token sequence.
