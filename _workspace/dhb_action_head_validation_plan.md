# DHB Action Head Validation Plan

- [x] Route SmolVLA-era inference through `SmolVLADHBAdapter`
- [x] Add unit regressions for state-forwarding and observation conversion
- [x] Confirm missing historical checkpoints and switch plan to fresh retraining
- [x] Patch `embodik` evaluation to accept fresh checkpoints and emit debug summaries
- [x] Verify trainer output format matches `embodik` evaluator
- [x] Run smoke retrain in Pixi env at `action_horizon=100` and produce `_workspace/dhb_head_e2e_fix_smoke.pt`
- [x] Run `embodik` predict stage on smoke checkpoint with debug JSON
- [x] Confirm intermediate checkpoint no longer shows persistent EE rotation / orientation drift
- [x] Run `embodik` IK stage on smoke predictions
- [x] Run `embodik` sim stage on smoke IK output
- [x] Review debug artifacts for wrong-Y signature, IK failures, and gripper-close position
- [x] Run recorded `embodik` sweep on smoke checkpoint (`0cm`, `3cm`, `3` seeds)
- [x] Run recorded `embodik` sweep on 50-epoch full checkpoint (`0cm`, `3cm`, `3` seeds)
- [x] Decide next iteration: continue e2e training toward the historically used `200` epochs
- [x] Add resume / warm-start support to `_workspace/train_dhb_head_e2e.py`
- [x] Continue e2e retraining from the current full checkpoint to `200` total epochs
- [x] Run recorded `embodik` sweep on the `200`-epoch checkpoint (`0cm`, `3cm`, `3` seeds)
- [x] Compare smoke vs `50ep` vs `200ep` checkpoints on loss and closed-loop success
- [ ] Add trainer logging hooks for future `wandb` integration
- [ ] Decide whether to keep smoke checkpoint as deployment baseline or redesign training / checkpoint selection

## SmolVLA Exit Criteria

Use SmolVLA as a debugging baseline only if all of these remain true:
- [x] State-forwarding regression tests pass
- [x] Persistent EE rotation / runaway orientation is gone
- [x] `embodik` IK remains stable and low-error across demos
- [x] At least one checkpoint is reproducibly executable with recorded videos
- [x] Failures are consistent enough to diagnose per demo

Stop scaling SmolVLA training if all of these are true:
- [x] Longer training improves train loss substantially
- [x] Closed-loop `embodik` success does not improve with that extra training
- [x] Failures remain concentrated in the same demos (`demo_1`, `demo_10`)
- [x] IK is not the bottleneck
- [x] Higher-gain retries do not rescue the failing demos

## Final SmolVLA Diagnostic Pass

- [x] Re-run the `embodik` sim stage with per-demo close/reopen and object-relative diagnostics
- [x] Review `demo_0`, `demo_1`, and `demo_10` videos side by side
- [x] Classify the dominant failure mode for the failing demos: under-reaching in `Y`, too high in `Z`, or gripper timing
- [x] Save that failure signature as the reference baseline for future frameworks

## StarVLA Handoff Gate

Switch the main effort to StarVLA once the final SmolVLA diagnostic pass is complete and the failure signature is clear.

Carry forward:
- [x] Keep `embodik` as the primary evaluation path
- [x] Keep video recording enabled for every validation run
- [x] Keep the same `0cm` / `3cm` / `3`-seed comparison protocol
- [x] Keep per-demo diagnostics in the evaluation pipeline
- [x] Keep `_workspace/dhb_head_e2e_fix_smoke.pt` as the best SmolVLA executable baseline

Do not continue SmolVLA training with `500+` or `1000` epochs unless there is a specific architecture or supervision change to test.
