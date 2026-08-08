# Primitive API

Primitive APIs live under `dhb_xr.primitives`.

## Contracts

- `MotionPath`: finite timestamped SE(3) path using `wxyz` quaternions.
- `PrimitiveCandidate`: unapproved geometry, end-effector actions, guards,
  reference frames, observation epoch, evidence, and metadata.
- `PrimitiveProgram` and `PrimitiveTransition`: validated guarded graph.
- `PrimitiveEvidence`: `planned`, `sim_verified`, or `real_verified`
  provenance with source and compatibility metadata.
- `EndEffectorCommand` and `ScheduledEndEffectorAction`: portable tool commands
  synchronized before, during, after, or on an event.

All contracts provide JSON-compatible `to_dict` and `from_dict` methods. Invalid
quaternions, timestamps, graph edges, unreachable nodes, unsupported metadata,
and missing command parameters fail closed.

## Generation and sources

- `minimum_jerk_path`
- `affordance_normal_path`
- `arc_path`
- `screw_path`
- `via_point_path`
- `AnalyticPrimitiveSource`
- `DHBRetargetSource`
- `BackendPrimitiveSource`

## Execution

- `ExecutionObservation`
- `PrimitiveFeedback`
- `GuardEvaluator`
- `PrimitiveProgramExecutor`
- `ExecutionDecision` and `ExecutionCommand`

## Memory and import

- `PrimitiveMemory`
- `VerificationRecord`
- `DemonstrationTrace` and `MinedPrimitiveSegment`
- `mine_demonstration_segments` and `segment_to_candidate`
- `exact_gripper_run_control_indices`
- `hysteretic_gripper_control_indices`
- `propose_primitive_boundaries`
- `split_motion_at_indices`
- `stitch_motion_paths`

## Portfolio evaluation

- `PrimitivePortfolio`
- `ObservationEpochGate`
- `EndpointGate`
- `SphereClearanceGate`
- `PortfolioEvaluation`, `SourceEvaluation`, and `CandidateEvaluation`

See the [primitive-program guide](../primitive_programs.md) for contracts,
authority boundaries, and a runnable example.
