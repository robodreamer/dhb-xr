# Primitive API

Primitive APIs live under `dhb_xr.primitives`. The package facade is lazy:
importing contracts or the executor does not import CasADi, Rockit, CUDA,
FAISS, or plotting. Names listed here are exported from
`dhb_xr.primitives.__all__`.

!!! warning "Development-preview maturity"

    This namespace has software and isolated component coverage, not simulator
    or robot task validation. Treat it as a provisional API: typed contracts and
    serialized field names should remain compatible, while source and gate
    implementations may change before promotion into validated release features.

!!! info "Source history (provenance)"

    In these APIs, *provenance* means where a motion came from and how it has
    been tracked: source dataset or URI, revision, action convention, license,
    generated artifacts, and verification records.

## Contracts and guards

| API | Purpose |
| --- | --- |
| `MotionPath` | Immutable timestamped SE(3) path with scalar-first `wxyz` quaternions. |
| `PrimitiveCandidate` | Unapproved geometry, end-effector semantics, guards, frames, epoch, evidence, and metadata. |
| `PrimitiveProgram` | Validated immutable graph of candidates and transitions. |
| `PrimitiveTransition` | Guarded success/failure/timeout/cancellation graph edge. |
| `PrimitiveEvidence` | Evidence level plus source history, revision, artifact, dataset, license, compatibility, and metadata. |
| `EndEffectorCommand` | Embodiment-neutral gripper or suction command. |
| `ScheduledEndEffectorAction` | Command timing, trigger, phase, and completion contract. |
| `GuardCondition`, `GuardExpression` | Serializable observation predicates. |
| `program_from_sequence` | Build a linear program from an ordered candidate sequence. |
| `guard_all`, `guard_any` | Construct conjunctive or disjunctive guard expressions. |

Related enums and errors are `ActionTiming`, `Comparison`, `ContactMode`,
`EndEffectorCommandType`, `EvidenceLevel`, `GuardMode`, `PrimitiveFamily`,
`TransitionEvent`, and `PrimitiveContractError`.

All concrete contracts provide JSON-compatible `to_dict` and `from_dict`
methods. Invalid quaternions, timestamps, graph edges, unreachable nodes,
unsupported metadata, and missing command parameters fail closed. Arrays
retained by immutable contracts are copied, normalized where applicable, and
made read-only.

## Requests, generation, and sources

| API | Purpose |
| --- | --- |
| `PrimitiveRequest` | Common grounded request passed to interchangeable proposal sources. |
| `RelationalPoseAnchor` | Scene-grounded intermediate pose at normalized path progress. |
| `PrimitiveSource` | Proposal-source protocol; it grants no safety authority. |
| `AnalyticPrimitiveSource` | Configured minimum-jerk, arc, screw, or via-point source. |
| `BackendPrimitiveSource` | Adapter for an OCP, flow, diffusion, or custom callable. |
| `DHBRetargetSource` | Existing `generate_trajectory` adapter with explicit backend/fallback policy. |
| `EndpointCorrectedReferenceSource` | Smooth raw-SE(3) endpoint-correction control. |
| `RelationalAnchorReferenceSource` | Piecewise raw-SE(3) relational-anchor control. |
| `PersistentRelationalFatropSource` | Adapter for a persistent matched relational retargeter. |
| `candidate_from_backend_result` | Convert a backend mapping into a typed candidate. |
| `motion_path_from_backend_result` | Convert backend positions/quaternions into a `MotionPath`. |

Analytic motion functions are `minimum_jerk_path`, `minimum_jerk_blend`,
`affordance_normal_path`, `arc_path`, `screw_path`, and `via_point_path`.

## Execution

| API | Purpose |
| --- | --- |
| `PrimitiveProgramExecutor` | Deterministic state machine with bounded retry, timeout, reobservation, and stale-epoch behavior. |
| `GuardEvaluator` | Evaluate declared guards against an observation. |
| `ExecutionObservation` | Timestamped observation epoch and signal mapping. |
| `PrimitiveFeedback` | Motion status plus the latest observation. |
| `ExecutionDecision` | Executor status, active node, transition, reason, and emitted commands. |
| `ExecutionCommand` | Typed motion, end-effector, stop, or reobservation command. |

Related enums and errors are `ExecutorStatus`, `ExecutionCommandType`, and
`PrimitiveExecutionError`. The executor coordinates commands; it does not
perform IK, collision certification, actuator mapping, or task-success
verification.

## Memory and demonstration import

| API | Purpose |
| --- | --- |
| `PrimitiveMemory` | Evidence-preserving candidate store and semantic/geometric retrieval facade. |
| `PrimitiveMemoryRecord` | Stored candidate, invariants, interaction signature, source details, and test history. |
| `PrimitiveMemoryMatch` | Retrieval result with geometric distance and semantic penalty. |
| `VerificationRecord` | Protocol-specific pass/fail record and the source of its supporting artifact. |
| `DemonstrationTrace` | Canonical motion, gripper, contact/object, dataset, license, URI, and convention input. |
| `MinedPrimitiveSegment` | Proposed segment that retains source data and boundary reasons. |
| `mine_demonstration_segments` | Propose typed segments from a demonstration trace. |
| `segment_to_candidate` | Promote a segment only after the caller supplies explicit semantics. |

`VerificationOutcome` enumerates verification results. A failed record is
retained and never promotes evidence.

## Segmentation and composition

| API | Purpose |
| --- | --- |
| `BoundaryProposal` | Candidate split index and its reasons. |
| `exact_gripper_run_control_indices` | Exact scalar-change baseline. |
| `hysteretic_gripper_control_indices` | Noise/ramp-tolerant open/closed segmentation. |
| `propose_primitive_boundaries` | Combine gripper, contact, and object-motion cues. |
| `split_motion_at_indices` | Split while sharing each transition waypoint. |
| `stitch_motion_paths` | Rejoin only geometrically compatible pieces. |

These functions propose boundaries; they do not infer executable skill names
or prove semantic correctness.

## Portfolio evaluation and gates

| API | Purpose |
| --- | --- |
| `PrimitivePortfolio` | Evaluate every source using the same ordered gates. |
| `CandidateGate` | Common gate protocol. |
| `ObservationEpochGate` | Reject stale proposals. |
| `EndpointGate` | Check Cartesian start and goal poses. |
| `RelationalAnchorGate` | Check intermediate scene-grounded poses. |
| `SphereClearanceGate` | Dense sampled-TCP sphere clearance. |
| `OrientedPayloadAabbClearanceGate` | Dense oriented payload box versus fixture AABBs. |
| `GateResult` | One accept/reject decision, reason, and metric mapping. |
| `CandidateEvaluation` | Candidate plus timing and ordered gate results. |
| `SourceEvaluation` | Source timing, candidate count, and optional error. |
| `PortfolioEvaluation` | Complete matched-source result. |

The geometry gates are proposal screens, not continuous-time, robot-link, or
contact-dynamics certificates.

See the [primitive-program guide](../primitive_programs.md) for architecture and
authority boundaries, and [release validation](../release_validation.md) for
the evidence matrix and artifact-producing commands.
