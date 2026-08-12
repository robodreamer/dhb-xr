# Development notes

This page keeps active, incompletely validated work separate from the package's
release-facing claims. The features below may be useful for research and API
review, but they are **not validated simulator or robot capabilities**.

## Experimental feature status

| Work in progress | Evidence currently available | Evidence still required |
| --- | --- | --- |
| Skill capsules and demonstration ingestion | Contract, serialization, and round-trip tests | Dataset-scale curation and retrieval study |
| Primitive programs and recovery graphs | Unit tests and a deterministic synthetic state-machine example | Closed-loop simulation and real-robot task trials |
| Analytic and learned candidate portfolios | Isolated candidate and common-gate tests | Matched robot-feasibility and task-outcome evaluation |
| Persistent relational FATROP retargeting | Solver tests and exploratory geometric fixtures | Frozen confirmatory protocol with full orientation and robot constraints |
| Gripper and event-based segmentation | Sample-exact and hysteretic change-point tests | Annotated demonstrations and semantic boundary evaluation |

See the [primitive-program guide](primitive_programs.md),
[provisional API inventory](api/primitives.md), and
[experimental validation notes](release_validation.md) for implementation
details and reproducible component checks.

## VLA evaluation hypothesis

Initial-frame relative pose chunks are already invariant under one shared
world-frame transform. DHB-XR's research hypothesis is not that DHB is *more
frame-invariant*. It is that a structured motion representation may provide a
useful common interface for retargeting, retrieval, time normalization,
tokenization, and later execution stages.

The release currently verifies numerical frame stability and software
interfaces. A matched learned-policy study is still required before claiming
that a DHB action head improves VLA task success, generalization, token rate, or
data efficiency over an initial-frame relative-pose head.

## Promotion rule

Move a development item into release highlights only when its claim is backed
at the corresponding evidence level:

1. Unit tests establish software behavior.
2. Isolated examples establish bounded component behavior.
3. Frozen matched benchmarks can support representation comparisons.
4. Closed-loop simulation can support simulator-task claims.
5. Repeated real-robot trials are required for physical-task claims.

Results at one level must not be described as evidence for a higher level.
