# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- A strong endpoint-corrected raw-SE(3) reuse control, an orientation-aware
  payload/fixture clearance gate, and an exploratory matched retarget-isolation
  benchmark for identifying where DHB-XR geometry is actually distinctive.
- Typed manipulation primitive contracts spanning SE(3) motion, portable
  end-effector commands, observation guards, evidence, and guarded program graphs.
- Minimum-jerk, affordance-normal, arc, screw, and shared-waypoint analytic paths.
- Deterministic primitive-program execution with bounded retries, reobservation,
  stale-epoch cancellation, and concise typed commands.
- Evidence-preserving primitive memory, p5570-compatible and hysteretic gripper
  segmentation, shared transition-waypoint utilities, and adapters for DHB/OCP
  and learned candidate sources.
- Canonical demonstration-trace ingestion that preserves dataset, license,
  action convention, raw signal provenance, and requires explicit semantics
  before candidate promotion.
- A matched primitive portfolio with common epoch, endpoint, and sampled-TCP
  clearance gates plus a reproducible 20-trial component benchmark.
- A runnable architecture example and primitive-program/API documentation.

### Changed
- CuSADi batch decoding now keeps the translational DHB frame independent from
  tool orientation and accepts explicit per-trajectory initial linear frames.
- Package and primitive exports are lazy, so contract/executor integrations do
  not import SciPy, plotting, or native solver dependencies at startup.

## [0.4.1] - 2026-05-27

### Added
- Fixed-horizon CusADi decode artifacts and generated CUDA source for `50`, `80`, `100`, `150`, and `200` sample trajectories.
- `dhb_xr-build-cusadi-decode` for compiling supported CusADi decode libraries into a dhb_xr-owned cache.
- GPU decode documentation covering install paths, cache behavior, runtime fallback policy, and deployment troubleshooting.

### Changed
- `generate_trajectory(..., backend="auto")` documentation now presents FATROP as the preferred single-trajectory backend with explicit fallback policy.
- The `cusadi` optional extra now depends on CasADi and PyTorch rather than an external editable CusADi checkout.

## [0.4.0] - 2026-05-14

### Added
- Torch DHB-QR decoder path.
- VLA action-head, chunk rollout, SmolVLA adapter, and waypoint controller utilities.
- Focused tests for action-head behavior, chunk rollout continuity, VLA pipeline stages, LIBERO orientation handling, Torch DHB-QR decoding, OAT tokenizer diagnostics, waypoint control, and solver retargeting.

### Changed
- Improved VLA pipeline token streaming and OAT prefix decoding behavior.
- Updated CasADi/Fatrop retargeting paths and timing documentation.
- Standardized VLA dataset adapter outputs on DHB-XR's public `wxyz` quaternion convention.
- Removed unused `dtaidistance` from database extras and included `einops` in the `all` extra.

### Fixed
- LIBERO quaternion/action semantics and rollout orientation handling.
- Fatrop boundary handling and trajectory-derived initial frame usage in retargeting.

## [0.3.0] - 2026-01-30

### Added
- **FAST Tokenizer**: Frequency-space action tokenizer using DCT + BPE compression, no training required (~28x compression on invariant sequences)
- **OAT-style Tokenizer**: Ordered Action Tokenization with Register Encoder, Finite Scalar Quantization (FSQ), and Masked Nested Dropout for anytime prefix decoding
- **FSQ Module**: Finite Scalar Quantization with deterministic rounding -- eliminates VQ codebook collapse by construction
- **Register Encoder**: Transformer-based temporal compression using learnable register tokens with causal-last attention
- **Nested Dropout**: Masked nested dropout module for inducing coarse-to-fine token ordering (Matryoshka-style)
- **OAT Decoder**: Cross-attention Transformer decoder for reconstructing sequences from register token latents
- **Benchmark Suite**: Three benchmark scripts comparing all 7 tokenization methods on rate-distortion, SE(3) stability, and prefix decoding
- **Interactive Notebook**: Comprehensive tutorial notebook (`tutorial_action_tokenization.ipynb`) with head-to-head comparisons and visualizations
- **Manuscript Updates**: Expanded Section 7 with FAST/OAT descriptions, new Appendix with empirical comparison tables

### Fixed
- **SE(3) Invariance**: Fixed invariant token stability under SE(3) transforms by using trajectory-relative initial frames (`init_pose` from trajectory start) instead of fixed global axes. This achieves exact invariance (0% token change, MSE = O(1e-13)) versus the previous approximate invariance
- **BPE Token Comparison**: Improved token stability reporting to account for different-length BPE sequences and separate pre-BPE (DCT integer) from post-BPE token differences

### Changed
- **`einops` dependency**: Added to `[tokenization]` optional dependencies for FSQ implementation

## [0.2.0] - 2026-02-05

### Added
- **VLA Integration**: Comprehensive LIBERO and RoboCASA support for Vision-Language-Action benchmarks
- **LIBERO-PRO Support**: Full integration with LIBERO-PRO perturbation robustness testing
- **Trajectory Adaptation**: Fatrop and CasADi solvers for trajectory retargeting (decode ~5-10ms, Fatrop NLP <1s)
- **Swap Demo**: Compelling demonstration showing DHB-XR vs naive replay under spatial perturbations (6.5cm improvement)
- **Simulation Integration**: Real-time LIBERO environment execution with OpenCV visualization and video recording
- **VLA Documentation**: Extensive documentation explaining how DHB-XR addresses VLA spatial generalization challenges

### Changed
- **Repository URLs**: Updated all links to point to robodreamer organization
- **Documentation Updates**: Added VLA context throughout README and documentation

### Fixed
- **Solver Integration**: Proper Fatrop/CasADi solver selection in trajectory adaptation functions

## [0.1.2] - 2026-02-04

### Added
- **MkDocs Documentation**: Migrated from Sphinx to MkDocs with Material theme
- **GitHub Actions CI/CD**: Automated documentation builds and deployment
- **Encoding Method Enum**: Replaced string literals with type-safe `EncodingMethod` enum
- **GitHub Pages**: Documentation hosting setup

### Changed
- **Project Metadata**: Updated author email and added scikit-learn dependency
- **Copyright Year**: Updated to 2026
- **Pixi Configuration**: Renamed project workspace

## [0.1.1] - 2026-02-03

### Added
- **Trajectory Preprocessing**: Robustness modules for handling reversals and zero-motion segments
- **DHB-Token**: VQ-VAE tokenization pipeline for discrete action representations
- **Generative Models**: Variational Flow Matching (VFM) for multi-modal trajectory generation
- **Motion Database**: Similarity search and retrieval with DTW support

## [0.1.0] - 2026-01-31

### Added
- **Core DHB-XR Implementation**: DHB-DR (Double-Reflection) and DHB-QR (Quaternion-Relative) invariant encoding
- **Time-Invariant Reparameterization**: DHB-TI for speed-independent trajectory representations
- **GPU Acceleration**: PyTorch-based batched operations and CusADi for large-scale optimization
- **Fatrop Integration**: Fast structured OCP solver for trajectory optimization
- **CasADi Support**: General nonlinear optimization for trajectory adaptation
- **Visualization**: Enhanced SE(3) trajectory plotting with matplotlib
- **Jupyter Notebooks**: Comprehensive demo notebook with interactive examples

### Changed
- **Dependencies**: Added CUDA support with PyTorch 2.0+ and optional GPU features
- **Documentation**: Enhanced README with performance benchmarks and usage examples

## [0.0.1] - 2026-01-30

### Added
- **Initial Project Structure**: Basic DHB encoding/decoding functionality
- **Project Configuration**: Pixi environment, GitHub Actions, and development tooling
- **Basic Testing**: Initial test suite setup
- **Documentation**: Sphinx-based documentation framework
