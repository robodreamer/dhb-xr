# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-02-05

### Added
- **VLA Integration**: Comprehensive LIBERO and RoboCASA support for Vision-Language-Action benchmarks
- **LIBERO-PRO Support**: Full integration with LIBERO-PRO perturbation robustness testing
- **Trajectory Adaptation**: Fatrop and CasADi solvers for real-time trajectory retargeting (~7ms)
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