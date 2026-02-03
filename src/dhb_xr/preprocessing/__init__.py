"""
Preprocessing module for DHB-XR trajectory data.

Provides utilities for preparing trajectories before DHB encoding:
- TrajectoryPreprocessor: Full preprocessing pipeline
- Diagnostics: Trajectory quality analysis
- Cleaning utilities: Smoothing, alignment, zero-motion handling

Example:
    >>> from dhb_xr.preprocessing import TrajectoryPreprocessor, analyze_trajectory
    >>>
    >>> # Analyze trajectory quality
    >>> diagnostics = analyze_trajectory(positions, quaternions)
    >>> print(f"Detected {diagnostics.num_reversals} reversals")
    >>>
    >>> # Preprocess for encoding
    >>> preprocessor = TrajectoryPreprocessor(align_to_x=True, smooth_reversals=True)
    >>> result = preprocessor.process(positions, quaternions)
    >>> clean_positions, clean_quaternions = result.positions, result.quaternions
"""

from .trajectory_cleaner import (
    TrajectoryPreprocessor,
    ProcessedTrajectory,
)
from .diagnostics import (
    TrajectoryDiagnostics,
    analyze_trajectory,
    detect_reversals,
    detect_zero_motion,
)

__all__ = [
    # Main preprocessor
    "TrajectoryPreprocessor",
    "ProcessedTrajectory",
    # Diagnostics
    "TrajectoryDiagnostics",
    "analyze_trajectory",
    "detect_reversals",
    "detect_zero_motion",
]
