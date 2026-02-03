"""
dhb_xr: DHB Extended Representations for SE(3) invariant trajectory encoding.
"""

__version__ = "0.1.0"

from dhb_xr.encoder.dhb_dr import encode_dhb_dr
from dhb_xr.decoder.dhb_dr import decode_dhb_dr

# Preprocessing
from dhb_xr.preprocessing import (
    TrajectoryPreprocessor,
    ProcessedTrajectory,
    TrajectoryDiagnostics,
    analyze_trajectory,
    detect_reversals,
    detect_zero_motion,
)

# Visualization
try:
    from dhb_xr.visualization.plot import (
        plot_se3_trajectory,
        plot_se3_trajectories,
        plot_invariants,
    )
    _VIS_AVAILABLE = True
except ImportError:
    _VIS_AVAILABLE = False
    plot_se3_trajectory = None
    plot_se3_trajectories = None
    plot_invariants = None

try:
    from dhb_xr import _dhb_xr_cpp
    cpp_version = _dhb_xr_cpp.cpp_version
    _CPP_AVAILABLE = True
except ImportError:
    cpp_version = None
    _CPP_AVAILABLE = False

__all__ = [
    "__version__",
    "encode_dhb_dr",
    "decode_dhb_dr",
    "cpp_version",
    # Preprocessing
    "TrajectoryPreprocessor",
    "ProcessedTrajectory",
    "TrajectoryDiagnostics",
    "analyze_trajectory",
    "detect_reversals",
    "detect_zero_motion",
    # Visualization
    "plot_se3_trajectory",
    "plot_se3_trajectories",
    "plot_invariants",
]
