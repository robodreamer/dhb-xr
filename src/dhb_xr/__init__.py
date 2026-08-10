"""DHB Extended Representations for SE(3) trajectory encoding.

Public convenience exports are loaded lazily so dependency-light subpackages
such as :mod:`dhb_xr.primitives` do not import SciPy, Matplotlib, or the native
extension merely because Python initialized the package root.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.5.0"

_LAZY_EXPORTS = {
    "encode_dhb_dr": ("dhb_xr.encoder.dhb_dr", "encode_dhb_dr"),
    "decode_dhb_dr": ("dhb_xr.decoder.dhb_dr", "decode_dhb_dr"),
    "EncodingMethod": ("dhb_xr.core.types", "EncodingMethod"),
    "DHBMethod": ("dhb_xr.core.types", "DHBMethod"),
    "TrajectoryPreprocessor": (
        "dhb_xr.preprocessing",
        "TrajectoryPreprocessor",
    ),
    "ProcessedTrajectory": ("dhb_xr.preprocessing", "ProcessedTrajectory"),
    "TrajectoryDiagnostics": (
        "dhb_xr.preprocessing",
        "TrajectoryDiagnostics",
    ),
    "analyze_trajectory": ("dhb_xr.preprocessing", "analyze_trajectory"),
    "detect_reversals": ("dhb_xr.preprocessing", "detect_reversals"),
    "detect_zero_motion": ("dhb_xr.preprocessing", "detect_zero_motion"),
    "plot_se3_trajectory": (
        "dhb_xr.visualization.plot",
        "plot_se3_trajectory",
    ),
    "plot_se3_trajectories": (
        "dhb_xr.visualization.plot",
        "plot_se3_trajectories",
    ),
    "plot_invariants": ("dhb_xr.visualization.plot", "plot_invariants"),
}
_OPTIONAL_VISUALIZATION_EXPORTS = {
    "plot_se3_trajectory",
    "plot_se3_trajectories",
    "plot_invariants",
}


def __getattr__(name: str) -> Any:
    if name == "cpp_version":
        try:
            value = import_module("dhb_xr._dhb_xr_cpp").cpp_version
        except ImportError:
            value = None
        globals()[name] = value
        return value
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module 'dhb_xr' has no attribute {name!r}") from error
    try:
        value = getattr(import_module(module_name), attribute)
    except ImportError:
        if name not in _OPTIONAL_VISUALIZATION_EXPORTS:
            raise
        value = None
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS, "cpp_version"})


__all__ = [
    "__version__",
    "encode_dhb_dr",
    "decode_dhb_dr",
    "EncodingMethod",
    "DHBMethod",
    "cpp_version",
    "TrajectoryPreprocessor",
    "ProcessedTrajectory",
    "TrajectoryDiagnostics",
    "analyze_trajectory",
    "detect_reversals",
    "detect_zero_motion",
    "plot_se3_trajectory",
    "plot_se3_trajectories",
    "plot_invariants",
]
