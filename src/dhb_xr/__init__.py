"""
dhb_xr: DHB Extended Representations for SE(3) invariant trajectory encoding.
"""

__version__ = "0.1.0"

from dhb_xr.encoder.dhb_dr import encode_dhb_dr
from dhb_xr.decoder.dhb_dr import decode_dhb_dr

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
]
