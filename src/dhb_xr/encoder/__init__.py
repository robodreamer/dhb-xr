"""DHB encoders: DHB-DR (Euler), DHB-QR (quaternion), DHB-TI (time-invariant), batched PyTorch."""

from dhb_xr.core.types import DHBMethod
from dhb_xr.encoder.dhb_dr import encode_dhb_dr
from dhb_xr.encoder.padding import extrapolate_boundary_poses
from dhb_xr.encoder.dhb_ti import (
    compute_progress,
    resample_by_progress,
    encode_dhb_dr_ti,
    encode_dhb_qr_ti,
)

__all__ = [
    "encode_dhb_dr",
    "encode_dhb_qr",
    "encode_dhb_dr_ti",
    "encode_dhb_qr_ti",
    "compute_progress",
    "resample_by_progress",
    "DHBMethod",
    "extrapolate_boundary_poses",
]

try:
    from dhb_xr.encoder.dhb_qr import encode_dhb_qr
except ImportError:
    encode_dhb_qr = None
