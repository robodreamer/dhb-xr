"""VLA framework integrations."""

from dhb_xr.integration.vla.robocasa import RoboCASAAdapter
from dhb_xr.integration.vla.libero import LiberoAdapter
from dhb_xr.integration.vla.pipeline import DHBVLAPipeline

try:
    from dhb_xr.integration.vla.action_head import DHBActionHead
    _HAS_ACTION_HEAD = True
except ImportError:
    _HAS_ACTION_HEAD = False

__all__ = [
    "RoboCASAAdapter",
    "LiberoAdapter",
    "DHBVLAPipeline",
]

if _HAS_ACTION_HEAD:
    __all__.append("DHBActionHead")
