"""Motion database: storage, similarity, retrieval."""

__all__ = ["MotionDatabase", "invariant_distance", "soft_dtw_distance"]

from dhb_xr.database.similarity import invariant_distance, soft_dtw_distance

try:
    from dhb_xr.database.motion_db import MotionDatabase
except ImportError:
    MotionDatabase = None
