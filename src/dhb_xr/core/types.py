"""Core data types for SE(3) trajectories and DHB invariants."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Literal, Dict, Any

import numpy as np


class DHBMethod(Enum):
    """DHB encoding method."""

    ORIGINAL = "original"  # 3 invariants per component (magnitude + 2 angles)
    DOUBLE_REFLECTION = "double_reflection"  # 4 invariants (magnitude + Euler XYZ)
    QUATERNION = "quaternion"  # 5 invariants (magnitude + unit quaternion)


class EncodingMethod(Enum):
    """Encoding method for initial frame computation."""

    POSITION = "pos"  # Position-based encoding: use initial position for frame origin
    VELOCITY = "vel"  # Velocity-based encoding: use first position difference for frame origin


@dataclass
class SE3Pose:
    """Single SE(3) pose: position and quaternion (w, x, y, z)."""

    position: np.ndarray  # (3,)
    quaternion: np.ndarray  # (4,) wxyz scalar-first

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64).reshape(3)
        self.quaternion = np.asarray(self.quaternion, dtype=np.float64).reshape(4)
        nq = np.linalg.norm(self.quaternion)
        if nq > 1e-10:
            self.quaternion = self.quaternion / nq

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {"position": self.position.copy(), "quaternion": self.quaternion.copy()}

    @classmethod
    def from_dict(cls, d: Dict[str, np.ndarray]) -> SE3Pose:
        return cls(position=d["position"], quaternion=d["quaternion"])


@dataclass
class Trajectory:
    """SE(3) trajectory: sequence of poses and optional timestamps."""

    poses: List[SE3Pose]
    timestamps: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.poses)

    @property
    def positions(self) -> np.ndarray:
        """(N, 3) position array."""
        return np.array([p.position for p in self.poses])

    @property
    def quaternions(self) -> np.ndarray:
        """(N, 4) quaternion array (wxyz)."""
        return np.array([p.quaternion for p in self.poses])

    @classmethod
    def from_arrays(
        cls,
        positions: np.ndarray,
        quaternions: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> Trajectory:
        """Build from (N,3) positions and (N,4) quaternions (wxyz)."""
        n = len(positions)
        assert len(quaternions) == n
        poses = [
            SE3Pose(position=positions[i], quaternion=quaternions[i])
            for i in range(n)
        ]
        return cls(poses=poses, timestamps=timestamps)


@dataclass
class InvariantSequence:
    """DHB invariant sequence (linear + angular)."""

    linear: np.ndarray  # (N-1, k) k=3 original, k=4 DR, k=5 QR
    angular: np.ndarray  # (N-1, k)
    method: Literal["dhb_dr", "dhb_qr", "original"]
    initial_pose: Dict[str, np.ndarray]
    linear_frame_initial: Optional[np.ndarray] = None  # (4,4) if available
    angular_frame_initial: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.linear = np.asarray(self.linear, dtype=np.float64)
        self.angular = np.asarray(self.angular, dtype=np.float64)
        assert self.linear.shape[0] == self.angular.shape[0]
        assert self.linear.shape[1] == self.angular.shape[1]

    @property
    def length(self) -> int:
        return self.linear.shape[0]

    @property
    def invariant_dim(self) -> int:
        return self.linear.shape[1]

    def concatenated(self) -> np.ndarray:
        """(N-1, 2*k) linear and angular stacked."""
        return np.concatenate([self.linear, self.angular], axis=1)
