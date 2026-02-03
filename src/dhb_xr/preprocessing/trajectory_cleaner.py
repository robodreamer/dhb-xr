"""
Trajectory preprocessing for DHB encoding.

Provides the TrajectoryPreprocessor class for preparing trajectories:
- Alignment to initial direction (+x)
- Gaussian smoothing to reduce noise/jitter
- Zero-motion segment handling
- 180° reversal smoothing
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp

from .diagnostics import (
    TrajectoryDiagnostics,
    analyze_trajectory,
    detect_reversals,
    detect_zero_motion,
)


@dataclass
class ProcessedTrajectory:
    """Result of trajectory preprocessing."""

    positions: np.ndarray
    quaternions: np.ndarray

    # Original data
    original_positions: np.ndarray = None
    original_quaternions: np.ndarray = None

    # Processing metadata
    was_aligned: bool = False
    alignment_rotation: np.ndarray = None  # 3x3 rotation matrix applied

    was_smoothed: bool = False
    smoothing_sigma: float = 0.0

    # Detected and handled issues
    removed_indices: List[int] = field(default_factory=list)
    interpolated_segments: List[Tuple[int, int]] = field(default_factory=list)

    # Diagnostics before/after
    diagnostics_before: TrajectoryDiagnostics = None
    diagnostics_after: TrajectoryDiagnostics = None

    def get_inverse_alignment(self) -> np.ndarray:
        """Get rotation to undo alignment (for decoding)."""
        if self.alignment_rotation is not None:
            return self.alignment_rotation.T
        return np.eye(3)


class TrajectoryPreprocessor:
    """
    Prepare trajectories for DHB encoding.

    Provides a configurable preprocessing pipeline to handle common
    issues in real-world trajectory data:

    - **align_to_x**: Rotate trajectory so initial motion aligns with +x.
      This ensures use_default_initial_frames=True works correctly.

    - **smooth_reversals**: Apply smoothing near detected 180° reversals.
      Prevents Euler singularities during encoding.

    - **remove_zero_motion**: Remove or interpolate zero-motion segments.
      Prevents undefined tangent frames.

    - **gaussian_sigma**: Overall Gaussian smoothing to reduce sensor noise.

    Example:
        >>> preprocessor = TrajectoryPreprocessor(
        ...     align_to_x=True,
        ...     smooth_reversals=True,
        ...     gaussian_sigma=1.0,
        ... )
        >>> result = preprocessor.process(positions, quaternions)
        >>> # Use result.positions, result.quaternions for encoding
        >>> # After decoding, apply result.get_inverse_alignment() to restore original frame
    """

    def __init__(
        self,
        align_to_x: bool = True,
        smooth_reversals: bool = True,
        remove_zero_motion: bool = True,
        interpolate_zero_motion: bool = True,
        gaussian_sigma: float = 0.0,
        reversal_threshold: float = -0.5,
        zero_motion_threshold: float = 1e-6,
        reversal_smoothing_radius: int = 3,
    ):
        """
        Initialize preprocessor.

        Args:
            align_to_x: Rotate trajectory to align initial direction with +x.
            smooth_reversals: Apply local smoothing near 180° reversals.
            remove_zero_motion: Handle zero-motion segments.
            interpolate_zero_motion: If True, interpolate zero-motion segments.
                If False, remove them entirely.
            gaussian_sigma: Sigma for Gaussian smoothing (0 = no smoothing).
            reversal_threshold: Dot product threshold for reversal detection.
            zero_motion_threshold: Distance threshold for zero motion.
            reversal_smoothing_radius: Number of samples to smooth around reversals.
        """
        self.align_to_x = align_to_x
        self.smooth_reversals = smooth_reversals
        self.remove_zero_motion = remove_zero_motion
        self.interpolate_zero_motion = interpolate_zero_motion
        self.gaussian_sigma = gaussian_sigma
        self.reversal_threshold = reversal_threshold
        self.zero_motion_threshold = zero_motion_threshold
        self.reversal_smoothing_radius = reversal_smoothing_radius

    def process(
        self,
        positions: np.ndarray,
        quaternions: np.ndarray,
        return_diagnostics: bool = True,
    ) -> ProcessedTrajectory:
        """
        Run full preprocessing pipeline.

        Order of operations:
        1. Analyze original trajectory
        2. Handle zero-motion segments (remove or interpolate)
        3. Smooth reversals (if detected)
        4. Apply Gaussian smoothing (if sigma > 0)
        5. Align to +x direction
        6. Analyze final trajectory

        Args:
            positions: (N, 3) position trajectory.
            quaternions: (N, 4) wxyz quaternion trajectory.
            return_diagnostics: Include before/after diagnostics.

        Returns:
            ProcessedTrajectory with cleaned data and metadata.
        """
        positions = np.asarray(positions, dtype=np.float64).copy()
        quaternions = np.asarray(quaternions, dtype=np.float64).copy()

        # Store originals
        original_positions = positions.copy()
        original_quaternions = quaternions.copy()

        # Initialize result
        result = ProcessedTrajectory(
            positions=positions,
            quaternions=quaternions,
            original_positions=original_positions,
            original_quaternions=original_quaternions,
        )

        # Analyze before
        if return_diagnostics:
            result.diagnostics_before = analyze_trajectory(
                positions, quaternions,
                reversal_threshold=self.reversal_threshold,
                zero_motion_threshold=self.zero_motion_threshold,
            )

        # Step 1: Handle zero-motion segments
        if self.remove_zero_motion:
            positions, quaternions, removed, interpolated = self._handle_zero_motion(
                positions, quaternions
            )
            result.removed_indices = removed
            result.interpolated_segments = interpolated

        # Step 2: Smooth reversals
        if self.smooth_reversals:
            reversal_indices, _ = detect_reversals(positions, self.reversal_threshold)
            if reversal_indices:
                positions, quaternions = self._smooth_around_reversals(
                    positions, quaternions, reversal_indices
                )

        # Step 3: Gaussian smoothing
        if self.gaussian_sigma > 0:
            positions, quaternions = self.smooth_trajectory(
                positions, quaternions, self.gaussian_sigma
            )
            result.was_smoothed = True
            result.smoothing_sigma = self.gaussian_sigma

        # Step 4: Align to +x
        if self.align_to_x:
            positions, quaternions, R = self.align_to_initial_direction(
                positions, quaternions
            )
            result.was_aligned = True
            result.alignment_rotation = R

        # Store final
        result.positions = positions
        result.quaternions = quaternions

        # Analyze after
        if return_diagnostics:
            result.diagnostics_after = analyze_trajectory(
                positions, quaternions,
                reversal_threshold=self.reversal_threshold,
                zero_motion_threshold=self.zero_motion_threshold,
            )

        return result

    def _handle_zero_motion(
        self,
        positions: np.ndarray,
        quaternions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[int], List[Tuple[int, int]]]:
        """Handle zero-motion segments by removing or interpolating."""
        segments = detect_zero_motion(positions, self.zero_motion_threshold)

        if not segments:
            return positions, quaternions, [], []

        removed_indices = []
        interpolated_segments = []

        if self.interpolate_zero_motion:
            # Interpolate zero-motion segments
            for start, end in segments:
                positions, quaternions = self._interpolate_segment(
                    positions, quaternions, start, end
                )
                interpolated_segments.append((start, end))
            return positions, quaternions, [], interpolated_segments
        else:
            # Remove zero-motion frames (keep first frame of each segment)
            keep_mask = np.ones(len(positions), dtype=bool)
            for start, end in segments:
                keep_mask[start + 1:end + 1] = False
                removed_indices.extend(range(start + 1, end + 1))

            positions = positions[keep_mask]
            quaternions = quaternions[keep_mask]
            return positions, quaternions, removed_indices, []

    def _interpolate_segment(
        self,
        positions: np.ndarray,
        quaternions: np.ndarray,
        start: int,
        end: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate a zero-motion segment."""
        # Get boundary positions
        p_start = positions[start]
        p_end = positions[min(end + 1, len(positions) - 1)]

        # Linear interpolation for positions
        segment_length = end - start + 1
        if segment_length > 0 and not np.allclose(p_start, p_end):
            for i, idx in enumerate(range(start, end + 1)):
                if idx < len(positions):
                    t = (i + 1) / (segment_length + 1)
                    positions[idx] = (1 - t) * p_start + t * p_end

        # SLERP for quaternions
        q_start = quaternions[start]
        q_end = quaternions[min(end + 1, len(quaternions) - 1)]

        if not np.allclose(q_start, q_end):
            try:
                # Convert to scipy Rotation (expects xyzw)
                r_start = Rotation.from_quat(np.roll(q_start, -1))  # wxyz to xyzw
                r_end = Rotation.from_quat(np.roll(q_end, -1))

                slerp = Slerp([0, 1], Rotation.concatenate([r_start, r_end]))

                for i, idx in enumerate(range(start, end + 1)):
                    if idx < len(quaternions):
                        t = (i + 1) / (segment_length + 1)
                        r_interp = slerp(t)
                        q_xyzw = r_interp.as_quat()
                        quaternions[idx] = np.roll(q_xyzw, 1)  # xyzw to wxyz
            except ValueError:
                pass  # Keep original if SLERP fails

        return positions, quaternions

    def _smooth_around_reversals(
        self,
        positions: np.ndarray,
        quaternions: np.ndarray,
        reversal_indices: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply local smoothing around reversal points."""
        radius = self.reversal_smoothing_radius

        for idx in reversal_indices:
            # Get window around reversal
            start = max(0, idx - radius)
            end = min(len(positions), idx + radius + 1)

            if end - start < 3:
                continue

            # Apply local Gaussian smoothing
            window_positions = positions[start:end].copy()
            for dim in range(3):
                window_positions[:, dim] = gaussian_filter1d(
                    window_positions[:, dim], sigma=1.0, mode='nearest'
                )
            positions[start:end] = window_positions

        return positions, quaternions

    def smooth_trajectory(
        self,
        positions: np.ndarray,
        quaternions: np.ndarray,
        sigma: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Gaussian smoothing to trajectory.

        Args:
            positions: (N, 3) position trajectory.
            quaternions: (N, 4) quaternion trajectory.
            sigma: Gaussian sigma parameter.

        Returns:
            Smoothed (positions, quaternions).
        """
        positions = positions.copy()

        # Smooth positions
        for dim in range(3):
            positions[:, dim] = gaussian_filter1d(
                positions[:, dim], sigma=sigma, mode='nearest'
            )

        # For quaternions, we use a simple moving average approach
        # (proper SLERP-based smoothing is more complex)
        # This preserves the general orientation but reduces jitter
        quaternions = self._smooth_quaternions(quaternions, sigma)

        return positions, quaternions

    def _smooth_quaternions(
        self,
        quaternions: np.ndarray,
        sigma: float,
    ) -> np.ndarray:
        """Smooth quaternion trajectory (simple approach)."""
        quaternions = quaternions.copy()
        n = len(quaternions)

        if n < 3:
            return quaternions

        # Ensure quaternion continuity (flip sign if needed)
        for i in range(1, n):
            if np.dot(quaternions[i], quaternions[i - 1]) < 0:
                quaternions[i] = -quaternions[i]

        # Apply Gaussian filter to each component
        for dim in range(4):
            quaternions[:, dim] = gaussian_filter1d(
                quaternions[:, dim], sigma=sigma, mode='nearest'
            )

        # Renormalize
        norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
        quaternions = quaternions / (norms + 1e-10)

        return quaternions

    def align_to_initial_direction(
        self,
        positions: np.ndarray,
        quaternions: np.ndarray,
        target_dir: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Rotate trajectory so initial motion aligns with target direction.

        Args:
            positions: (N, 3) position trajectory.
            quaternions: (N, 4) wxyz quaternion trajectory.
            target_dir: Target direction (default: +x = [1,0,0]).

        Returns:
            Tuple of (aligned_positions, aligned_quaternions, rotation_matrix).
        """
        if target_dir is None:
            target_dir = np.array([1.0, 0.0, 0.0])

        target_dir = np.asarray(target_dir)
        target_dir = target_dir / (np.linalg.norm(target_dir) + 1e-10)

        # Compute initial direction
        from .diagnostics import compute_initial_direction
        init_dir, _ = compute_initial_direction(positions)

        # Check if already aligned
        if np.allclose(init_dir, target_dir, atol=1e-6):
            return positions, quaternions, np.eye(3)

        # Compute rotation from init_dir to target_dir
        R = self._rotation_between_vectors(init_dir, target_dir)

        # Apply rotation to positions (relative to first position)
        positions = positions.copy()
        origin = positions[0].copy()
        positions = (R @ (positions - origin).T).T + origin

        # Apply rotation to quaternions
        quaternions = quaternions.copy()
        R_quat = Rotation.from_matrix(R).as_quat()  # xyzw
        R_quat_wxyz = np.roll(R_quat, 1)  # wxyz

        for i in range(len(quaternions)):
            # Quaternion multiplication: R * q
            q = quaternions[i]
            q_rot = self._quat_multiply(R_quat_wxyz, q)
            quaternions[i] = q_rot

        return positions, quaternions, R

    def _rotation_between_vectors(
        self,
        v1: np.ndarray,
        v2: np.ndarray,
    ) -> np.ndarray:
        """Compute rotation matrix that rotates v1 to v2."""
        v1 = v1 / (np.linalg.norm(v1) + 1e-10)
        v2 = v2 / (np.linalg.norm(v2) + 1e-10)

        # Handle parallel/anti-parallel cases
        dot = np.dot(v1, v2)
        if dot > 0.9999:
            return np.eye(3)
        if dot < -0.9999:
            # Find orthogonal axis
            if abs(v1[0]) < 0.9:
                axis = np.cross(v1, np.array([1, 0, 0]))
            else:
                axis = np.cross(v1, np.array([0, 1, 0]))
            axis = axis / np.linalg.norm(axis)
            return Rotation.from_rotvec(np.pi * axis).as_matrix()

        # Rodrigues' rotation formula
        axis = np.cross(v1, v2)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.clip(dot, -1, 1))

        return Rotation.from_rotvec(angle * axis).as_matrix()

    def _quat_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply quaternions (wxyz convention)."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    # Convenience methods for standalone use

    def detect_reversals(self, positions: np.ndarray) -> List[int]:
        """Find indices where 180° reversals occur."""
        indices, _ = detect_reversals(positions, self.reversal_threshold)
        return indices

    def detect_zero_motion(
        self,
        positions: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """Find zero-motion segments as (start, end) tuples."""
        return detect_zero_motion(positions, self.zero_motion_threshold)
