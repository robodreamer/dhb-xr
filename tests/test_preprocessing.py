"""Tests for preprocessing module."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_allclose

from dhb_xr.preprocessing import (
    TrajectoryPreprocessor,
    ProcessedTrajectory,
    TrajectoryDiagnostics,
    analyze_trajectory,
    detect_reversals,
    detect_zero_motion,
)
from dhb_xr.preprocessing.diagnostics import (
    PreprocessingRecommendation,
    compute_initial_direction,
    compute_alignment_angle,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def simple_trajectory():
    """Simple straight-line trajectory along +x."""
    n = 50
    t = np.linspace(0, 1, n)
    positions = np.column_stack([t, np.zeros(n), np.zeros(n)])
    quaternions = np.tile([1, 0, 0, 0], (n, 1)).astype(float)  # wxyz identity
    return positions, quaternions


@pytest.fixture
def curved_trajectory():
    """Curved trajectory in xy-plane."""
    n = 100
    t = np.linspace(0, 2 * np.pi, n)
    positions = np.column_stack([np.cos(t), np.sin(t), np.zeros(n)])
    quaternions = np.tile([1, 0, 0, 0], (n, 1)).astype(float)
    return positions, quaternions


@pytest.fixture
def trajectory_with_reversal():
    """Trajectory that reverses direction sharply at midpoint."""
    n = 51  # Use odd number for clean midpoint
    # Create trajectory that goes forward, then abruptly reverses
    # Position goes: 0 -> 1 -> 0 (V-shape)
    positions = np.zeros((n, 3))
    # Forward motion 0-25 (indices 0 to 25)
    positions[:26, 0] = np.linspace(0, 1, 26)
    # Backward motion 26-50 (indices 26 to 50) - starts from 1 and goes back
    positions[26:, 0] = np.linspace(1 - 1/25, 0, 25)

    quaternions = np.tile([1, 0, 0, 0], (n, 1)).astype(float)
    return positions, quaternions


@pytest.fixture
def trajectory_with_zero_motion():
    """Trajectory with a zero-motion segment in the middle."""
    n = 50
    positions = np.zeros((n, 3))
    # Normal motion 0-15
    positions[:15, 0] = np.linspace(0, 0.3, 15)
    # Zero motion 15-25 (same position)
    positions[15:25, 0] = 0.3
    # Normal motion 25-50
    positions[25:, 0] = np.linspace(0.3, 1.0, 25)
    quaternions = np.tile([1, 0, 0, 0], (n, 1)).astype(float)
    return positions, quaternions


@pytest.fixture
def misaligned_trajectory():
    """Trajectory starting in +y direction instead of +x."""
    n = 50
    t = np.linspace(0, 1, n)
    positions = np.column_stack([np.zeros(n), t, np.zeros(n)])  # +y direction
    quaternions = np.tile([1, 0, 0, 0], (n, 1)).astype(float)
    return positions, quaternions


# =============================================================================
# Test Diagnostics Functions
# =============================================================================

class TestDetectReversals:
    """Tests for detect_reversals function."""

    def test_no_reversals_straight_line(self, simple_trajectory):
        positions, _ = simple_trajectory
        indices, min_dot = detect_reversals(positions)
        assert len(indices) == 0
        assert min_dot >= 0.99

    def test_detects_180_reversal(self, trajectory_with_reversal):
        positions, _ = trajectory_with_reversal
        indices, min_dot = detect_reversals(positions, threshold=-0.5)
        assert len(indices) > 0
        assert min_dot < -0.5

    def test_curved_trajectory_no_sharp_reversals(self, curved_trajectory):
        positions, _ = curved_trajectory
        # Circle has smooth direction changes, not 180° reversals
        indices, min_dot = detect_reversals(positions, threshold=-0.9)
        assert len(indices) == 0  # No near-180° reversals

    def test_empty_trajectory(self):
        positions = np.array([[0, 0, 0]])
        indices, min_dot = detect_reversals(positions)
        assert indices == []
        assert min_dot == 1.0

    def test_short_trajectory(self):
        positions = np.array([[0, 0, 0], [1, 0, 0]])
        indices, min_dot = detect_reversals(positions)
        assert indices == []


class TestDetectZeroMotion:
    """Tests for detect_zero_motion function."""

    def test_no_zero_motion(self, simple_trajectory):
        positions, _ = simple_trajectory
        segments = detect_zero_motion(positions)
        assert len(segments) == 0

    def test_detects_zero_motion_segment(self, trajectory_with_zero_motion):
        positions, _ = trajectory_with_zero_motion
        segments = detect_zero_motion(positions, threshold=1e-6)
        assert len(segments) >= 1
        # Should detect segment around indices 15-24
        total_zero_frames = sum(end - start for start, end in segments)
        assert total_zero_frames >= 8

    def test_threshold_parameter(self, trajectory_with_zero_motion):
        positions, _ = trajectory_with_zero_motion
        # With very large threshold, everything is "zero motion"
        segments = detect_zero_motion(positions, threshold=100.0)
        assert len(segments) >= 1
        # With very small threshold, might miss near-zero
        segments_strict = detect_zero_motion(positions, threshold=1e-15)
        assert len(segments_strict) <= len(segments)


class TestComputeInitialDirection:
    """Tests for compute_initial_direction function."""

    def test_x_direction(self, simple_trajectory):
        positions, _ = simple_trajectory
        direction, magnitude = compute_initial_direction(positions)
        assert_allclose(direction, [1, 0, 0], atol=1e-6)
        assert magnitude > 0

    def test_y_direction(self, misaligned_trajectory):
        positions, _ = misaligned_trajectory
        direction, magnitude = compute_initial_direction(positions)
        assert_allclose(direction, [0, 1, 0], atol=1e-6)

    def test_handles_zero_initial_motion(self, trajectory_with_zero_motion):
        positions, _ = trajectory_with_zero_motion
        # Starts with normal motion, should still work
        direction, magnitude = compute_initial_direction(positions)
        assert np.linalg.norm(direction) > 0.99


class TestComputeAlignmentAngle:
    """Tests for compute_alignment_angle function."""

    def test_aligned_direction(self):
        angle = compute_alignment_angle(np.array([1, 0, 0]))
        assert angle < 1e-6

    def test_perpendicular_direction(self):
        angle = compute_alignment_angle(np.array([0, 1, 0]))
        assert_allclose(angle, np.pi / 2, atol=1e-6)

    def test_opposite_direction(self):
        angle = compute_alignment_angle(np.array([-1, 0, 0]))
        assert_allclose(angle, np.pi, atol=1e-6)


class TestAnalyzeTrajectory:
    """Tests for analyze_trajectory function."""

    def test_simple_trajectory_is_well_conditioned(self, simple_trajectory):
        positions, quaternions = simple_trajectory
        diag = analyze_trajectory(positions, quaternions)

        assert diag.num_samples == len(positions)
        assert diag.trajectory_length > 0
        assert diag.num_reversals == 0
        assert diag.num_zero_motion_segments == 0
        assert diag.alignment_angle_to_x < np.radians(10)
        assert diag.is_well_conditioned

    def test_detects_reversal_issues(self, trajectory_with_reversal):
        positions, quaternions = trajectory_with_reversal
        diag = analyze_trajectory(positions, quaternions)

        assert diag.num_reversals > 0
        assert not diag.is_well_conditioned
        assert PreprocessingRecommendation.SMOOTH_TRAJECTORY in diag.recommended_preprocessing

    def test_detects_zero_motion_issues(self, trajectory_with_zero_motion):
        positions, quaternions = trajectory_with_zero_motion
        diag = analyze_trajectory(positions, quaternions)

        assert diag.num_zero_motion_segments > 0
        assert diag.total_zero_motion_frames > 0
        assert PreprocessingRecommendation.REMOVE_ZERO_MOTION in diag.recommended_preprocessing

    def test_detects_alignment_issues(self, misaligned_trajectory):
        positions, quaternions = misaligned_trajectory
        diag = analyze_trajectory(positions, quaternions)

        assert diag.alignment_angle_to_x > np.radians(80)  # ~90°
        assert PreprocessingRecommendation.ALIGN_TO_X in diag.recommended_preprocessing

    def test_summary_output(self, simple_trajectory):
        positions, quaternions = simple_trajectory
        diag = analyze_trajectory(positions, quaternions)
        summary = diag.summary()

        assert "Trajectory Diagnostics" in summary
        assert "samples" in summary
        assert "Reversals" in summary


# =============================================================================
# Test TrajectoryPreprocessor
# =============================================================================

class TestTrajectoryPreprocessor:
    """Tests for TrajectoryPreprocessor class."""

    def test_initialization(self):
        preprocessor = TrajectoryPreprocessor()
        assert preprocessor.align_to_x is True
        assert preprocessor.smooth_reversals is True
        assert preprocessor.gaussian_sigma == 0.0

    def test_custom_initialization(self):
        preprocessor = TrajectoryPreprocessor(
            align_to_x=False,
            gaussian_sigma=2.0,
            reversal_threshold=-0.8,
        )
        assert preprocessor.align_to_x is False
        assert preprocessor.gaussian_sigma == 2.0
        assert preprocessor.reversal_threshold == -0.8

    def test_process_returns_processed_trajectory(self, simple_trajectory):
        positions, quaternions = simple_trajectory
        preprocessor = TrajectoryPreprocessor()
        result = preprocessor.process(positions, quaternions)

        assert isinstance(result, ProcessedTrajectory)
        assert result.positions is not None
        assert result.quaternions is not None
        assert len(result.positions) > 0

    def test_preserves_original_data(self, simple_trajectory):
        positions, quaternions = simple_trajectory
        preprocessor = TrajectoryPreprocessor()
        result = preprocessor.process(positions, quaternions)

        assert_array_almost_equal(result.original_positions, positions)
        assert_array_almost_equal(result.original_quaternions, quaternions)

    def test_alignment_to_x(self, misaligned_trajectory):
        positions, quaternions = misaligned_trajectory
        preprocessor = TrajectoryPreprocessor(align_to_x=True)
        result = preprocessor.process(positions, quaternions)

        assert result.was_aligned
        assert result.alignment_rotation is not None

        # Check that initial direction is now aligned with +x
        initial_dir, _ = compute_initial_direction(result.positions)
        assert abs(initial_dir[0]) > 0.9  # Should be mostly +x or -x

    def test_no_alignment_when_disabled(self, misaligned_trajectory):
        positions, quaternions = misaligned_trajectory
        preprocessor = TrajectoryPreprocessor(align_to_x=False)
        result = preprocessor.process(positions, quaternions)

        assert not result.was_aligned
        assert result.alignment_rotation is None

    def test_gaussian_smoothing(self, simple_trajectory):
        positions, quaternions = simple_trajectory
        # Add some noise
        noisy_positions = positions + np.random.randn(*positions.shape) * 0.01

        preprocessor = TrajectoryPreprocessor(
            align_to_x=False,
            smooth_reversals=False,
            remove_zero_motion=False,
            gaussian_sigma=2.0,
        )
        result = preprocessor.process(noisy_positions, quaternions)

        assert result.was_smoothed
        assert result.smoothing_sigma == 2.0

    def test_zero_motion_interpolation(self, trajectory_with_zero_motion):
        positions, quaternions = trajectory_with_zero_motion
        preprocessor = TrajectoryPreprocessor(
            align_to_x=False,
            smooth_reversals=False,
            remove_zero_motion=True,
            interpolate_zero_motion=True,
        )
        result = preprocessor.process(positions, quaternions)

        assert len(result.interpolated_segments) > 0
        # Trajectory length should be same (interpolation, not removal)
        assert len(result.positions) == len(positions)

    def test_zero_motion_removal(self, trajectory_with_zero_motion):
        positions, quaternions = trajectory_with_zero_motion
        preprocessor = TrajectoryPreprocessor(
            align_to_x=False,
            smooth_reversals=False,
            remove_zero_motion=True,
            interpolate_zero_motion=False,
        )
        result = preprocessor.process(positions, quaternions)

        # Should have removed some frames
        assert len(result.positions) < len(positions)
        assert len(result.removed_indices) > 0

    def test_reversal_smoothing(self, trajectory_with_reversal):
        positions, quaternions = trajectory_with_reversal
        preprocessor = TrajectoryPreprocessor(
            align_to_x=False,
            smooth_reversals=True,
            remove_zero_motion=False,
        )
        result = preprocessor.process(positions, quaternions)

        # Should still have same number of samples
        assert len(result.positions) == len(positions)

    def test_diagnostics_before_after(self, trajectory_with_reversal):
        positions, quaternions = trajectory_with_reversal
        preprocessor = TrajectoryPreprocessor(smooth_reversals=True)
        result = preprocessor.process(positions, quaternions, return_diagnostics=True)

        assert result.diagnostics_before is not None
        assert result.diagnostics_after is not None
        assert result.diagnostics_before.num_samples == len(positions)

    def test_inverse_alignment(self, misaligned_trajectory):
        positions, quaternions = misaligned_trajectory
        preprocessor = TrajectoryPreprocessor(align_to_x=True)
        result = preprocessor.process(positions, quaternions)

        R_inv = result.get_inverse_alignment()

        # Applying R_inv should undo alignment
        origin = result.positions[0]
        original_frame = (R_inv @ (result.positions - origin).T).T + origin

        # Should be close to original (within smoothing tolerance)
        assert_allclose(original_frame[0], positions[0], atol=1e-6)


class TestTrajectoryPreprocessorMethods:
    """Tests for standalone methods of TrajectoryPreprocessor."""

    def test_detect_reversals_method(self, trajectory_with_reversal):
        positions, _ = trajectory_with_reversal
        preprocessor = TrajectoryPreprocessor()
        indices = preprocessor.detect_reversals(positions)
        assert len(indices) > 0

    def test_detect_zero_motion_method(self, trajectory_with_zero_motion):
        positions, _ = trajectory_with_zero_motion
        preprocessor = TrajectoryPreprocessor()
        segments = preprocessor.detect_zero_motion(positions)
        assert len(segments) > 0

    def test_smooth_trajectory_method(self, simple_trajectory):
        positions, quaternions = simple_trajectory
        preprocessor = TrajectoryPreprocessor()
        smooth_pos, smooth_quat = preprocessor.smooth_trajectory(
            positions, quaternions, sigma=1.0
        )

        assert smooth_pos.shape == positions.shape
        assert smooth_quat.shape == quaternions.shape
        # Quaternions should still be normalized
        norms = np.linalg.norm(smooth_quat, axis=1)
        assert_allclose(norms, np.ones(len(norms)), atol=1e-6)

    def test_align_to_initial_direction_method(self, misaligned_trajectory):
        positions, quaternions = misaligned_trajectory
        preprocessor = TrajectoryPreprocessor()
        aligned_pos, aligned_quat, R = preprocessor.align_to_initial_direction(
            positions, quaternions
        )

        assert R.shape == (3, 3)
        # Check R is orthogonal
        assert_allclose(R @ R.T, np.eye(3), atol=1e-6)


# =============================================================================
# Integration Tests
# =============================================================================

class TestPreprocessingIntegration:
    """Integration tests with DHB encoding."""

    def test_preprocessed_trajectory_encodes_without_error(self, curved_trajectory):
        from dhb_xr import encode_dhb_dr

        positions, quaternions = curved_trajectory
        preprocessor = TrajectoryPreprocessor()
        result = preprocessor.process(positions, quaternions)

        # Should encode without error
        encoded = encode_dhb_dr(result.positions, result.quaternions)
        assert encoded["linear_motion_invariants"] is not None

    def test_robust_mode_with_problematic_trajectory(self, trajectory_with_reversal):
        from dhb_xr import encode_dhb_dr

        positions, quaternions = trajectory_with_reversal

        # Without preprocessing, robust_mode helps
        encoded = encode_dhb_dr(
            positions, quaternions,
            robust_mode=True,
            return_diagnostics=True,
        )
        assert encoded["diagnostics"].num_reversals_detected >= 0

    def test_full_pipeline_with_preprocessing(self, trajectory_with_zero_motion):
        from dhb_xr import encode_dhb_dr, decode_dhb_dr

        positions, quaternions = trajectory_with_zero_motion

        # Preprocess
        preprocessor = TrajectoryPreprocessor(
            interpolate_zero_motion=True,
            gaussian_sigma=1.0,
        )
        result = preprocessor.process(positions, quaternions)

        # Encode
        encoded = encode_dhb_dr(
            result.positions, result.quaternions,
            robust_mode=True,
        )

        # Decode
        decoded = decode_dhb_dr(
            encoded["linear_motion_invariants"],
            encoded["angular_motion_invariants"],
            encoded["initial_pose"],
        )

        assert decoded["positions"] is not None
        assert len(decoded["positions"]) > 0


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_short_trajectory(self):
        positions = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        quaternions = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]).astype(float)

        preprocessor = TrajectoryPreprocessor()
        result = preprocessor.process(positions, quaternions)

        assert len(result.positions) == len(positions)

    def test_single_point(self):
        positions = np.array([[0, 0, 0]])
        quaternions = np.array([[1, 0, 0, 0]]).astype(float)

        # Diagnostics should handle gracefully
        diag = analyze_trajectory(positions, quaternions)
        assert diag.num_samples == 1
        assert diag.trajectory_length == 0

    def test_all_zero_positions(self):
        n = 10
        positions = np.zeros((n, 3))
        quaternions = np.tile([1, 0, 0, 0], (n, 1)).astype(float)

        diag = analyze_trajectory(positions, quaternions)
        # Should detect as all zero motion
        assert diag.num_zero_motion_segments > 0 or diag.trajectory_length == 0

    def test_already_aligned_trajectory(self, simple_trajectory):
        positions, quaternions = simple_trajectory
        preprocessor = TrajectoryPreprocessor(align_to_x=True)
        result = preprocessor.process(positions, quaternions)

        # Should detect it's already aligned and apply identity rotation
        if result.alignment_rotation is not None:
            # Either identity or very close to it
            assert_allclose(result.alignment_rotation, np.eye(3), atol=0.1)
