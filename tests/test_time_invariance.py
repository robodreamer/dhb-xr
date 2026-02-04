"""Tests for DHB-TI time-invariance behavior with different resampling methods.

These tests verify that DHB-TI produces similar invariants for the same motion
executed at different speeds, and compare the impact of interpolation methods.
"""

import numpy as np
import pytest
from dhb_xr.encoder.dhb_ti import encode_dhb_dr_ti
from dhb_xr.utils.resampling import resample_trajectory
from dhb_xr.core import geometry as geom


# ============================================================================
# Test fixtures: synthetic trajectories
# ============================================================================

@pytest.fixture
def base_trajectory():
    """Create a smooth, well-conditioned base trajectory.

    Uses a combination of linear and sinusoidal motion to create
    a non-trivial but smooth path that's suitable for resampling tests.
    """
    n_base = 100
    t = np.linspace(0, 1, n_base)

    # Smooth trajectory: linear in x/z, sinusoidal in y
    positions = np.column_stack([
        0.3 * t,                      # Linear in x
        0.15 * np.sin(np.pi * t),     # Smooth sinusoid in y
        0.1 * t                       # Linear in z
    ])

    # Smooth rotation: gradual rotation around z-axis
    quaternions = np.array([
        geom.axis_angle_to_quat(np.array([0, 0, 0.5 * ti]))
        for ti in t
    ])

    return positions, quaternions


@pytest.fixture
def complex_trajectory():
    """Create a more complex trajectory with varying curvature."""
    n_base = 150
    t = np.linspace(0, 2 * np.pi, n_base)

    # Figure-8 like motion with varying speed
    positions = np.column_stack([
        0.2 * np.sin(t),
        0.1 * np.sin(2 * t),
        0.05 * np.cos(t)
    ])

    # Varying rotation
    quaternions = np.array([
        geom.axis_angle_to_quat(np.array([0.3 * np.sin(ti), 0.2 * np.cos(ti), 0]))
        for ti in t
    ])

    return positions, quaternions


# ============================================================================
# Resampling functions for comparison
# ============================================================================

def resample_linear(positions, quaternions, target_len):
    """Resample using linear interpolation (numpy.interp)."""
    t_orig = np.linspace(0, 1, len(positions))
    t_new = np.linspace(0, 1, target_len)

    pos_new = np.column_stack([
        np.interp(t_new, t_orig, positions[:, i]) for i in range(3)
    ])
    quat_new = np.column_stack([
        np.interp(t_new, t_orig, quaternions[:, i]) for i in range(4)
    ])
    # Normalize quaternions
    quat_new = quat_new / np.linalg.norm(quat_new, axis=1, keepdims=True)

    return pos_new, quat_new


def resample_cubic(positions, quaternions, target_len):
    """Resample using cubic spline interpolation (scipy CubicSpline)."""
    return resample_trajectory(positions, quaternions, target_len, smoothing=False)


def resample_cubic_smooth(positions, quaternions, target_len, window=10):
    """Resample using cubic spline + Savitzky-Golay smoothing."""
    return resample_trajectory(
        positions, quaternions, target_len,
        smoothing=True, window_length=min(window, target_len - 1)
    )


# ============================================================================
# Helper functions
# ============================================================================

def compute_invariant_difference(inv1, inv2):
    """Compute the mean and max L2 difference between two invariant arrays."""
    if inv1.shape != inv2.shape:
        raise ValueError(f"Shape mismatch: {inv1.shape} vs {inv2.shape}")

    diffs = np.linalg.norm(inv1 - inv2, axis=1)
    return {
        'mean': float(np.mean(diffs)),
        'max': float(np.max(diffs)),
        'std': float(np.std(diffs))
    }


def encode_and_compare(pos1, quat1, pos2, quat2, M=80):
    """Encode two trajectories and compute invariant difference."""
    result1 = encode_dhb_dr_ti(pos1, quat1, M=M, progress_kind='hybrid')
    result2 = encode_dhb_dr_ti(pos2, quat2, M=M, progress_kind='hybrid')

    linear_diff = compute_invariant_difference(
        result1['linear_motion_invariants'],
        result2['linear_motion_invariants']
    )
    angular_diff = compute_invariant_difference(
        result1['angular_motion_invariants'],
        result2['angular_motion_invariants']
    )

    return {
        'linear': linear_diff,
        'angular': angular_diff,
        'result1': result1,
        'result2': result2
    }


# ============================================================================
# Time-invariance tests: same trajectory at different speeds
# ============================================================================

class TestTimeInvarianceWithLinearInterpolation:
    """Tests for time-invariance using linear interpolation (baseline)."""

    def test_downsampling_invariance(self, base_trajectory):
        """Downsampling (fewer frames = faster motion) should preserve invariants well."""
        pos_base, quat_base = base_trajectory

        # Normal: 100 frames, Fast: 50 frames
        pos_fast, quat_fast = resample_linear(pos_base, quat_base, 50)

        comparison = encode_and_compare(pos_base, quat_base, pos_fast, quat_fast)

        # Downsampling with linear interp should work reasonably well
        assert comparison['linear']['mean'] < 0.001, \
            f"Downsampling invariance failed: mean diff = {comparison['linear']['mean']}"

    def test_upsampling_invariance(self, base_trajectory):
        """Upsampling (more frames = slower motion) may show more deviation with linear interp."""
        pos_base, quat_base = base_trajectory

        # Normal: 100 frames, Slow: 200 frames
        pos_slow, quat_slow = resample_linear(pos_base, quat_base, 200)

        comparison = encode_and_compare(pos_base, quat_base, pos_slow, quat_slow)

        # Note: Linear interpolation upsampling may have higher error
        # This test documents the expected behavior
        assert comparison['linear']['mean'] < 0.002, \
            f"Upsampling invariance failed: mean diff = {comparison['linear']['mean']}"


class TestTimeInvarianceWithCubicInterpolation:
    """Tests for time-invariance using cubic spline interpolation (recommended)."""

    def test_downsampling_invariance(self, base_trajectory):
        """Cubic spline downsampling should preserve invariants well."""
        pos_base, quat_base = base_trajectory

        pos_fast, quat_fast = resample_cubic(pos_base, quat_base, 50)

        comparison = encode_and_compare(pos_base, quat_base, pos_fast, quat_fast)

        # Cubic spline should be better than linear
        assert comparison['linear']['mean'] < 0.0005, \
            f"Cubic downsampling failed: mean diff = {comparison['linear']['mean']}"

    def test_upsampling_invariance(self, base_trajectory):
        """Cubic spline upsampling should preserve invariants much better than linear."""
        pos_base, quat_base = base_trajectory

        pos_slow, quat_slow = resample_cubic(pos_base, quat_base, 200)

        comparison = encode_and_compare(pos_base, quat_base, pos_slow, quat_slow)

        # Cubic spline upsampling should be significantly better
        assert comparison['linear']['mean'] < 0.0005, \
            f"Cubic upsampling failed: mean diff = {comparison['linear']['mean']}"

    def test_extreme_resampling(self, base_trajectory):
        """Test extreme resampling ratios (5x slower, 5x faster)."""
        pos_base, quat_base = base_trajectory
        n_base = len(pos_base)

        # Very slow: 5x frames
        pos_very_slow, quat_very_slow = resample_cubic(pos_base, quat_base, n_base * 5)

        # Very fast: 0.2x frames
        pos_very_fast, quat_very_fast = resample_cubic(pos_base, quat_base, n_base // 5)

        comparison_slow = encode_and_compare(pos_base, quat_base, pos_very_slow, quat_very_slow)
        comparison_fast = encode_and_compare(pos_base, quat_base, pos_very_fast, quat_very_fast)

        # Even extreme resampling should maintain reasonable invariance
        assert comparison_slow['linear']['mean'] < 0.002
        assert comparison_fast['linear']['mean'] < 0.002


class TestTimeInvarianceComparison:
    """Direct comparison of interpolation methods."""

    def test_cubic_better_than_linear_for_upsampling(self, base_trajectory):
        """Cubic spline should outperform linear interpolation for upsampling."""
        pos_base, quat_base = base_trajectory

        # Upsample to 200 frames
        pos_linear, quat_linear = resample_linear(pos_base, quat_base, 200)
        pos_cubic, quat_cubic = resample_cubic(pos_base, quat_base, 200)

        comp_linear = encode_and_compare(pos_base, quat_base, pos_linear, quat_linear)
        comp_cubic = encode_and_compare(pos_base, quat_base, pos_cubic, quat_cubic)

        # Cubic should be better (lower mean difference)
        assert comp_cubic['linear']['mean'] < comp_linear['linear']['mean'], \
            f"Cubic ({comp_cubic['linear']['mean']:.6f}) should be better than linear ({comp_linear['linear']['mean']:.6f})"

    def test_both_methods_similar_for_downsampling(self, base_trajectory):
        """Both methods should perform similarly for downsampling (picking fewer points)."""
        pos_base, quat_base = base_trajectory

        # Downsample to 50 frames
        pos_linear, quat_linear = resample_linear(pos_base, quat_base, 50)
        pos_cubic, quat_cubic = resample_cubic(pos_base, quat_base, 50)

        comp_linear = encode_and_compare(pos_base, quat_base, pos_linear, quat_linear)
        comp_cubic = encode_and_compare(pos_base, quat_base, pos_cubic, quat_cubic)

        # Both should be reasonable
        assert comp_linear['linear']['mean'] < 0.001
        assert comp_cubic['linear']['mean'] < 0.001

    def test_invariance_score_summary(self, base_trajectory):
        """Generate a summary of invariance scores for documentation."""
        pos_base, quat_base = base_trajectory
        n_base = len(pos_base)

        results = {}

        for method_name, resample_fn in [
            ('linear', resample_linear),
            ('cubic', resample_cubic)
        ]:
            for ratio, target_len in [
                ('0.5x (fast)', n_base // 2),
                ('2x (slow)', n_base * 2),
                ('0.25x (very fast)', n_base // 4),
                ('4x (very slow)', n_base * 4)
            ]:
                pos_new, quat_new = resample_fn(pos_base, quat_base, target_len)
                comp = encode_and_compare(pos_base, quat_base, pos_new, quat_new)
                results[(method_name, ratio)] = comp['linear']['mean']

        # Print summary for reference (visible in pytest -v output)
        print("\n\nTime-Invariance Score Summary (lower = better):")
        print("-" * 60)
        print(f"{'Method':<10} {'Speed Ratio':<20} {'Mean Diff':>15}")
        print("-" * 60)
        for (method, ratio), score in sorted(results.items()):
            print(f"{method:<10} {ratio:<20} {score:>15.6f}")

        # All should be reasonably small
        for key, score in results.items():
            assert score < 0.01, f"Invariance score too high for {key}: {score}"


class TestComplexTrajectory:
    """Test time-invariance on more complex trajectories."""

    def test_figure8_trajectory(self, complex_trajectory):
        """Test time-invariance on a figure-8 trajectory with varying curvature."""
        pos_base, quat_base = complex_trajectory

        # Use cubic spline for best results
        pos_fast, quat_fast = resample_cubic(pos_base, quat_base, 75)
        pos_slow, quat_slow = resample_cubic(pos_base, quat_base, 300)

        comp_fast = encode_and_compare(pos_base, quat_base, pos_fast, quat_fast)
        comp_slow = encode_and_compare(pos_base, quat_base, pos_slow, quat_slow)

        # Both should maintain reasonable invariance
        assert comp_fast['linear']['mean'] < 0.005, \
            f"Complex trajectory fast failed: {comp_fast['linear']['mean']}"
        assert comp_slow['linear']['mean'] < 0.005, \
            f"Complex trajectory slow failed: {comp_slow['linear']['mean']}"


class TestRotationOnlyTrajectory:
    """Test time-invariance for pure rotation trajectories."""

    def test_pure_rotation(self):
        """Test time-invariance when only rotation changes (no translation)."""
        n_base = 100
        t = np.linspace(0, 1, n_base)

        # No translation
        positions = np.zeros((n_base, 3))

        # Rotation around z-axis
        quaternions = np.array([
            geom.axis_angle_to_quat(np.array([0, 0, np.pi * ti]))
            for ti in t
        ])

        # Resample
        pos_fast, quat_fast = resample_cubic(positions, quaternions, 50)
        pos_slow, quat_slow = resample_cubic(positions, quaternions, 200)

        comp_fast = encode_and_compare(positions, quaternions, pos_fast, quat_fast)
        comp_slow = encode_and_compare(positions, quaternions, pos_slow, quat_slow)

        # Angular invariants should be well-preserved
        assert comp_fast['angular']['mean'] < 0.01
        assert comp_slow['angular']['mean'] < 0.01


# ============================================================================
# Recommended usage tests
# ============================================================================

class TestRecommendedUsage:
    """Tests that demonstrate recommended usage patterns."""

    def test_recommended_resampling_workflow(self, base_trajectory):
        """Demonstrate the recommended workflow for time-invariant encoding."""
        pos_raw, quat_raw = base_trajectory

        # Step 1: Use cubic spline resampling (recommended)
        target_length = 80
        pos_resampled, quat_resampled = resample_trajectory(
            pos_raw, quat_raw, target_length, smoothing=False
        )

        # Step 2: Encode with DHB-TI
        result = encode_dhb_dr_ti(
            pos_resampled, quat_resampled,
            M=target_length,
            progress_kind='hybrid'
        )

        # Note: The output M from encode_dhb_dr_ti may differ slightly from input M
        # due to internal resampling. We just verify the output is valid.
        assert result['linear_motion_invariants'].ndim == 2
        assert result['linear_motion_invariants'].shape[1] == 4
        assert result['angular_motion_invariants'].ndim == 2
        assert result['angular_motion_invariants'].shape[1] == 4

    def test_avoid_linear_interpolation_for_upsampling(self, base_trajectory):
        """Document that linear interpolation should be avoided for upsampling."""
        pos_base, quat_base = base_trajectory
        n_base = len(pos_base)

        # Upsample 4x
        pos_linear, quat_linear = resample_linear(pos_base, quat_base, n_base * 4)
        pos_cubic, quat_cubic = resample_cubic(pos_base, quat_base, n_base * 4)

        comp_linear = encode_and_compare(pos_base, quat_base, pos_linear, quat_linear)
        comp_cubic = encode_and_compare(pos_base, quat_base, pos_cubic, quat_cubic)

        # Document the improvement from using cubic spline
        improvement = comp_linear['linear']['mean'] / (comp_cubic['linear']['mean'] + 1e-10)

        # Cubic should be at least 2x better for upsampling
        assert improvement > 2.0, \
            f"Expected significant improvement from cubic spline, got {improvement:.2f}x"
