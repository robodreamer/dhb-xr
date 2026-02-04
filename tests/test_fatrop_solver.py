"""Tests for Fatrop-based trajectory optimization."""

import numpy as np
import pytest

# Check if rockit is available
try:
    import rockit
    HAS_ROCKIT = True
except ImportError:
    HAS_ROCKIT = False

pytestmark = pytest.mark.skipif(not HAS_ROCKIT, reason="rockit not installed")


@pytest.fixture
def demo_trajectory():
    """Generate a simple helix demo trajectory."""
    n = 100
    t = np.linspace(0, 2 * np.pi, n)
    positions = np.column_stack([
        0.3 * np.cos(t),
        0.3 * np.sin(t),
        0.1 * t / (2 * np.pi),
    ])
    quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    return positions, quaternions


@pytest.fixture
def boundary_poses():
    """Define new start/goal poses for adaptation."""
    start_pose = {
        'position': np.array([0.1, 0.1, 0.0]),
        'quaternion': np.array([1.0, 0.0, 0.0, 0.0]),
    }
    goal_pose = {
        'position': np.array([0.4, -0.2, 0.15]),
        'quaternion': np.array([1.0, 0.0, 0.0, 0.0]),
    }
    return start_pose, goal_pose


class TestFatropTrajectoryGenerator:
    """Tests for FatropTrajectoryGenerator class."""

    def test_import(self):
        """Test that Fatrop solver can be imported."""
        from dhb_xr.optimization.fatrop_solver import (
            FatropTrajectoryGenerator,
            ConstrainedTrajectoryGenerator,
            generate_trajectory_fatrop,
        )
        assert FatropTrajectoryGenerator is not None
        assert ConstrainedTrajectoryGenerator is not None
        assert generate_trajectory_fatrop is not None

    def test_generator_initialization(self):
        """Test that FatropTrajectoryGenerator can be initialized."""
        from dhb_xr.optimization.fatrop_solver import FatropTrajectoryGenerator

        # Test with Fatrop (may fail if fatrop binary not available)
        try:
            gen = FatropTrajectoryGenerator(N=20, use_fatrop=True)
            assert gen.N == 20
            assert gen.use_fatrop is True
        except Exception:
            # Fatrop binary may not be available
            pass

        # Test with IPOPT (should always work with CasADi)
        gen = FatropTrajectoryGenerator(N=20, use_fatrop=False)
        assert gen.N == 20
        assert gen.use_fatrop is False

    def test_generate_with_ipopt(self, demo_trajectory, boundary_poses):
        """Test trajectory generation with IPOPT solver."""
        from dhb_xr.optimization.fatrop_solver import FatropTrajectoryGenerator
        from dhb_xr.encoder.dhb_dr import encode_dhb_dr
        from dhb_xr.core import geometry as geom

        positions, quaternions = demo_trajectory
        start_pose, goal_pose = boundary_poses

        # Encode demo
        init_pose = {'position': positions[0], 'quaternion': quaternions[0]}
        from dhb_xr.core.types import EncodingMethod
        result = encode_dhb_dr(positions, quaternions, init_pose=init_pose, method=EncodingMethod.POSITION)
        demo_lin = result['linear_motion_invariants']
        demo_ang = result['angular_motion_invariants']

        # Create generator with IPOPT
        N = 30
        gen = FatropTrajectoryGenerator(N=N, use_fatrop=False, verbose=False)

        # Resample invariants to match N
        from scipy.interpolate import interp1d
        t_orig = np.linspace(0, 1, len(demo_lin))
        t_new = np.linspace(0, 1, N)
        demo_lin_resampled = interp1d(t_orig, demo_lin, axis=0, fill_value='extrapolate')(t_new)
        demo_ang_resampled = interp1d(t_orig, demo_ang, axis=0, fill_value='extrapolate')(t_new)

        # Generate
        result = gen.generate(
            demo_lin_invariants=demo_lin_resampled,
            demo_ang_invariants=demo_ang_resampled,
            start_pos=start_pose['position'],
            start_rot=geom.quat_to_rot(start_pose['quaternion']),
            goal_pos=goal_pose['position'],
            goal_rot=geom.quat_to_rot(goal_pose['quaternion']),
        )

        # Check result structure
        assert 'positions' in result
        assert 'rotations' in result
        assert 'linear_invariants' in result
        assert 'solve_time' in result
        assert 'success' in result

        # Check shapes
        assert result['positions'].shape[0] == N  # N+1 states but rockit returns N control samples
        assert result['positions'].shape[1] == 3

        # Check boundary conditions are approximately satisfied
        # Note: May not be exact due to discretization
        pos_start_error = np.linalg.norm(result['positions'][0] - start_pose['position'])
        pos_end_error = np.linalg.norm(result['positions'][-1] - goal_pose['position'])

        print(f"Start position error: {pos_start_error:.6f}")
        print(f"End position error: {pos_end_error:.6f}")
        print(f"Solve time: {result['solve_time']*1000:.1f} ms")
        print(f"Success: {result['success']}")

        # Relaxed tolerance for optimization
        assert pos_start_error < 0.1, f"Start position error too large: {pos_start_error}"


class TestGenerateTrajectoryFatrop:
    """Tests for the high-level generate_trajectory_fatrop function."""

    def test_basic_generation(self, demo_trajectory, boundary_poses):
        """Test basic trajectory generation."""
        from dhb_xr.optimization.fatrop_solver import generate_trajectory_fatrop

        positions, quaternions = demo_trajectory
        start_pose, goal_pose = boundary_poses

        result = generate_trajectory_fatrop(
            positions, quaternions,
            start_pose, goal_pose,
            traj_length=30,
            use_fatrop=False,  # Use IPOPT for reliable testing
            verbose=False,
        )

        # Check result structure
        assert 'positions' in result
        assert 'quaternions' in result
        assert 'linear_invariants' in result
        assert 'angular_invariants' in result
        assert 'solve_time' in result
        assert 'success' in result
        assert 'solver' in result

        assert result['solver'] == 'ipopt'
        print(f"Solve time: {result['solve_time']*1000:.1f} ms")

    def test_fatrop_vs_ipopt_speed(self, demo_trajectory, boundary_poses):
        """Compare Fatrop vs IPOPT solve times (if Fatrop available)."""
        from dhb_xr.optimization.fatrop_solver import generate_trajectory_fatrop
        import time

        positions, quaternions = demo_trajectory
        start_pose, goal_pose = boundary_poses

        # IPOPT timing
        times_ipopt = []
        for _ in range(3):
            t0 = time.perf_counter()
            result = generate_trajectory_fatrop(
                positions, quaternions,
                start_pose, goal_pose,
                traj_length=30,
                use_fatrop=False,
            )
            times_ipopt.append(time.perf_counter() - t0)

        print(f"IPOPT: {np.mean(times_ipopt)*1000:.1f} ms (avg of {len(times_ipopt)} runs)")

        # Try Fatrop
        try:
            times_fatrop = []
            for _ in range(3):
                t0 = time.perf_counter()
                result = generate_trajectory_fatrop(
                    positions, quaternions,
                    start_pose, goal_pose,
                    traj_length=30,
                    use_fatrop=True,
                )
                times_fatrop.append(time.perf_counter() - t0)

            print(f"Fatrop: {np.mean(times_fatrop)*1000:.1f} ms (avg of {len(times_fatrop)} runs)")
            print(f"Speedup: {np.mean(times_ipopt)/np.mean(times_fatrop):.1f}x")
        except Exception as e:
            print(f"Fatrop not available: {e}")
            pytest.skip("Fatrop binary not available")


class TestGetOptimizer:
    """Tests for the get_optimizer factory function."""

    def test_get_optimizer_fatrop(self):
        """Test getting Fatrop optimizer via factory."""
        from dhb_xr.optimization import get_optimizer

        opt = get_optimizer(backend="fatrop", N=30)
        if opt is not None:
            assert hasattr(opt, 'generate')
        else:
            pytest.skip("Fatrop not available")

    def test_get_optimizer_ipopt(self):
        """Test getting IPOPT optimizer via factory."""
        from dhb_xr.optimization import get_optimizer

        opt = get_optimizer(backend="ipopt", N=30)
        assert opt is not None
        assert hasattr(opt, 'generate')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
