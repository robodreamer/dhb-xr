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
    positions = np.column_stack(
        [
            0.3 * np.cos(t),
            0.3 * np.sin(t),
            0.1 * t / (2 * np.pi),
        ]
    )
    quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    return positions, quaternions


@pytest.fixture
def boundary_poses():
    """Define new start/goal poses for adaptation."""
    start_pose = {
        "position": np.array([0.1, 0.1, 0.0]),
        "quaternion": np.array([1.0, 0.0, 0.0, 0.0]),
    }
    goal_pose = {
        "position": np.array([0.4, -0.2, 0.15]),
        "quaternion": np.array([1.0, 0.0, 0.0, 0.0]),
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
        init_pose = {"position": positions[0], "quaternion": quaternions[0]}
        from dhb_xr.core.types import EncodingMethod

        result = encode_dhb_dr(
            positions, quaternions, init_pose=init_pose, method=EncodingMethod.POSITION
        )
        demo_lin = result["linear_motion_invariants"]
        demo_ang = result["angular_motion_invariants"]

        # Create generator with IPOPT
        N = 30
        gen = FatropTrajectoryGenerator(N=N, use_fatrop=False, verbose=False)

        # Resample invariants to match N
        from scipy.interpolate import interp1d

        t_orig = np.linspace(0, 1, len(demo_lin))
        t_new = np.linspace(0, 1, N)
        demo_lin_resampled = interp1d(t_orig, demo_lin, axis=0, fill_value="extrapolate")(t_new)
        demo_ang_resampled = interp1d(t_orig, demo_ang, axis=0, fill_value="extrapolate")(t_new)

        # Generate
        result = gen.generate(
            demo_lin_invariants=demo_lin_resampled,
            demo_ang_invariants=demo_ang_resampled,
            start_pos=start_pose["position"],
            start_rot=geom.quat_to_rot(start_pose["quaternion"]),
            goal_pos=goal_pose["position"],
            goal_rot=geom.quat_to_rot(goal_pose["quaternion"]),
        )

        # Check result structure
        assert "positions" in result
        assert "rotations" in result
        assert "linear_invariants" in result
        assert "solve_time" in result
        assert "success" in result

        # Check shapes
        assert result["positions"].shape[0] == N  # N+1 states but rockit returns N control samples
        assert result["positions"].shape[1] == 3

        # Check boundary conditions are approximately satisfied
        # Note: May not be exact due to discretization
        pos_start_error = np.linalg.norm(result["positions"][0] - start_pose["position"])
        pos_end_error = np.linalg.norm(result["positions"][-1] - goal_pose["position"])

        print(f"Start position error: {pos_start_error:.6f}")
        print(f"End position error: {pos_end_error:.6f}")
        print(f"Solve time: {result['solve_time'] * 1000:.1f} ms")
        print(f"Success: {result['success']}")

        # Relaxed tolerance for optimization
        assert pos_start_error < 0.1, f"Start position error too large: {pos_start_error}"


class TestGenerateTrajectoryFatrop:
    """Tests for the high-level generate_trajectory_fatrop function."""

    def test_uses_trajectory_derived_initial_frames_for_encoding(
        self, demo_trajectory, boundary_poses, monkeypatch
    ):
        """High-level Fatrop path should encode with trajectory-derived initial frames.

        The CasADi path already relies on `use_default_initial_frames=False` for
        invariance-consistent adaptation. Fatrop should match that contract.
        """
        from dhb_xr.optimization import fatrop_solver

        positions, quaternions = demo_trajectory
        start_pose, goal_pose = boundary_poses
        calls = {}

        def fake_encode(pos, quat, **kwargs):
            calls.update(kwargs)
            n = 30
            return {
                "linear_motion_invariants": np.tile(np.array([[0.01, 0.0, 0.0, 0.0]]), (n, 1)),
                "angular_motion_invariants": np.tile(np.array([[0.0, 0.0, 0.0, 0.0]]), (n, 1)),
                "linear_frame_initial": np.eye(4),
                "angular_frame_initial": np.eye(4),
                "initial_pose": kwargs["init_pose"],
            }

        class DummyGenerator:
            def __init__(self, N, use_fatrop, verbose):
                self.N = N

            def generate(self, **kwargs):
                n = self.N
                return {
                    "positions": np.linspace(start_pose["position"], goal_pose["position"], n),
                    "linear_invariants": np.tile(np.array([[0.01, 0.0, 0.0, 0.0]]), (n - 1, 1)),
                    "angular_invariants": np.tile(np.array([[0.0, 0.0, 0.0, 0.0]]), (n - 1, 1)),
                    "solve_time": 0.0,
                    "success": True,
                }

        def fake_decode(*args, **kwargs):
            n = 29
            return {
                "positions": np.tile(start_pose["position"], (n, 1)),
                "quaternions": np.tile(start_pose["quaternion"], (n, 1)),
            }

        monkeypatch.setattr(fatrop_solver, "encode_dhb_dr", fake_encode)
        monkeypatch.setattr(fatrop_solver, "FatropTrajectoryGenerator", DummyGenerator)
        monkeypatch.setattr(fatrop_solver, "decode_dhb_dr", fake_decode)

        fatrop_solver.generate_trajectory_fatrop(
            positions,
            quaternions,
            start_pose,
            goal_pose,
            traj_length=30,
            use_fatrop=False,
            verbose=False,
        )

        assert calls["use_default_initial_frames"] is False
        assert calls["dhb_method"] == fatrop_solver.DHBMethod.DOUBLE_REFLECTION

    def test_basic_generation(self, demo_trajectory, boundary_poses):
        """Test basic trajectory generation."""
        from dhb_xr.optimization.fatrop_solver import generate_trajectory_fatrop

        positions, quaternions = demo_trajectory
        start_pose, goal_pose = boundary_poses

        result = generate_trajectory_fatrop(
            positions,
            quaternions,
            start_pose,
            goal_pose,
            traj_length=30,
            use_fatrop=False,  # Use IPOPT for reliable testing
            verbose=False,
        )

        # Check result structure
        assert "positions" in result
        assert "quaternions" in result
        assert "linear_invariants" in result
        assert "angular_invariants" in result
        assert "solve_time" in result
        assert "success" in result
        assert "solver" in result

        assert result["solver"] == "ipopt"
        print(f"Solve time: {result['solve_time'] * 1000:.1f} ms")

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
                positions,
                quaternions,
                start_pose,
                goal_pose,
                traj_length=30,
                use_fatrop=False,
            )
            times_ipopt.append(time.perf_counter() - t0)

        print(f"IPOPT: {np.mean(times_ipopt) * 1000:.1f} ms (avg of {len(times_ipopt)} runs)")

        # Try Fatrop
        try:
            times_fatrop = []
            for _ in range(3):
                t0 = time.perf_counter()
                result = generate_trajectory_fatrop(
                    positions,
                    quaternions,
                    start_pose,
                    goal_pose,
                    traj_length=30,
                    use_fatrop=True,
                )
                times_fatrop.append(time.perf_counter() - t0)

            print(
                f"Fatrop: {np.mean(times_fatrop) * 1000:.1f} ms (avg of {len(times_fatrop)} runs)"
            )
            print(f"Speedup: {np.mean(times_ipopt) / np.mean(times_fatrop):.1f}x")
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
            assert hasattr(opt, "generate")
        else:
            pytest.skip("Fatrop not available")

    def test_get_optimizer_ipopt(self):
        """Test getting IPOPT optimizer via factory."""
        from dhb_xr.optimization import get_optimizer

        opt = get_optimizer(backend="ipopt", N=30)
        assert opt is not None
        assert hasattr(opt, "generate")


class TestFatropOrientationSmooth:
    """Tests that Fatrop adaptation produces smooth SE(3) orientations.

    The core claim of DHB-XR is full 6D SE(3) adaptation. These tests verify
    the Fatrop solver produces smooth, continuous quaternion trajectories —
    not discontinuous/spinning orientations.
    """

    @staticmethod
    def _angular_step_degrees(quats_wxyz: np.ndarray) -> np.ndarray:
        """Angular change in degrees between consecutive timesteps."""
        from scipy.spatial.transform import Rotation

        rots = Rotation.from_quat(quats_wxyz[:, [1, 2, 3, 0]])  # wxyz->xyzw
        angles = []
        for t in range(1, len(rots)):
            rel = rots[t - 1].inv() * rots[t]
            angles.append(np.degrees(rel.magnitude()))
        return np.array(angles)

    def test_identity_adaptation_preserves_orientation(self):
        """Identity adaptation (same start/goal) must return smooth orientations.

        When start/goal poses are the SAME as the demo's start/goal, the adapted
        trajectory should closely match the original — especially no spinning.
        Max angular step must be < 5 deg/step.
        """
        from dhb_xr.optimization.fatrop_solver import generate_trajectory_fatrop

        # Create a trajectory with non-trivial orientation (30 deg rotation around Z)
        n = 40
        t = np.linspace(0, 1, n)
        positions = np.column_stack(
            [
                0.1 + 0.3 * t,
                0.5 - 0.1 * t,
                0.9 + 0.1 * np.sin(np.pi * t),
            ]
        )
        angles = np.radians(30) * t
        quaternions = np.zeros((n, 4))
        quaternions[:, 0] = np.cos(angles / 2)
        quaternions[:, 3] = np.sin(angles / 2)

        # Identity adaptation: start/goal = demo start/goal
        result = generate_trajectory_fatrop(
            positions,
            quaternions,
            pose_target_init={"position": positions[0], "quaternion": quaternions[0]},
            pose_target_final={"position": positions[-1], "quaternion": quaternions[-1]},
            traj_length=30,
            use_fatrop=False,  # IPOPT for reliability
        )

        adapted_quat = result["quaternions"]
        steps = self._angular_step_degrees(adapted_quat)
        max_step = steps.max()

        assert result["success"], "Solver failed to converge"
        assert max_step < 5.0, (
            f"Identity adaptation has {max_step:.1f} deg/step max angular change — "
            f"orientations are spinning. Must be < 5 deg/step."
        )
        expected_angles = np.radians(30) * np.linspace(0.0, 1.0, len(adapted_quat))
        expected = np.zeros_like(adapted_quat)
        expected[:, 0] = np.cos(expected_angles / 2.0)
        expected[:, 3] = np.sin(expected_angles / 2.0)
        alignment = np.abs(np.sum(adapted_quat * expected, axis=1))
        maximum_error = float(2.0 * np.arccos(np.clip(alignment, 0.0, 1.0)).max())
        assert maximum_error < np.deg2rad(3.0), (
            "Identity adaptation is smooth but does not preserve the reference "
            f"orientation: maximum error {np.degrees(maximum_error):.2f} deg."
        )

    def test_shifted_goal_produces_smooth_orientation(self):
        """Position-only perturbation must still produce smooth orientations.

        Shift goal position by 10cm in Y (like P2 perturbation) while keeping
        same start/goal orientations. Angular steps must be < 10 deg/step.
        """
        from dhb_xr.optimization.fatrop_solver import generate_trajectory_fatrop

        n = 40
        t = np.linspace(0, 1, n)
        positions = np.column_stack(
            [
                0.1 + 0.3 * t,
                0.5 - 0.1 * t,
                0.9 + 0.1 * np.sin(np.pi * t),
            ]
        )
        angles = np.radians(20) * t
        quaternions = np.zeros((n, 4))
        quaternions[:, 0] = np.cos(angles / 2)
        quaternions[:, 3] = np.sin(angles / 2)

        # Shift goal by 10cm in Y
        shifted_goal = positions[-1].copy()
        shifted_goal[1] += 0.10

        result = generate_trajectory_fatrop(
            positions,
            quaternions,
            pose_target_init={"position": positions[0], "quaternion": quaternions[0]},
            pose_target_final={"position": shifted_goal, "quaternion": quaternions[-1]},
            traj_length=30,
            use_fatrop=False,
        )

        adapted_quat = result["quaternions"]
        steps = self._angular_step_degrees(adapted_quat)
        max_step = steps.max()

        assert max_step < 10.0, (
            f"Shifted-goal adaptation has {max_step:.1f} deg/step max angular change — "
            f"orientations are spinning. Must be < 10 deg/step."
        )

    def test_adapted_quaternions_are_unit(self):
        """All adapted quaternions must have unit norm."""
        from dhb_xr.optimization.fatrop_solver import generate_trajectory_fatrop

        n = 30
        t = np.linspace(0, 1, n)
        positions = np.column_stack([0.3 * t, 0.5 * np.ones(n), 0.9 * np.ones(n)])
        quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))

        result = generate_trajectory_fatrop(
            positions,
            quaternions,
            pose_target_init={"position": positions[0], "quaternion": quaternions[0]},
            pose_target_final={"position": positions[-1], "quaternion": quaternions[-1]},
            traj_length=20,
            use_fatrop=False,
        )

        norms = np.linalg.norm(result["quaternions"], axis=1)
        max_deviation = np.max(np.abs(norms - 1.0))
        assert max_deviation < 1e-4, (
            f"Quaternion norm deviation {max_deviation:.2e} — quaternions are not unit."
        )


class TestFatropTranslationRegression:
    """Regression coverage for pure-translation adaptation quality."""

    def test_pure_translation_stays_close_to_translated_demo(self):
        """Pure translation should preserve the demo path up to a small residual.

        Under a rigid translation, the DHB invariants are unchanged, so the adapted
        trajectory should remain close to the translated demo while still satisfying
        the translated start and goal constraints exactly.
        """
        from scipy.interpolate import interp1d
        from dhb_xr.optimization.fatrop_solver import generate_trajectory_fatrop

        n_demo = 100
        t = np.linspace(0.0, 2.0 * np.pi, n_demo)
        positions = np.column_stack(
            [
                0.30 * np.cos(t),
                0.30 * np.sin(t),
                0.10 * t / (2.0 * np.pi),
            ]
        )
        quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n_demo, 1))

        offset = np.array([0.12, -0.08, 0.05])
        translated_demo = positions + offset
        target_start = {
            "position": translated_demo[0],
            "quaternion": quaternions[0],
        }
        target_goal = {
            "position": translated_demo[-1],
            "quaternion": quaternions[-1],
        }

        result = generate_trajectory_fatrop(
            positions,
            quaternions,
            target_start,
            target_goal,
            traj_length=40,
            use_fatrop=False,  # IPOPT is more reliable in CI than the Fatrop binary
            verbose=False,
        )

        adapted_positions = result["positions"]
        resampled_demo = interp1d(
            np.linspace(0.0, 1.0, len(translated_demo)),
            translated_demo,
            axis=0,
            fill_value="extrapolate",
        )(np.linspace(0.0, 1.0, len(adapted_positions)))

        pointwise_error = np.linalg.norm(adapted_positions - resampled_demo, axis=1)
        rms_error = np.sqrt(np.mean(pointwise_error**2))
        max_error = np.max(pointwise_error)

        np.testing.assert_allclose(
            adapted_positions[0],
            target_start["position"],
            atol=1e-8,
            err_msg="Pure-translation adaptation must satisfy the translated start pose.",
        )
        np.testing.assert_allclose(
            adapted_positions[-1],
            target_goal["position"],
            atol=1e-8,
            err_msg="Pure-translation adaptation must satisfy the translated goal pose.",
        )
        assert rms_error < 0.05, (
            f"Pure-translation adaptation drifted too far from the translated demo path: "
            f"RMS error {rms_error:.3f} m."
        )
        assert max_error < 0.12, (
            f"Pure-translation adaptation deviated too far from the translated demo path: "
            f"max error {max_error:.3f} m."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
