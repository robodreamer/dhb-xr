import numpy as np
from scipy.spatial.transform import Rotation

from dhb_xr.integration.vla.libero import libero_obs_quat_to_axis_angle, libero_obs_quat_wxyz_to_xyzw


def test_libero_obs_quat_conversion_matches_hdf5_convention():
    """Raw LIBERO env quaternions should be treated as wxyz, not xyzw."""
    target_euler = np.array([-0.00942506, -0.05467911, -0.00308998], dtype=np.float64)
    target_xyzw = Rotation.from_euler("xyz", target_euler, degrees=False).as_quat()

    # Same orientation as above, but in scalar-first wxyz order as observed from LIBERO env.
    obs_quat_wxyz = np.array([0.9995, -0.0005, -0.0298, -0.0040], dtype=np.float64)

    reordered = libero_obs_quat_wxyz_to_xyzw(obs_quat_wxyz)
    rotvec = libero_obs_quat_to_axis_angle(obs_quat_wxyz)

    reordered = reordered / np.linalg.norm(reordered)
    target_xyzw = target_xyzw / np.linalg.norm(target_xyzw)
    angle_err = 2 * np.arccos(np.clip(abs(np.dot(reordered, target_xyzw)), -1.0, 1.0))

    assert np.degrees(angle_err) < 1.0
    np.testing.assert_allclose(
        Rotation.from_rotvec(rotvec).as_euler("xyz", degrees=False),
        Rotation.from_quat(target_xyzw).as_euler("xyz", degrees=False),
        atol=1e-2,
    )
