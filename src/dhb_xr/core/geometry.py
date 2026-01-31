"""
SO(3) / SE(3) geometry utilities.
Quaternion convention: wxyz (scalar-first).
"""

import numpy as np
from scipy.spatial.transform import Rotation

_EPS = 1e-10


def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """wxyz (scalar first) -> xyzw (scipy convention)."""
    q = np.asarray(q).reshape(4)
    return np.array([q[1], q[2], q[3], q[0]], dtype=q.dtype)


def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """xyzw (scipy) -> wxyz (scalar first)."""
    q = np.asarray(q).reshape(4)
    return np.array([q[3], q[0], q[1], q[2]], dtype=q.dtype)


def quat_to_rot(quat: np.ndarray) -> np.ndarray:
    """
    Quaternion to rotation matrix (SO3).
    quat: (4,) wxyz.
    Returns: (3,3) rotation matrix.
    """
    q = np.asarray(quat).reshape(4)
    r = Rotation.from_quat(_wxyz_to_xyzw(q))
    return r.as_matrix()


def rot_to_quat(rot: np.ndarray) -> np.ndarray:
    """
    Rotation matrix to quaternion (wxyz).
    rot: (3,3).
    Returns: (4,) wxyz.
    """
    R = np.asarray(rot).reshape(3, 3)
    r = Rotation.from_matrix(R)
    return _xyzw_to_wxyz(r.as_quat())


def x_rot(angle: float) -> np.ndarray:
    """Rotation matrix about x-axis."""
    return Rotation.from_euler("x", angle).as_matrix()


def y_rot(angle: float) -> np.ndarray:
    """Rotation matrix about y-axis."""
    return Rotation.from_euler("y", angle).as_matrix()


def z_rot(angle: float) -> np.ndarray:
    """Rotation matrix about z-axis."""
    return Rotation.from_euler("z", angle).as_matrix()


def euler_to_rot(angles: np.ndarray) -> np.ndarray:
    """Euler XYZ (extrinsic) to rotation matrix. angles: (3,) [rx, ry, rz]."""
    a = np.asarray(angles).reshape(3)
    return Rotation.from_euler("xyz", a).as_matrix()


def rot_to_euler(rot: np.ndarray) -> np.ndarray:
    """Rotation matrix to Euler XYZ. Returns (3,) [rx, ry, rz]."""
    R = np.asarray(rot).reshape(3, 3)
    return Rotation.from_matrix(R).as_euler("xyz")


def rot_to_axis_angle(rot: np.ndarray) -> np.ndarray:
    """Rotation matrix to axis-angle (rotation vector). (3,) in rad."""
    R = np.asarray(rot).reshape(3, 3)
    r = Rotation.from_matrix(R)
    return r.as_rotvec()


def axis_angle_to_rot(axis_angle: np.ndarray) -> np.ndarray:
    """Axis-angle to rotation matrix."""
    v = np.asarray(axis_angle).reshape(3)
    return Rotation.from_rotvec(v).as_matrix()


def axis_angle_to_quat(rotvec: np.ndarray) -> np.ndarray:
    """Axis-angle to quaternion (wxyz)."""
    v = np.asarray(rotvec).reshape(3)
    n = np.linalg.norm(v)
    if n < _EPS:
        return np.array([1.0, 0.0, 0.0, 0.0])
    r = Rotation.from_rotvec(v)
    return _xyzw_to_wxyz(r.as_quat())


def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """Quaternion (wxyz) to axis-angle. Single (4,) or batch (N,4) -> (3,) or (N,3)."""
    q = np.asarray(quat)
    if q.ndim == 1:
        q = q.reshape(1, 4)
        single = True
    else:
        single = False
    r = Rotation.from_quat(_wxyz_to_xyzw(q))
    out = r.as_rotvec()
    return out.reshape(-1)[:3] if single else out


def quat_inv(quat: np.ndarray) -> np.ndarray:
    """Inverse of unit quaternion (wxyz). q^{-1} = (w, -x, -y, -z)."""
    q = np.asarray(quat).reshape(4)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=q.dtype)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply quaternions (wxyz): q1 * q2."""
    w1, x1, y1, z1 = np.asarray(q1).reshape(4)
    w2, x2, y2, z2 = np.asarray(q2).reshape(4)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation between two unit quaternions (wxyz).
    t in [0, 1]. Returns (4,) wxyz.
    """
    from scipy.spatial.transform import Slerp
    q0 = np.asarray(q0).reshape(4)
    q1 = np.asarray(q1).reshape(4)
    q1 = quat_ensure_continuous(q1, q0)
    key_times = np.array([0.0, 1.0])
    rots = Rotation.from_quat(np.stack([_wxyz_to_xyzw(q0), _wxyz_to_xyzw(q1)]))
    slerp = Slerp(key_times, rots)
    r = slerp(float(t))
    return _xyzw_to_wxyz(r.as_quat())


def quat_relative_axis_angle(q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
    """
    Axis-angle (3,) of the relative rotation from q_from to q_to:
    R_to @ R_from^T, i.e. q_to * q_from^{-1}. Magnitude is geodesic distance in rad.
    """
    q_rel = quat_multiply(q_to, quat_inv(q_from))
    return quat_to_axis_angle(q_rel)


def quat_ensure_continuous(q_current: np.ndarray, q_prev: np.ndarray) -> np.ndarray:
    """Flip sign of q_current so q_current · q_prev >= 0 (same hemisphere)."""
    qc = np.asarray(q_current).reshape(4)
    qp = np.asarray(q_prev).reshape(4)
    if np.dot(qc, qp) < 0:
        return -qc
    return qc


def bound_angle(angle: float) -> float:
    """Wrap angle to (-pi, pi]."""
    a = float(angle)
    if a > np.pi:
        a -= 2 * np.pi
    elif a <= -np.pi:
        a += 2 * np.pi
    return a
