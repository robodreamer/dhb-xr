"""Plot SE(3) trajectories and invariant sequences."""

import numpy as np
from typing import Optional, Dict, Any

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def plot_se3_trajectory(
    positions: np.ndarray,
    quaternions: Optional[np.ndarray] = None,
    ax: Optional[Any] = None,
    title: str = "SE(3) trajectory",
    show_orientation: bool = False,
) -> Any:
    """Plot 3D position trajectory. Optionally show orientation frames at samples."""
    if not HAS_MPL:
        raise ImportError("matplotlib required for plot_se3_trajectory")
    positions = np.asarray(positions)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], "b-", label="position")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    if show_orientation and quaternions is not None:
        from dhb_xr.core import geometry as geom
        step = max(1, len(positions) // 10)
        for i in range(0, len(positions), step):
            R = geom.quat_to_rot(quaternions[i])
            p = positions[i]
            for j, (c, label) in enumerate([("r", "x"), ("g", "y"), ("b", "z")]):
                ax.quiver(p[0], p[1], p[2], R[0, j], R[1, j], R[2, j], color=c, length=0.05)
    return ax


def plot_invariants(
    linear_inv: np.ndarray,
    angular_inv: np.ndarray,
    ax: Optional[Any] = None,
    title: str = "DHB invariants",
) -> Any:
    """Plot linear and angular invariant time series."""
    if not HAS_MPL:
        raise ImportError("matplotlib required for plot_invariants")
    n = linear_inv.shape[0]
    k_lin, k_ang = linear_inv.shape[1], angular_inv.shape[1]
    if ax is None:
        fig, axes = plt.subplots(2, 1, sharex=True)
        ax_lin, ax_ang = axes[0], axes[1]
    else:
        ax_lin, ax_ang = ax
    t = np.arange(n)
    for j in range(k_lin):
        ax_lin.plot(t, linear_inv[:, j], label=f"lin_{j}")
    ax_lin.set_ylabel("linear")
    ax_lin.legend(loc="right", fontsize=8)
    ax_lin.set_title(title)
    for j in range(k_ang):
        ax_ang.plot(t, angular_inv[:, j], label=f"ang_{j}")
    ax_ang.set_ylabel("angular")
    ax_ang.legend(loc="right", fontsize=8)
    ax_ang.set_xlabel("step")
    return (ax_lin, ax_ang)
