"""Plot SE(3) trajectories and invariant sequences with cubes and coordinate frames."""

import numpy as np
from typing import Optional, Dict, Any, List, Union

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Axis colors: x=red, y=green, z=blue
AXIS_COLORS = ["r", "g", "b"]


def draw_box(
    ax,
    center: np.ndarray,
    rotation_matrix: np.ndarray,
    size: tuple = (0.05, 0.05, 0.05),
    color: str = "blue",
    alpha: float = 0.3,
) -> None:
    """
    Draw a 3D oriented box (cuboid) at the given center with the given rotation.

    Args:
        ax: Matplotlib 3D axes
        center: (3,) position of box center
        rotation_matrix: (3, 3) rotation matrix
        size: (width, height, depth) of box
        color: Face color
        alpha: Transparency
    """
    # Cuboid vertices before rotation (unit cube centered at origin)
    vertices = np.array([
        [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],  # Bottom
        [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5]  # Top
    ], dtype=np.float64)

    # Scale and rotate vertices
    vertices = vertices * np.array(size)
    vertices = np.dot(vertices, rotation_matrix.T)
    vertices += center

    # Define the indices of vertices for each face of the cuboid
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]]
    ]

    # Create a Poly3DCollection object
    face_collection = Poly3DCollection(faces, facecolors=color, linewidths=1, edgecolors='k', alpha=alpha)
    ax.add_collection3d(face_collection)


def draw_frame(
    ax,
    position: np.ndarray,
    rotation_matrix: np.ndarray,
    length: float = 0.05,
    linewidth: float = 2.0,
) -> None:
    """
    Draw a coordinate frame (x=red, y=green, z=blue arrows) at the given pose.

    Args:
        ax: Matplotlib 3D axes
        position: (3,) position of frame origin
        rotation_matrix: (3, 3) rotation matrix
        length: Length of arrows
        linewidth: Line width
    """
    for j, c in enumerate(AXIS_COLORS):
        ax.quiver(
            position[0], position[1], position[2],
            rotation_matrix[0, j], rotation_matrix[1, j], rotation_matrix[2, j],
            color=c, length=length, normalize=True, linewidth=linewidth
        )


def plot_se3_trajectory(
    positions: np.ndarray,
    quaternions: Optional[np.ndarray] = None,
    ax: Optional[Any] = None,
    title: str = "SE(3) trajectory",
    show_orientation: bool = False,
    vis_type: str = "arrow",
    box_size: tuple = (0.03, 0.03, 0.03),
    color: str = "b",
    alpha: float = 0.7,
    num_frames: int = 8,
    label: Optional[str] = None,
) -> Any:
    """
    Plot 3D position trajectory with optional orientation visualization.

    Args:
        positions: (N, 3) position array
        quaternions: (N, 4) quaternion array (wxyz), optional
        ax: Matplotlib 3D axes, created if None
        title: Plot title
        show_orientation: If True, show orientation frames or cubes
        vis_type: "arrow" for coordinate frames, "cube" for oriented boxes
        box_size: Size of cubes (if vis_type="cube")
        color: Trajectory line color
        alpha: Transparency for cubes
        num_frames: Number of orientation samples to show along trajectory
        label: Legend label for trajectory

    Returns:
        Matplotlib axes
    """
    if not HAS_MPL:
        raise ImportError("matplotlib required for plot_se3_trajectory")
    positions = np.asarray(positions)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], f"{color}-", label=label or "position", alpha=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)

    if show_orientation and quaternions is not None:
        from dhb_xr.core import geometry as geom
        n = len(positions)
        indices = np.linspace(0, n - 1, num_frames, dtype=int)
        for i in indices:
            R = geom.quat_to_rot(quaternions[i])
            p = positions[i]
            if vis_type == "cube":
                draw_box(ax, p, R, size=box_size, color=color, alpha=alpha)
            else:  # arrow
                draw_frame(ax, p, R, length=0.05)
    return ax


def plot_se3_trajectories(
    trajectories: Union[List[Dict], Dict[str, Dict]],
    ax: Optional[Any] = None,
    show_orientation: bool = True,
    vis_type: str = "cube",
    box_size_scale: float = 0.05,
    num_frames: int = 6,
    title: str = "SE(3) Trajectories",
    show_legend: bool = True,
) -> Any:
    """
    Plot multiple SE(3) trajectories with orientation visualization.

    Args:
        trajectories: List of dicts with 'positions' and 'quaternions', or dict of name -> trajectory
        ax: Matplotlib 3D axes
        show_orientation: Show orientation cubes/frames
        vis_type: "cube" or "arrow"
        box_size_scale: Size of cubes
        num_frames: Number of orientation samples per trajectory
        title: Plot title
        show_legend: Show legend

    Returns:
        Matplotlib axes
    """
    if not HAS_MPL:
        raise ImportError("matplotlib required for plot_se3_trajectories")

    # Normalize input to dict
    if isinstance(trajectories, list):
        trajectories = {f"traj_{i}": t for i, t in enumerate(trajectories)}

    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

    colors = ["blue", "green", "red", "orange", "purple", "cyan", "magenta", "brown"]
    box_size = (box_size_scale, box_size_scale, box_size_scale)

    for idx, (name, traj) in enumerate(trajectories.items()):
        positions = np.asarray(traj["positions"])
        quaternions = np.asarray(traj.get("quaternions"))
        color = colors[idx % len(colors)]

        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color=color, label=name, linewidth=2)

        if show_orientation and quaternions is not None:
            from dhb_xr.core import geometry as geom
            n = len(positions)
            indices = np.linspace(0, n - 1, num_frames, dtype=int)
            for i in indices:
                R = geom.quat_to_rot(quaternions[i])
                p = positions[i]
                if vis_type == "cube":
                    draw_box(ax, p, R, size=box_size, color=color, alpha=0.5)
                else:
                    draw_frame(ax, p, R, length=0.05)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    if show_legend:
        ax.legend()

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
