#!/usr/bin/env python3
"""Show the central DHB-XR idea with one trajectory and one rigid transform.

The example moves an entire SE(3) trajectory into a different world frame. Its
world coordinates change, while its trajectory-relative geometry and DHB-DR
signature remain equal up to floating-point precision.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dhb_xr import encode_dhb_dr
from dhb_xr.core import geometry as geom
from dhb_xr.core.types import DHBMethod, EncodingMethod


def make_demonstration(samples: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """Return a smooth position-and-orientation trajectory."""
    phase = np.linspace(0.0, 1.0, samples)
    positions = np.column_stack(
        (
            0.35 + 0.38 * phase,
            0.12 * np.sin(2.0 * np.pi * phase),
            0.28 + 0.16 * np.sin(np.pi * phase),
        )
    )
    quaternions = np.array(
        [
            geom.rot_to_quat(
                geom.z_rot(0.8 * value)
                @ geom.y_rot(-0.35 * np.sin(np.pi * value))
                @ geom.x_rot(0.20 * np.sin(2.0 * np.pi * value))
            )
            for value in phase
        ]
    )
    return positions, quaternions


def move_world_frame(
    positions: np.ndarray,
    quaternions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one shared rigid transform to every trajectory pose."""
    world_rotation = geom.z_rot(np.deg2rad(65.0)) @ geom.x_rot(np.deg2rad(15.0))
    world_translation = np.array([0.30, -0.38, 0.16])
    moved_positions = (world_rotation @ positions.T).T + world_translation
    moved_quaternions = np.array(
        [
            geom.rot_to_quat(world_rotation @ geom.quat_to_rot(quaternion))
            for quaternion in quaternions
        ]
    )
    return moved_positions, moved_quaternions


def encode_relative(
    positions: np.ndarray,
    quaternions: np.ndarray,
) -> dict[str, np.ndarray]:
    """Encode using a trajectory-relative initial frame."""
    return encode_dhb_dr(
        positions,
        quaternions,
        method=EncodingMethod.POSITION,
        use_default_initial_frames=False,
        dhb_method=DHBMethod.DOUBLE_REFLECTION,
        init_pose={"position": positions[0], "quaternion": quaternions[0]},
        robust_mode=True,
    )


def positions_in_initial_frame(
    positions: np.ndarray,
    quaternions: np.ndarray,
) -> np.ndarray:
    """Express positions relative to the first trajectory pose."""
    initial_rotation = geom.quat_to_rot(quaternions[0])
    return (initial_rotation.T @ (positions - positions[0]).T).T


def run_example(output: Path | None = None) -> dict[str, float]:
    """Run the comparison, optionally writing its publication-style figure."""
    positions, quaternions = make_demonstration()
    moved_positions, moved_quaternions = move_world_frame(positions, quaternions)
    original = encode_relative(positions, quaternions)
    moved = encode_relative(moved_positions, moved_quaternions)

    original_signature = np.column_stack(
        (original["linear_motion_invariants"], original["angular_motion_invariants"])
    )
    moved_signature = np.column_stack(
        (moved["linear_motion_invariants"], moved["angular_motion_invariants"])
    )
    local_positions = positions_in_initial_frame(positions, quaternions)
    moved_local_positions = positions_in_initial_frame(moved_positions, moved_quaternions)
    metrics = {
        "world_position_rms_delta_m": float(np.sqrt(np.mean((positions - moved_positions) ** 2))),
        "initial_frame_position_max_delta_m": float(
            np.max(np.abs(local_positions - moved_local_positions))
        ),
        "dhb_signature_max_delta": float(np.max(np.abs(original_signature - moved_signature))),
    }

    print("DHB-XR frame-invariance example")
    print(f"  World-coordinate RMS change: {metrics['world_position_rms_delta_m']:.3f} m")
    print(
        "  Initial-frame geometry max change: "
        f"{metrics['initial_frame_position_max_delta_m']:.2e} m"
    )
    print(f"  DHB signature max change: {metrics['dhb_signature_max_delta']:.2e}")
    if output is not None:
        plot_overview(
            positions,
            moved_positions,
            quaternions,
            moved_quaternions,
            local_positions,
            moved_local_positions,
            original_signature,
            moved_signature,
            metrics,
            output,
        )
        print(f"  Figure: {output}")
    return metrics


def plot_overview(
    positions: np.ndarray,
    moved_positions: np.ndarray,
    quaternions: np.ndarray,
    moved_quaternions: np.ndarray,
    local_positions: np.ndarray,
    moved_local_positions: np.ndarray,
    original_signature: np.ndarray,
    moved_signature: np.ndarray,
    metrics: dict[str, float],
    output: Path,
) -> None:
    """Render a four-panel scientific overview from the computed arrays."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output = Path(output)
    if output.suffix.lower() not in {".png", ".svg"}:
        raise ValueError("figure output must end in .png or .svg")

    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    original_color = "#0072B2"
    moved_color = "#D55E00"
    fig = plt.figure(figsize=(16.0, 4.6), constrained_layout=True)
    grid = fig.add_gridspec(1, 4, width_ratios=(1.15, 1.15, 1.0, 1.0))
    world_ax = fig.add_subplot(grid[0], projection="3d")
    local_ax = fig.add_subplot(grid[1], projection="3d")
    linear_ax = fig.add_subplot(grid[2])
    angular_ax = fig.add_subplot(grid[3])

    _plot_path(world_ax, positions, original_color, "demonstration")
    _plot_path(world_ax, moved_positions, moved_color, "same motion, moved frame")
    _draw_frame(world_ax, positions[0], geom.quat_to_rot(quaternions[0]))
    _draw_frame(world_ax, moved_positions[0], geom.quat_to_rot(moved_quaternions[0]))
    world_ax.set_title("a  World coordinates change", loc="left", fontweight="bold")
    world_ax.set_xlabel("x (m)")
    world_ax.set_ylabel("y (m)")
    world_ax.set_zlabel("z (m)")
    world_ax.legend(fontsize=8, loc="upper left")
    _equal_3d_axes(world_ax, np.vstack((positions, moved_positions)))

    _plot_path(local_ax, local_positions, original_color, "demonstration", linewidth=4.0)
    _plot_path(
        local_ax,
        moved_local_positions,
        moved_color,
        "moved frame",
        linewidth=2.0,
        linestyle="--",
    )
    _draw_frame(local_ax, np.zeros(3), np.eye(3))
    local_ax.set_title("b  Initial-frame geometry overlaps", loc="left", fontweight="bold")
    local_ax.set_xlabel("local x (m)")
    local_ax.set_ylabel("local y (m)")
    local_ax.set_zlabel("local z (m)")
    local_ax.legend(fontsize=8, loc="upper left")
    local_ax.text2D(
        0.03,
        0.84,
        f"max Δ = {metrics['initial_frame_position_max_delta_m']:.1e} m",
        transform=local_ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.9},
    )
    _equal_3d_axes(local_ax, np.vstack((local_positions, moved_local_positions)))

    phase = np.linspace(0.0, 1.0, original_signature.shape[0])
    _plot_signature_channels(
        linear_ax,
        phase,
        original_signature[:, :4],
        moved_signature[:, :4],
        colors,
    )
    linear_ax.set_title("c  Linear DHB channels overlap", loc="left", fontweight="bold")
    linear_ax.set_ylabel("invariant value")

    _plot_signature_channels(
        angular_ax,
        phase,
        original_signature[:, 4:],
        moved_signature[:, 4:],
        colors,
    )
    angular_ax.set_title("d  Angular DHB channels overlap", loc="left", fontweight="bold")
    angular_ax.text(
        0.03,
        0.04,
        f"max Δ across all channels = {metrics['dhb_signature_max_delta']:.1e}",
        transform=angular_ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.9},
    )
    style_legend = [
        Line2D([0], [0], color="#444444", linewidth=2.0, label="demonstration"),
        Line2D(
            [0],
            [0],
            color="#444444",
            linewidth=1.5,
            linestyle="--",
            label="moved world frame",
        ),
    ]
    angular_ax.legend(handles=style_legend, fontsize=8, loc="upper right")

    fig.suptitle(
        "DHB-XR: the world frame moves; the motion signature does not\n"
        "One shared rigid transform (65° rotation + translation) is applied to every pose",
        fontsize=16,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_path(
    axis,
    positions: np.ndarray,
    color: str,
    label: str,
    *,
    linewidth: float = 2.5,
    linestyle: str = "-",
) -> None:
    axis.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
    )
    axis.scatter(*positions[0], color=color, s=35, marker="o")
    axis.scatter(*positions[-1], color=color, s=55, marker="*")


def _draw_frame(axis, origin: np.ndarray, rotation: np.ndarray, scale: float = 0.07) -> None:
    for column, color in enumerate(("#D55E00", "#009E73", "#0072B2")):
        direction = rotation[:, column] * scale
        axis.quiver(*origin, *direction, color=color, linewidth=1.2, arrow_length_ratio=0.22)


def _equal_3d_axes(axis, points: np.ndarray) -> None:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = 0.5 * (minimum + maximum)
    radius = 0.55 * np.max(maximum - minimum)
    radius = max(float(radius), 0.1)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1.0, 1.0, 1.0))


def _plot_signature_channels(
    axis,
    phase: np.ndarray,
    original: np.ndarray,
    moved: np.ndarray,
    colors: tuple[str, ...],
) -> None:
    for channel, color in enumerate(colors):
        axis.plot(
            phase,
            original[:, channel],
            color=color,
            linewidth=2.0,
            label=rf"$u_{channel}$",
        )
        axis.plot(
            phase,
            moved[:, channel],
            color=color,
            linewidth=1.2,
            linestyle="--",
        )
    axis.set_xlabel("normalized trajectory phase")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncols=2, loc="upper left")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/dhb-xr-frame-invariance.svg"),
        help="PNG or SVG figure path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_example(args.output)


if __name__ == "__main__":
    main()
