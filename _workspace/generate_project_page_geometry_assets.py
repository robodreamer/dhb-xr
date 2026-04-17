#!/usr/bin/env python3
"""Regenerate task-grounded geometry media for the project page.

This script replaces the older abstract trajectory assets with scene-grounded
visuals derived from real LIBERO end-effector demonstrations. The public-facing
figures use the exact decode-only translation mode for the strongest currently
verified claim: pure translation re-grounding from a shifted scene placement.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from mpl_toolkits.mplot3d import art3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline
from scipy.spatial.transform import Rotation

from dhb_xr.encoder.dhb_dr import encode_dhb_dr
from dhb_xr.decoder.dhb_dr import decode_dhb_dr
from dhb_xr.core import geometry as geom
from dhb_xr.core.types import DHBMethod, EncodingMethod
from dhb_xr.optimization.fatrop_solver import generate_trajectory_fatrop
from dhb_xr.visualization.plot import draw_frame


matplotlib.rcParams.update(
    {
        "figure.dpi": 180,
        "savefig.dpi": 180,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path("/home/andypark/Projects/data/libero/libero_spatial")
BDDL_DIR = Path("/home/andypark/Projects/repos/LIBERO-PRO/libero/libero/bddl_files/libero_spatial")
ASSET_DIR = ROOT / "docs" / "paper" / "assets"

FIG6_PATH = ASSET_DIR / "fig6_3d_trajectory_adaptation.png"
FIG7_PATH = ASSET_DIR / "fig7_6d_adaptation_vs_affine.png"
HERO_PATH = ASSET_DIR / "hero_3d_adaptation.mp4"
DETAIL_PATH = ASSET_DIR / "demo_3d_adaptation_detail.mp4"
TEASER_PATH = ASSET_DIR / "teaser.png"
META_PATH = ROOT / "_workspace" / "project_page_geometry_assets.json"

BG = "#0f172a"
PANEL_BG = "#111c35"
GRID = "#94a3b8"
DEMO = "#cbd5e1"
DHB = "#60a5fa"
AFFINE = "#fb7185"
TABLE = "#152238"
TARGET = "#22c55e"
CONTEXT = "#f59e0b"
SOURCE = "#38bdf8"
TEXT = "#e5eefc"
MUTED = "#9fb2cf"


@dataclass(frozen=True)
class TaskSpec:
    name: str
    short_label: str
    source_region: str
    target_region: str
    context_region: str | None
    start_delta: tuple[float, float, float]
    goal_delta: tuple[float, float, float]
    camera: tuple[float, float]


TASKS = [
    TaskSpec(
        name="pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        short_label="Between Plate / Ramekin",
        source_region="between_plate_ramekin_region",
        target_region="plate_region",
        context_region="ramekin_region",
        start_delta=(0.03, -0.025, 0.0),
        goal_delta=(0.055, 0.01, 0.0),
        camera=(24, -58),
    ),
    TaskSpec(
        name="pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
        short_label="Next To Cookie Box",
        source_region="next_to_box_region",
        target_region="plate_region",
        context_region="box_region",
        start_delta=(-0.025, 0.03, 0.0),
        goal_delta=(0.04, 0.04, 0.0),
        camera=(28, -48),
    ),
    TaskSpec(
        name="pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        short_label="From Stove To Plate",
        source_region="stove_region",
        target_region="plate_region",
        context_region=None,
        start_delta=(0.045, 0.03, 0.0),
        goal_delta=(0.07, -0.015, 0.0),
        camera=(26, -40),
    ),
]


def load_demo(task_name: str) -> tuple[np.ndarray, np.ndarray]:
    demo_path = DATA_DIR / f"{task_name}_demo.hdf5"
    with h5py.File(demo_path, "r") as f:
        demo_key = sorted(k for k in f["data"].keys() if "demo" in k)[0]
        pos = np.array(f["data"][demo_key]["obs"]["ee_pos"])
        euler = np.array(f["data"][demo_key]["obs"]["ee_ori"])
    xyzw = Rotation.from_euler("xyz", euler).as_quat()
    quat = np.column_stack([xyzw[:, 3], xyzw[:, 0], xyzw[:, 1], xyzw[:, 2]])
    return pos, quat


def parse_regions(task_name: str) -> dict[str, dict[str, np.ndarray]]:
    bddl_text = (BDDL_DIR / f"{task_name}.bddl").read_text()
    blocks = re.findall(r"\(([\w_]+)\s*\n\s*\(:target [^)]+\)\s*\n\s*\(:ranges \(\s*\(\s*([^)]+)\)\s*\)\s*\)\s*(?:\n\s*\(:yaw_rotation \([^)]+\)\s*\))?\s*\)", bddl_text)
    regions: dict[str, dict[str, np.ndarray]] = {}
    for name, values in blocks:
        nums = np.array([float(v) for v in values.split()], dtype=float)
        if len(nums) != 4:
            continue
        x0, y0, x1, y1 = nums
        regions[name] = {
            "center": np.array([(x0 + x1) / 2, (y0 + y1) / 2], dtype=float),
            "size": np.array([abs(x1 - x0), abs(y1 - y0)], dtype=float),
            "bounds": np.array([x0, y0, x1, y1], dtype=float),
        }
    return regions


def adapt_demo(
    positions: np.ndarray,
    quaternions: np.ndarray,
    start_delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    init_pose = {"position": positions[0], "quaternion": quaternions[0]}
    encoded = encode_dhb_dr(
        positions,
        quaternions,
        init_pose=init_pose,
        method=EncodingMethod.POSITION,
        use_default_initial_frames=True,
        dhb_method=DHBMethod.DOUBLE_REFLECTION,
    )
    decoded = decode_dhb_dr(
        encoded["linear_motion_invariants"],
        encoded["angular_motion_invariants"],
        {"position": positions[0] + start_delta, "quaternion": quaternions[0]},
        method=EncodingMethod.POSITION,
        dhb_method=DHBMethod.DOUBLE_REFLECTION,
        drop_padded=True,
    )
    return decoded["positions"], decoded["quaternions"]


def make_rect(bounds: np.ndarray, z: float) -> list[list[float]]:
    x0, y0, x1, y1 = bounds
    return [
        [x0, y0, z],
        [x1, y0, z],
        [x1, y1, z],
        [x0, y1, z],
    ]


def set_dark_3d(ax) -> None:
    ax.set_facecolor(PANEL_BG)
    ax.xaxis.set_pane_color((0.07, 0.12, 0.22, 0.96))
    ax.yaxis.set_pane_color((0.07, 0.12, 0.22, 0.96))
    ax.zaxis.set_pane_color((0.07, 0.12, 0.22, 0.96))
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"]["color"] = (0.75, 0.81, 0.91, 0.28)
        axis._axinfo["tick"]["color"] = TEXT
        axis._axinfo["axisline"]["color"] = MUTED
    ax.tick_params(colors=TEXT, labelsize=7)
    ax.title.set_color(TEXT)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.zaxis.label.set_color(MUTED)


def set_light_3d(ax) -> None:
    ax.set_facecolor("white")
    ax.xaxis.set_pane_color((0.98, 0.99, 1.0, 1.0))
    ax.yaxis.set_pane_color((0.98, 0.99, 1.0, 1.0))
    ax.zaxis.set_pane_color((0.98, 0.99, 1.0, 1.0))
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"]["color"] = (0.6, 0.65, 0.75, 0.18)
        axis._axinfo["tick"]["color"] = "#334155"
        axis._axinfo["axisline"]["color"] = "#94a3b8"
    ax.tick_params(colors="#334155", labelsize=8)
    ax.title.set_color("#0f172a")
    ax.xaxis.label.set_color("#475569")
    ax.yaxis.label.set_color("#475569")
    ax.zaxis.label.set_color("#475569")


def equalize(ax, positions: np.ndarray) -> None:
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    centers = (mins + maxs) / 2
    radius = max(maxs - mins) / 2
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius * 0.8, centers[2] + radius * 0.8)


def draw_reference_plane(ax, positions: np.ndarray, color: str, alpha: float, z_offset: float = 0.02) -> None:
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    pad_xy = 0.03
    z = float(mins[2] - z_offset)
    plane = Poly3DCollection(
        [[
            [mins[0] - pad_xy, mins[1] - pad_xy, z],
            [maxs[0] + pad_xy, mins[1] - pad_xy, z],
            [maxs[0] + pad_xy, maxs[1] + pad_xy, z],
            [mins[0] - pad_xy, maxs[1] + pad_xy, z],
        ]],
        facecolors=color,
        edgecolors=(0.65, 0.7, 0.8, 0.18),
        linewidths=0.6,
        alpha=alpha,
    )
    ax.add_collection3d(plane)


def draw_workspace(ax, regions: dict[str, dict[str, np.ndarray]], table_z: float, task: TaskSpec) -> None:
    all_x = []
    all_y = []
    for region in regions.values():
        x0, y0, x1, y1 = region["bounds"]
        all_x.extend([x0, x1])
        all_y.extend([y0, y1])
    pad = 0.05
    x0, x1 = min(all_x) - pad, max(all_x) + pad
    y0, y1 = min(all_y) - pad, max(all_y) + pad
    plane = Poly3DCollection(
        [[[x0, y0, table_z], [x1, y0, table_z], [x1, y1, table_z], [x0, y1, table_z]]],
        facecolors=TABLE,
        edgecolors=(0.85, 0.9, 1.0, 0.08),
        linewidths=0.5,
        alpha=0.85,
    )
    ax.add_collection3d(plane)

    def add_region(name: str, color: str, alpha: float, label: str) -> None:
        if name not in regions:
            return
        rect = Poly3DCollection(
            [make_rect(regions[name]["bounds"], table_z + 0.002)],
            facecolors=color,
            edgecolors=color,
            linewidths=1.1,
            alpha=alpha,
        )
        ax.add_collection3d(rect)
        center = regions[name]["center"]
        ax.text(
            center[0],
            center[1],
            table_z + 0.008,
            label,
            color=color,
            fontsize=7,
            ha="center",
            va="bottom",
        )

    add_region(task.source_region, SOURCE, 0.18, "source")
    add_region(task.target_region, TARGET, 0.2, "plate")
    if task.context_region:
        add_region(task.context_region, CONTEXT, 0.16, "context")


def add_key_frames(ax, pos: np.ndarray, quat: np.ndarray, color: str) -> None:
    idxs = np.linspace(0, len(pos) - 1, 5, dtype=int)
    for idx in idxs:
        rot = Rotation.from_quat([quat[idx, 1], quat[idx, 2], quat[idx, 3], quat[idx, 0]]).as_matrix()
        draw_frame(ax, pos[idx], rot, length=0.03, linewidth=1.2)


def build_task_payload(task: TaskSpec) -> dict[str, np.ndarray | dict]:
    demo_pos, demo_quat = load_demo(task.name)
    regions = parse_regions(task.name)
    adapted_pos, adapted_quat = adapt_demo(
        demo_pos,
        demo_quat,
        np.array(task.start_delta),
    )
    raw_replay = demo_pos[: len(adapted_pos)]
    demo_shifted = demo_pos[: len(adapted_pos)] + np.array(task.start_delta)
    return {
        "demo_pos": demo_pos,
        "demo_quat": demo_quat,
        "adapted_pos": adapted_pos,
        "adapted_quat": adapted_quat,
        "raw_replay": raw_replay,
        "demo_shifted": demo_shifted,
        "regions": regions,
    }


def endpoint_warp_positions(
    positions: np.ndarray,
    start_new: np.ndarray,
    goal_new: np.ndarray,
    n: int,
) -> np.ndarray:
    """Naive endpoint-only warp baseline."""
    pts = interp1d(
        np.linspace(0.0, 1.0, len(positions)),
        positions,
        axis=0,
        fill_value="extrapolate",
    )(np.linspace(0.0, 1.0, n))
    t = np.linspace(0.0, 1.0, n)[:, None]
    return pts + (1.0 - t) * (start_new - pts[0]) + t * (goal_new - pts[-1])


def smooth_curve(values: np.ndarray, n: int = 240) -> tuple[np.ndarray, np.ndarray]:
    """Cubic-spline smoothing for presentation plots."""
    values = np.asarray(values, dtype=np.float64)
    t = np.linspace(0.0, 1.0, len(values))
    t_smooth = np.linspace(0.0, 1.0, n)
    k = min(3, len(values) - 1)
    if values.ndim == 1:
        smoothed = make_interp_spline(t, values, k=k)(t_smooth)
    else:
        smoothed = np.column_stack(
            [make_interp_spline(t, values[:, i], k=k)(t_smooth) for i in range(values.shape[1])]
        )
    return t_smooth, smoothed


def quat_progress_deg(quats: np.ndarray) -> np.ndarray:
    """Angle from the initial orientation over the trajectory."""
    quats = np.asarray(quats, dtype=np.float64)
    q0 = quats[0]
    out = []
    for q in quats:
        rvec = geom.quat_relative_axis_angle(q0, q)
        out.append(float(np.degrees(np.linalg.norm(rvec))))
    return np.asarray(out, dtype=np.float64)


def build_controlled_6d_case() -> dict[str, np.ndarray | float | str]:
    """Controlled 6D retargeting benchmark using the repaired Fatrop path."""
    n = 50
    t = np.linspace(0.0, 1.0, n)
    demo_pos = np.column_stack(
        [
            0.34 * t,
            0.10 * np.sin(np.pi * t) + 0.05 * np.sin(2.0 * np.pi * t),
            0.05 + 0.18 * t + 0.04 * np.sin(np.pi * t) ** 2,
        ]
    )
    demo_quat = []
    for s in t:
        yaw = 0.6 * np.sin(np.pi * s)
        pitch = -0.35 * s
        qy = np.array([np.cos(pitch / 2.0), 0.0, np.sin(pitch / 2.0), 0.0])
        qz = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
        demo_quat.append(geom.quat_multiply(qz, qy))
    demo_quat = np.asarray(demo_quat, dtype=np.float64)

    q_init_rot = np.array([np.cos(np.pi / 8.0), 0.0, 0.0, np.sin(np.pi / 8.0)])
    q_goal_rot = np.array([np.cos(-np.pi / 10.0), 0.0, np.sin(-np.pi / 10.0), 0.0])
    pose_init = {
        "position": demo_pos[0] + np.array([0.08, -0.06, 0.03]),
        "quaternion": geom.quat_multiply(q_init_rot, demo_quat[0]),
    }
    pose_goal = {
        "position": demo_pos[-1] + np.array([0.16, 0.07, 0.07]),
        "quaternion": geom.quat_multiply(q_goal_rot, demo_quat[-1]),
    }

    result = generate_trajectory_fatrop(
        demo_pos,
        demo_quat,
        pose_init,
        pose_goal,
        traj_length=36,
        use_fatrop=False,
        verbose=False,
    )

    fatrop_pos = np.asarray(result["positions"], dtype=np.float64)
    fatrop_quat = np.asarray(result["quaternions"], dtype=np.float64)
    demo_rs = interp1d(
        np.linspace(0.0, 1.0, len(demo_pos)),
        demo_pos,
        axis=0,
        fill_value="extrapolate",
    )(np.linspace(0.0, 1.0, len(fatrop_pos)))
    affine_pos = endpoint_warp_positions(
        demo_pos,
        pose_init["position"],
        pose_goal["position"],
        len(fatrop_pos),
    )
    slerp_quat = np.array(
        [
            geom.quat_slerp(
                pose_init["quaternion"],
                pose_goal["quaternion"],
                i / max(1, len(fatrop_quat) - 1),
            )
            for i in range(len(fatrop_quat))
        ]
    )

    deviation = np.linalg.norm(fatrop_pos - affine_pos, axis=1)
    path_len_demo = float(np.sum(np.linalg.norm(np.diff(demo_rs, axis=0), axis=1)))
    path_len_fatrop = float(np.sum(np.linalg.norm(np.diff(fatrop_pos, axis=0), axis=1)))
    path_len_affine = float(np.sum(np.linalg.norm(np.diff(affine_pos, axis=0), axis=1)))
    goal_quat_absdot = float(abs(np.dot(fatrop_quat[-1], pose_goal["quaternion"])))

    demo_quat_rs = np.array(
        [
            geom.quat_slerp(
                demo_quat[0],
                demo_quat[-1],
                i / max(1, len(fatrop_quat) - 1),
            )
            for i in range(len(fatrop_quat))
        ]
    )

    return {
        "demo_pos": demo_rs,
        "fatrop_pos": fatrop_pos,
        "affine_pos": affine_pos,
        "fatrop_quat": fatrop_quat,
        "demo_quat": demo_quat_rs,
        "slerp_quat": slerp_quat,
        "fatrop_progress_deg": quat_progress_deg(fatrop_quat),
        "slerp_progress_deg": quat_progress_deg(slerp_quat),
        "demo_progress_deg": quat_progress_deg(demo_quat_rs),
        "deviation": deviation,
        "mean_deviation_cm": float(deviation.mean() * 100.0),
        "max_deviation_cm": float(deviation.max() * 100.0),
        "path_len_demo": path_len_demo,
        "path_len_fatrop": path_len_fatrop,
        "path_len_affine": path_len_affine,
        "goal_quat_absdot": goal_quat_absdot,
        "goal_quat_error": float(1.0 - goal_quat_absdot),
        "solver": str(result["solver"]),
    }


def save_fig6(payloads: dict[str, dict[str, np.ndarray | dict]]) -> None:
    fig = plt.figure(figsize=(15.5, 5.4), facecolor=BG)
    plt.subplots_adjust(left=0.03, right=0.99, top=0.88, bottom=0.08, wspace=0.08)
    for i, task in enumerate(TASKS, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        data = payloads[task.name]
        demo_pos = data["demo_pos"]
        adapted_pos = data["adapted_pos"]
        adapted_quat = data["adapted_quat"]
        raw_replay = data["raw_replay"]
        combined = np.vstack([raw_replay, adapted_pos])
        table_z = float(combined[:, 2].min() - 0.11)
        set_dark_3d(ax)
        draw_workspace(ax, data["regions"], table_z, task)
        ax.plot(raw_replay[:, 0], raw_replay[:, 1], raw_replay[:, 2], color=DEMO, linewidth=2.0, linestyle="--", alpha=0.68, label="raw replay")
        ax.plot(adapted_pos[:, 0], adapted_pos[:, 1], adapted_pos[:, 2], color=DHB, linewidth=2.8, alpha=0.96, label="DHB re-grounded")
        ax.scatter(*adapted_pos[0], s=45, c=SOURCE, marker="o", depthshade=False)
        ax.scatter(*adapted_pos[-1], s=65, c=TARGET, marker="*", depthshade=False)
        peak = adapted_pos[np.argmax(adapted_pos[:, 2])]
        ax.scatter(*peak, s=22, c="#f8fafc", marker="D", depthshade=False)
        add_key_frames(ax, adapted_pos, adapted_quat, DHB)
        equalize(ax, combined)
        elev, azim = task.camera
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(task.short_label, pad=10, fontsize=11, color=TEXT)
        ax.text2D(
            0.04,
            0.92,
            f"offset ({task.start_delta[0]*100:+.0f}, {task.start_delta[1]*100:+.0f}) cm",
            transform=ax.transAxes,
            color=MUTED,
            fontsize=7,
        )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
    fig.suptitle(
        "Scene-grounded DHB re-grounding on real LIBERO demonstrations",
        color=TEXT,
        fontsize=16,
        y=0.97,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.03,
        "Dashed gray = replay at the original scene location. Blue = DHB decode from the shifted initial pose.",
        ha="center",
        color=MUTED,
        fontsize=10,
    )
    fig.savefig(FIG6_PATH, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def save_fig7(payload: dict[str, np.ndarray | dict], task: TaskSpec) -> None:
    del payload, task
    case = build_controlled_6d_case()
    demo_pos = case["demo_pos"]
    fatrop_pos = case["fatrop_pos"]
    affine_pos = case["affine_pos"]
    _, demo_pos_s = smooth_curve(demo_pos)
    _, fatrop_pos_s = smooth_curve(fatrop_pos)
    _, affine_pos_s = smooth_curve(affine_pos)
    all_pos = np.vstack([demo_pos_s, fatrop_pos_s, affine_pos_s])
    mid_idx = len(fatrop_pos) // 2
    max_lateral_gap_cm = float(np.max(np.abs(fatrop_pos[:, 1] - affine_pos[:, 1])) * 100.0)
    max_vertical_gap_cm = float(np.max(np.abs(fatrop_pos[:, 2] - affine_pos[:, 2])) * 100.0)
    mid_gap_cm = float(np.linalg.norm(fatrop_pos[mid_idx] - affine_pos[mid_idx]) * 100.0)

    fig = plt.figure(figsize=(14.0, 7.1), facecolor="white")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.7, 1.0], height_ratios=[1.0, 1.0], wspace=0.16, hspace=0.20)

    ax_main = fig.add_subplot(gs[:, 0], projection="3d")
    set_light_3d(ax_main)
    draw_reference_plane(ax_main, all_pos, color="#e2e8f0", alpha=0.35)
    ax_main.plot(
        demo_pos_s[:, 0],
        demo_pos_s[:, 1],
        demo_pos_s[:, 2],
        color="#94a3b8",
        linewidth=2.2,
        linestyle="--",
        alpha=0.95,
        label="demo trajectory",
    )
    ax_main.plot(
        affine_pos_s[:, 0],
        affine_pos_s[:, 1],
        affine_pos_s[:, 2],
        color="#f59e0b",
        linewidth=2.6,
        linestyle=(0, (5, 2)),
        alpha=0.98,
        label="endpoint-only warp",
    )
    ax_main.plot(
        fatrop_pos_s[:, 0],
        fatrop_pos_s[:, 1],
        fatrop_pos_s[:, 2],
        color="#2563eb",
        linewidth=3.2,
        alpha=0.98,
        label="DHB retarget",
    )
    ax_main.scatter(*fatrop_pos[0], c="#2563eb", s=52, marker="o", depthshade=False)
    ax_main.scatter(*fatrop_pos[-1], c="#16a34a", s=80, marker="*", depthshade=False)
    start_rot = Rotation.from_quat(
        [case["fatrop_quat"][0, 1], case["fatrop_quat"][0, 2], case["fatrop_quat"][0, 3], case["fatrop_quat"][0, 0]]
    ).as_matrix()
    goal_rot = Rotation.from_quat(
        [case["fatrop_quat"][-1, 1], case["fatrop_quat"][-1, 2], case["fatrop_quat"][-1, 3], case["fatrop_quat"][-1, 0]]
    ).as_matrix()
    draw_frame(ax_main, fatrop_pos[0], start_rot, length=0.03, linewidth=1.4)
    draw_frame(ax_main, fatrop_pos[-1], goal_rot, length=0.03, linewidth=1.4)
    equalize(ax_main, all_pos)
    ax_main.view_init(elev=28, azim=-54)
    ax_main.set_title("3D comparison", pad=12, fontsize=14, fontweight="bold")
    ax_main.set_xlabel("x (m)")
    ax_main.set_ylabel("y (m)")
    ax_main.set_zlabel("z (m)")
    ax_main.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=8.2, framealpha=0.96)
    ax_main.text2D(
        0.04,
        0.89,
        "Both methods match the new start and goal poses.",
        transform=ax_main.transAxes,
        fontsize=9.5,
        color="#334155",
    )
    ax_main.text2D(
        0.04,
        0.84,
        "The difference is the interior 6D motion geometry.",
        transform=ax_main.transAxes,
        fontsize=9.5,
        color="#334155",
    )

    ax_side = fig.add_subplot(gs[0, 1])
    ax_side.plot(demo_pos_s[:, 1], demo_pos_s[:, 2], color="#94a3b8", linewidth=2.0, linestyle="--", alpha=0.95, label="demo")
    ax_side.plot(
        affine_pos_s[:, 1],
        affine_pos_s[:, 2],
        color="#f59e0b",
        linewidth=2.4,
        linestyle=(0, (5, 2)),
        alpha=0.98,
        label="endpoint-only warp",
    )
    ax_side.plot(fatrop_pos_s[:, 1], fatrop_pos_s[:, 2], color="#2563eb", linewidth=2.9, alpha=0.98, label="DHB retarget")
    ax_side.scatter(fatrop_pos[mid_idx, 1], fatrop_pos[mid_idx, 2], c="#2563eb", s=28, zorder=5)
    ax_side.scatter(affine_pos[mid_idx, 1], affine_pos[mid_idx, 2], c="#f59e0b", s=28, zorder=5)
    ax_side.annotate(
        "lift and side-sweep preserved",
        xy=(fatrop_pos[mid_idx, 1], fatrop_pos[mid_idx, 2]),
        xytext=(fatrop_pos[mid_idx, 1] - 0.11, fatrop_pos[mid_idx, 2] + 0.04),
        arrowprops=dict(arrowstyle="->", color="#2563eb", lw=1.4),
        fontsize=8.8,
        color="#1d4ed8",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#eff6ff", edgecolor="#93c5fd"),
    )
    ax_side.annotate(
        "interior path collapses",
        xy=(affine_pos[mid_idx, 1], affine_pos[mid_idx, 2]),
        xytext=(affine_pos[mid_idx, 1] + 0.02, affine_pos[mid_idx, 2] - 0.05),
        arrowprops=dict(arrowstyle="->", color="#b45309", lw=1.4),
        fontsize=8.6,
        color="#92400e",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#fffbeb", edgecolor="#fcd34d"),
    )
    ax_side.set_title("side view (y-z)", fontsize=12.5, fontweight="bold", pad=8)
    ax_side.set_xlabel("y (m)")
    ax_side.set_ylabel("z (m)")
    ax_side.grid(alpha=0.18)

    ax_top = fig.add_subplot(gs[1, 1])
    ax_top.plot(demo_pos_s[:, 0], demo_pos_s[:, 1], color="#94a3b8", linewidth=2.0, linestyle="--", alpha=0.95)
    ax_top.plot(affine_pos_s[:, 0], affine_pos_s[:, 1], color="#f59e0b", linewidth=2.4, linestyle=(0, (5, 2)), alpha=0.98)
    ax_top.plot(fatrop_pos_s[:, 0], fatrop_pos_s[:, 1], color="#2563eb", linewidth=2.9, alpha=0.98)
    ax_top.scatter(fatrop_pos[0, 0], fatrop_pos[0, 1], c="#2563eb", s=28, zorder=5)
    ax_top.scatter(fatrop_pos[-1, 0], fatrop_pos[-1, 1], c="#16a34a", s=48, marker="*", zorder=5)
    ax_top.set_title("top view (x-y)", fontsize=12.5, fontweight="bold", pad=8)
    ax_top.set_xlabel("x (m)")
    ax_top.set_ylabel("y (m)")
    ax_top.grid(alpha=0.18)
    ax_top.text(0.08, 0.88, "same new start", transform=ax_top.transAxes, ha="left", va="center", fontsize=8.7, color="#1d4ed8", fontweight="bold", bbox=dict(boxstyle="round,pad=0.28", facecolor="#eff6ff", edgecolor="#bfdbfe"))
    ax_top.text(0.40, 0.88, "same new goal", transform=ax_top.transAxes, ha="left", va="center", fontsize=8.7, color="#166534", fontweight="bold", bbox=dict(boxstyle="round,pad=0.28", facecolor="#f0fdf4", edgecolor="#86efac"))
    ax_top.text(0.69, 0.88, f"{mid_gap_cm:.1f} cm interior gap", transform=ax_top.transAxes, ha="left", va="center", fontsize=8.6, color="#9a3412", fontweight="bold", bbox=dict(boxstyle="round,pad=0.28", facecolor="#fff7ed", edgecolor="#fdba74"))
    ax_top.text(
        0.03,
        0.05,
        f"goal orientation error {case['goal_quat_error']:.1e}   |   max lateral gap {max_lateral_gap_cm:.1f} cm   |   max vertical gap {max_vertical_gap_cm:.1f} cm",
        transform=ax_top.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.6,
        color="#475569",
    )

    fig.suptitle("Controlled 6D DHB Retargeting", fontsize=17, fontweight="bold", y=0.955)
    fig.text(
        0.5,
        0.91,
        "Both methods satisfy the new boundary poses. DHB retargeting preserves the interior lifted arc and lateral geometry instead of collapsing toward an endpoint-only warp.",
        ha="center",
        va="center",
        fontsize=9.6,
        color="#475569",
    )
    fig.subplots_adjust(top=0.86, left=0.06, right=0.98, bottom=0.08)
    fig.savefig(FIG7_PATH, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def render_canvas(fig) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return data.reshape(height, width, 4)[..., :3].copy()


def save_hero_video(payloads: dict[str, dict[str, np.ndarray | dict]]) -> None:
    del payloads
    case = build_controlled_6d_case()
    _, demo_pos_s = smooth_curve(case["demo_pos"], n=220)
    _, fatrop_pos_s = smooth_curve(case["fatrop_pos"], n=220)
    _, affine_pos_s = smooth_curve(case["affine_pos"], n=220)
    all_pos = np.vstack([demo_pos_s, fatrop_pos_s, affine_pos_s])

    frames = []
    steps = 84
    fig = plt.figure(figsize=(12.8, 4.8), facecolor="#0b1220")
    for frame_idx in range(steps):
        fig.clf()
        idx = max(2, int((frame_idx + 1) / steps * len(fatrop_pos_s)))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        set_dark_3d(ax)
        draw_reference_plane(ax, all_pos, color="#1e293b", alpha=0.62)
        ax.plot(demo_pos_s[:, 0], demo_pos_s[:, 1], demo_pos_s[:, 2], color=DEMO, linewidth=1.8, linestyle="--", alpha=0.42)
        ax.plot(affine_pos_s[:, 0], affine_pos_s[:, 1], affine_pos_s[:, 2], color="#f59e0b", linewidth=2.1, linestyle=(0, (5, 2)), alpha=0.92)
        ax.plot(fatrop_pos_s[:, 0], fatrop_pos_s[:, 1], fatrop_pos_s[:, 2], color=DHB, linewidth=2.3, alpha=0.16)
        ax.plot(fatrop_pos_s[:idx, 0], fatrop_pos_s[:idx, 1], fatrop_pos_s[:idx, 2], color=DHB, linewidth=3.2, alpha=0.98)
        ax.scatter(*fatrop_pos_s[idx - 1], s=38, c="#f8fafc", depthshade=False)
        ax.scatter(*fatrop_pos_s[0], s=34, c=DHB, depthshade=False)
        ax.scatter(*fatrop_pos_s[-1], s=62, c=TARGET, marker="*", depthshade=False)
        equalize(ax, all_pos)
        ax.view_init(elev=28, azim=-60 + frame_idx * 0.33)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.text2D(0.05, 0.90, "DHB retarget", transform=ax.transAxes, color=DHB, fontsize=11, fontweight="bold")
        ax.text2D(0.05, 0.84, "endpoint-only warp", transform=ax.transAxes, color="#f59e0b", fontsize=10)
        ax.text2D(0.05, 0.78, "demo trajectory", transform=ax.transAxes, color=DEMO, fontsize=10)
        ax.text2D(0.05, 0.14, "same boundary poses", transform=ax.transAxes, color=TEXT, fontsize=9)
        ax.text2D(0.05, 0.09, "different interior geometry", transform=ax.transAxes, color=MUTED, fontsize=8.5)

        fig.suptitle(
            "Controlled 6D geometric retargeting",
            color="#f8fafc",
            fontsize=17,
            fontweight="bold",
            y=0.955,
        )
        fig.text(
            0.5,
            0.04,
            "DHB preserves the interior 6D motion pattern while endpoint-only warping collapses the middle of the trajectory.",
            ha="center",
            color="#94a3b8",
            fontsize=10,
        )
        frames.append(render_canvas(fig))
    plt.close(fig)
    imageio.mimwrite(HERO_PATH, frames, fps=18, codec="libx264", quality=7)


def save_detail_video(payload: dict[str, np.ndarray | dict], task: TaskSpec) -> None:
    demo_pos = payload["demo_pos"]
    adapted_pos = payload["adapted_pos"]
    adapted_quat = payload["adapted_quat"]
    raw_replay = payload["raw_replay"]
    demo_shifted = payload["demo_shifted"]
    regions = payload["regions"]
    demo_progress = np.linspace(0.0, 1.0, len(raw_replay))
    adapted_progress = np.linspace(0.0, 1.0, len(adapted_pos))
    shifted_progress = np.linspace(0.0, 1.0, len(demo_shifted))
    table_z = float(np.vstack([raw_replay, adapted_pos, demo_shifted])[:, 2].min() - 0.11)
    clearance_floor = table_z + 0.055

    frames = []
    steps = len(adapted_pos)
    fig = plt.figure(figsize=(12.8, 5.0), facecolor=BG)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1.0, 1.0], wspace=0.18, hspace=0.24)
    for frame_idx in range(steps):
        fig.clf()
        grid = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1.0, 1.0], wspace=0.18, hspace=0.24)
        idx = max(2, frame_idx + 1)

        ax3d = fig.add_subplot(grid[:, 0], projection="3d")
        set_dark_3d(ax3d)
        draw_workspace(ax3d, regions, table_z, task)
        ax3d.plot(raw_replay[:, 0], raw_replay[:, 1], raw_replay[:, 2], color=DEMO, linewidth=1.8, linestyle="--", alpha=0.42)
        ax3d.plot(adapted_pos[:idx, 0], adapted_pos[:idx, 1], adapted_pos[:idx, 2], color=DHB, linewidth=3.0)
        ax3d.plot(demo_shifted[:idx, 0], demo_shifted[:idx, 1], demo_shifted[:idx, 2], color="#16a34a", linewidth=2.0, linestyle=":")
        add_key_frames(ax3d, adapted_pos[:idx], adapted_quat[:idx], DHB)
        equalize(ax3d, np.vstack([raw_replay, adapted_pos, demo_shifted]))
        ax3d.view_init(elev=24, azim=-62 + frame_idx * 0.25)
        ax3d.set_title("3D retargeted transfer", color=TEXT, pad=10, fontsize=12)
        ax3d.set_xlabel("x")
        ax3d.set_ylabel("y")
        ax3d.set_zlabel("z")

        ax_top = fig.add_subplot(grid[0, 1])
        for name, color, alpha in [
            (task.source_region, SOURCE, 0.18),
            (task.target_region, TARGET, 0.18),
            (task.context_region, CONTEXT, 0.16),
        ]:
            if not name or name not in regions:
                continue
            x0, y0, x1, y1 = regions[name]["bounds"]
            ax_top.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=color, edgecolor=color, alpha=alpha))
        ax_top.plot(raw_replay[:, 0], raw_replay[:, 1], color=DEMO, linewidth=1.5, linestyle="--", alpha=0.55)
        ax_top.plot(adapted_pos[:idx, 0], adapted_pos[:idx, 1], color=DHB, linewidth=2.2)
        ax_top.plot(demo_shifted[:idx, 0], demo_shifted[:idx, 1], color="#16a34a", linewidth=1.9, linestyle=":")
        ax_top.scatter(*adapted_pos[idx - 1, :2], s=26, c="#f8fafc")
        ax_top.set_title("Top-down workspace", color=TEXT, fontsize=11)
        ax_top.set_facecolor(PANEL_BG)
        ax_top.tick_params(colors=MUTED)
        for spine in ax_top.spines.values():
            spine.set_color("#2f4366")
        ax_top.grid(alpha=0.15)
        ax_top.set_xlabel("x", color=MUTED)
        ax_top.set_ylabel("y", color=MUTED)

        ax_h = fig.add_subplot(grid[1, 1])
        ax_h.set_facecolor(PANEL_BG)
        ax_h.axhspan(table_z, clearance_floor, color="#7f1d1d", alpha=0.28)
        ax_h.plot(demo_progress, raw_replay[:, 2], color=DEMO, linewidth=1.5, linestyle="--", alpha=0.55, label="raw replay")
        ax_h.plot(adapted_progress, adapted_pos[:, 2], color=DHB, linewidth=2.2, label="DHB")
        ax_h.plot(shifted_progress, demo_shifted[:, 2], color="#16a34a", linewidth=1.9, linestyle=":", label="translated demo")
        ax_h.axvline(adapted_progress[idx - 1], color="#f8fafc", linewidth=1.0, alpha=0.95)
        ax_h.set_title("Height profile", color=TEXT, fontsize=11)
        ax_h.tick_params(colors=MUTED)
        for spine in ax_h.spines.values():
            spine.set_color("#2f4366")
        ax_h.grid(alpha=0.15)
        ax_h.set_xlabel("progress", color=MUTED)
        ax_h.set_ylabel("z", color=MUTED)
        ax_h.legend(loc="lower right", fontsize=7, framealpha=0.85)

        fig.suptitle("Why the geometry matters: the new path follows the shifted scene instead of replaying the old one", color=TEXT, fontsize=15, fontweight="bold", y=0.96)
        frames.append(render_canvas(fig))
    plt.close(fig)
    imageio.mimwrite(DETAIL_PATH, frames, fps=18, codec="libx264", quality=7)


def draw_round_box(
    ax,
    xy: tuple[float, float],
    wh: tuple[float, float],
    fc: str,
    ec: str | None = None,
    text: str = "",
    text_color: str = "#0f172a",
    fontsize: int = 11,
    weight: str = "bold",
):
    from matplotlib.patches import FancyBboxPatch

    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.03",
        linewidth=1.2,
        edgecolor=ec or fc,
        facecolor=fc,
    )
    ax.add_patch(patch)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=text_color, fontsize=fontsize, fontweight=weight)
    return patch


def save_teaser(payload: dict[str, np.ndarray | dict], task: TaskSpec) -> None:
    adapted_pos = payload["adapted_pos"]
    raw_replay = payload["raw_replay"]
    demo_shifted = payload["demo_shifted"]
    regions = payload["regions"]
    case = build_controlled_6d_case()
    _, demo_pos_s = smooth_curve(case["demo_pos"])
    _, fatrop_pos_s = smooth_curve(case["fatrop_pos"])
    _, affine_pos_s = smooth_curve(case["affine_pos"])

    fig = plt.figure(figsize=(14.6, 5.6), facecolor="white")
    outer = fig.add_gridspec(1, 3, width_ratios=[1.1, 0.95, 1.15], wspace=0.20)

    axA = fig.add_subplot(outer[0, 0])
    axA.set_title("1. Standard tokens drift", fontsize=15, fontweight="bold", pad=10)
    for name, color, label, alpha in [
        (task.context_region, CONTEXT, "context", 0.12),
        (task.source_region, SOURCE, "source", 0.15),
        (task.target_region, TARGET, "target", 0.15),
    ]:
        if not name or name not in regions:
            continue
        x0, y0, x1, y1 = regions[name]["bounds"]
        axA.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=color, edgecolor=color, alpha=alpha, linewidth=1.4))
        axA.text((x0 + x1) / 2, y1 + 0.012, label, color=color, fontsize=8.5, ha="center")
    axA.plot(raw_replay[:, 0], raw_replay[:, 1], color="#94a3b8", linewidth=2.2, linestyle="--", alpha=0.9, label="original workspace")
    axA.plot(demo_shifted[:, 0], demo_shifted[:, 1], color="#ef4444", linewidth=2.4, alpha=0.9, label="shifted workspace")
    axA.scatter(*raw_replay[0, :2], c="#94a3b8", s=28, zorder=5)
    axA.scatter(*demo_shifted[0, :2], c="#ef4444", s=28, zorder=5)
    axA.set_xlabel("x (m)")
    axA.set_ylabel("y (m)")
    axA.grid(alpha=0.18)
    axA.set_aspect("equal", adjustable="box")
    axA.legend(loc="lower left", fontsize=7.2, framealpha=0.95)
    axA.text(0.30, 0.885, "18–94% token change", transform=axA.transAxes, ha="center", va="center", fontsize=10.2, color="#b91c1c", fontweight="bold", bbox=dict(boxstyle="round,pad=0.35", facecolor="#fef2f2", edgecolor="#fecaca"))
    axA.text(0.30, 0.835, "raw coords and delta-EEF drift", transform=axA.transAxes, ha="center", va="center", fontsize=8.3, color="#7f1d1d")
    axA.text(
        0.03,
        0.04,
        "Moving the workspace changes the action targets\nunless the representation is invariant.",
        transform=axA.transAxes,
        fontsize=8.8,
        color="#334155",
        ha="left",
        va="bottom",
    )

    innerB = outer[0, 1].subgridspec(3, 1, height_ratios=[0.9, 0.9, 1.1], hspace=0.22)
    axB0 = fig.add_subplot(innerB[0, 0])
    axB0.axis("off")
    axB0.set_title("2. DHB encodes motion shape", fontsize=15, fontweight="bold", pad=10)
    draw_round_box(axB0, (0.06, 0.50), (0.30, 0.24), "#e2e8f0", text="demo trajectory", text_color="#0f172a", fontsize=11)
    draw_round_box(axB0, (0.42, 0.46), (0.20, 0.32), "#dbeafe", text="DHB\nframes", text_color="#1d4ed8", fontsize=12)
    draw_round_box(axB0, (0.69, 0.50), (0.24, 0.24), "#dcfce7", text="tokens", text_color="#166534", fontsize=11)
    arrow_kw = dict(arrowstyle="-|>", lw=1.6, color="#64748b")
    axB0.annotate("", xy=(0.42, 0.62), xytext=(0.36, 0.62), arrowprops=arrow_kw)
    axB0.annotate("", xy=(0.69, 0.62), xytext=(0.62, 0.62), arrowprops=arrow_kw)

    axB1 = fig.add_subplot(innerB[1, 0])
    axB1.set_facecolor("#f8fafc")
    lin = np.asarray([0.18, 0.30, 0.44, 0.56, 0.51, 0.39, 0.28, 0.22])
    ang = np.asarray([0.12, 0.18, 0.27, 0.34, 0.36, 0.31, 0.24, 0.17])
    axB1.imshow(lin[None, :], cmap="Blues", aspect="auto", extent=[0, 8, 0.55, 0.90], vmin=0.0, vmax=0.60)
    axB1.imshow(ang[None, :], cmap="Greens", aspect="auto", extent=[0, 8, 0.10, 0.45], vmin=0.0, vmax=0.45)
    for x in range(9):
        axB1.axvline(x, color="white", linewidth=1.0, alpha=0.9)
    axB1.text(0.15, 0.73, "linear speed invariants", color="#1d4ed8", fontsize=9.2, fontweight="bold")
    axB1.text(0.15, 0.27, "angular speed invariants", color="#166534", fontsize=9.2, fontweight="bold")
    axB1.set_xlim(0, 8)
    axB1.set_ylim(0, 1)
    axB1.set_xticks([])
    axB1.set_yticks([])
    for spine in axB1.spines.values():
        spine.set_visible(False)

    axB2 = fig.add_subplot(innerB[2, 0])
    axB2.axis("off")
    draw_round_box(axB2, (0.06, 0.54), (0.38, 0.25), "#ecfdf5", ec="#86efac")
    axB2.text(0.25, 0.67, "0.0% token change", ha="center", va="center", fontsize=13, color="#166534", fontweight="bold")
    axB2.text(0.25, 0.58, "stable under SE(3)", ha="center", va="center", fontsize=8.8, color="#166534")
    draw_round_box(axB2, (0.54, 0.54), (0.34, 0.25), "#eff6ff", ec="#bfdbfe")
    axB2.text(0.71, 0.67, "49.8×", ha="center", va="center", fontsize=13, color="#1d4ed8", fontweight="bold")
    axB2.text(0.71, 0.58, "compression with FAST", ha="center", va="center", fontsize=8.8, color="#1d4ed8")
    axB2.text(
        0.06,
        0.18,
        "DHB removes workspace pose from the token target,\nso the same skill stays token-identical after rigid transforms.",
        ha="left",
        va="bottom",
        fontsize=8.8,
        color="#334155",
    )

    axC = fig.add_subplot(outer[0, 2], projection="3d")
    set_light_3d(axC)
    draw_reference_plane(axC, np.vstack([demo_pos_s, fatrop_pos_s, affine_pos_s]), color="#e2e8f0", alpha=0.35)
    axC.set_title("3. Decode from a new state", fontsize=15, fontweight="bold", pad=10)
    axC.plot(demo_pos_s[:, 0], demo_pos_s[:, 1], demo_pos_s[:, 2], color="#94a3b8", linewidth=2.0, linestyle="--", alpha=0.9)
    axC.plot(affine_pos_s[:, 0], affine_pos_s[:, 1], affine_pos_s[:, 2], color="#f59e0b", linewidth=2.3, linestyle=(0, (5, 2)), alpha=0.96)
    axC.plot(fatrop_pos_s[:, 0], fatrop_pos_s[:, 1], fatrop_pos_s[:, 2], color="#2563eb", linewidth=3.0, alpha=0.98)
    axC.scatter(*case["fatrop_pos"][0], c="#2563eb", s=38, depthshade=False)
    axC.scatter(*case["fatrop_pos"][-1], c="#16a34a", s=62, marker="*", depthshade=False)
    equalize(axC, np.vstack([demo_pos_s, fatrop_pos_s, affine_pos_s]))
    axC.view_init(elev=28, azim=-54)
    axC.set_xlabel("")
    axC.set_ylabel("")
    axC.set_zlabel("")
    axC.set_xticks([])
    axC.set_yticks([])
    axC.set_zticks([])
    axC.text2D(0.05, 0.88, "same endpoints", transform=axC.transAxes, fontsize=9.6, color="#334155")
    axC.text2D(0.05, 0.82, "different interior path", transform=axC.transAxes, fontsize=9.6, color="#334155")
    axC.text2D(0.05, 0.12, "DHB preserves interior geometry", transform=axC.transAxes, fontsize=9.0, color="#1d4ed8", fontweight="bold")
    axC.text2D(0.05, 0.07, "endpoint-only warping collapses the middle", transform=axC.transAxes, fontsize=8.4, color="#92400e")

    fig.suptitle("DHB-XR separates motion shape from workspace pose", fontsize=18, fontweight="bold", y=0.95)
    fig.text(
        0.5,
        0.90,
        "Standard token targets drift under rigid transforms. DHB invariants stay fixed and decode into the same 6-DOF motion pattern from a new state.",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#475569",
    )
    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.05, right=0.98)
    fig.savefig(TEASER_PATH, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    payloads = {task.name: build_task_payload(task) for task in TASKS}
    save_fig6(payloads)
    save_fig7(payloads[TASKS[0].name], TASKS[0])
    save_hero_video(payloads)
    save_detail_video(payloads[TASKS[0].name], TASKS[0])
    save_teaser(payloads[TASKS[0].name], TASKS[0])
    META_PATH.write_text(
        json.dumps(
            {
                "tasks": [task.name for task in TASKS],
                "outputs": [
                    str(TEASER_PATH.relative_to(ROOT)),
                    str(FIG6_PATH.relative_to(ROOT)),
                    str(FIG7_PATH.relative_to(ROOT)),
                    str(HERO_PATH.relative_to(ROOT)),
                    str(DETAIL_PATH.relative_to(ROOT)),
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
