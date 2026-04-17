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
from scipy.spatial.transform import Rotation

from dhb_xr.encoder.dhb_dr import encode_dhb_dr
from dhb_xr.decoder.dhb_dr import decode_dhb_dr
from dhb_xr.core.types import DHBMethod, EncodingMethod
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


def equalize(ax, positions: np.ndarray) -> None:
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    centers = (mins + maxs) / 2
    radius = max(maxs - mins) / 2
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius * 0.8, centers[2] + radius * 0.8)


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
    demo_pos = payload["demo_pos"]
    adapted_pos = payload["adapted_pos"]
    raw_replay = payload["raw_replay"]
    demo_shifted = payload["demo_shifted"]
    regions = payload["regions"]
    demo_progress = np.linspace(0.0, 1.0, len(raw_replay))
    adapted_progress = np.linspace(0.0, 1.0, len(adapted_pos))
    shifted_progress = np.linspace(0.0, 1.0, len(demo_shifted))
    table_z = float(np.vstack([raw_replay, adapted_pos, demo_shifted])[:, 2].min() - 0.11)
    clearance_floor = table_z + 0.055

    fig = plt.figure(figsize=(13.6, 4.8), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.0], wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("Workspace Transfer View", fontsize=13, fontweight="bold")
    for name, color, label, alpha in [
        (task.source_region, SOURCE, "source bowl", 0.18),
        (task.target_region, TARGET, "target plate", 0.18),
        (task.context_region, CONTEXT, "context object", 0.16),
    ]:
        if not name or name not in regions:
            continue
        x0, y0, x1, y1 = regions[name]["bounds"]
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=color, edgecolor=color, alpha=alpha, linewidth=1.5)
        ax1.add_patch(rect)
        cx, cy = regions[name]["center"]
        ax1.text(cx, cy + 0.015, label, color=color, fontsize=8, ha="center")
    ax1.plot(raw_replay[:, 0], raw_replay[:, 1], color=DEMO, linewidth=2.0, linestyle="--", label="raw replay")
    ax1.plot(adapted_pos[:, 0], adapted_pos[:, 1], color=DHB, linewidth=2.6, label="DHB re-grounded")
    ax1.plot(demo_shifted[:, 0], demo_shifted[:, 1], color="#16a34a", linewidth=2.0, linestyle=":", label="translated demo")
    ax1.scatter(*adapted_pos[0, :2], c=SOURCE, s=55, zorder=5)
    ax1.scatter(*adapted_pos[-1, :2], c=TARGET, s=70, marker="*", zorder=5)
    ax1.annotate(
        "object moved",
        xy=adapted_pos[0, :2],
        xytext=(adapted_pos[0, 0] + 0.07, adapted_pos[0, 1] - 0.04),
        arrowprops=dict(arrowstyle="->", color="#334155", lw=1.5),
        fontsize=8,
        color="#334155",
    )
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.grid(alpha=0.18)
    ax1.legend(loc="lower left", fontsize=8, framealpha=0.92)
    ax1.set_aspect("equal", adjustable="box")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("Lift Clearance Over Time", fontsize=13, fontweight="bold")
    ax2.axhspan(table_z, clearance_floor, color="#fee2e2", alpha=0.6, label="obstacle / rim zone")
    ax2.plot(demo_progress, raw_replay[:, 2], color=DEMO, linewidth=2.0, linestyle="--", label="raw replay")
    ax2.plot(adapted_progress, adapted_pos[:, 2], color=DHB, linewidth=2.6, label="DHB re-grounded")
    ax2.plot(shifted_progress, demo_shifted[:, 2], color="#16a34a", linewidth=2.0, linestyle=":", label="translated demo")
    ax2.set_xlabel("normalized progress")
    ax2.set_ylabel("height z (m)")
    ax2.grid(alpha=0.2)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title("Corridor Side View", fontsize=13, fontweight="bold")
    ax3.fill_between([0, 1], [table_z, table_z], [clearance_floor, clearance_floor], color="#fee2e2", alpha=0.6)
    ax3.plot(raw_replay[:, 1], raw_replay[:, 2], color=DEMO, linewidth=2.0, linestyle="--", label="raw replay")
    ax3.plot(adapted_pos[:, 1], adapted_pos[:, 2], color=DHB, linewidth=2.6, label="DHB re-grounded")
    ax3.plot(demo_shifted[:, 1], demo_shifted[:, 2], color="#16a34a", linewidth=2.0, linestyle=":", label="translated demo")
    ax3.text(
        adapted_pos[:, 1].mean(),
        clearance_floor + 0.008,
        "DHB tracks the shifted translation target; raw replay stays in the old corridor",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#7f1d1d",
    )
    ax3.set_xlabel("y corridor (m)")
    ax3.set_ylabel("height z (m)")
    ax3.grid(alpha=0.2)

    fig.suptitle(
        "DHB re-grounding follows translated scene placement; raw replay remains tied to the original workspace",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.savefig(FIG7_PATH, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def render_canvas(fig) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return data.reshape(height, width, 4)[..., :3].copy()


def save_hero_video(payloads: dict[str, dict[str, np.ndarray | dict]]) -> None:
    frames = []
    steps = 72
    fig = plt.figure(figsize=(12.4, 4.4), facecolor=BG)
    for frame_idx in range(steps):
        fig.clf()
        plt.subplots_adjust(left=0.03, right=0.99, top=0.84, bottom=0.09, wspace=0.08)
        progress_idx = max(2, int((frame_idx + 1) / steps * 72))
        for i, task in enumerate(TASKS, start=1):
            ax = fig.add_subplot(1, 3, i, projection="3d")
            data = payloads[task.name]
            demo_pos = data["demo_pos"]
            adapted_pos = data["adapted_pos"]
            adapted_quat = data["adapted_quat"]
            combined = np.vstack([demo_pos, adapted_pos])
            table_z = float(combined[:, 2].min() - 0.11)
            set_dark_3d(ax)
            draw_workspace(ax, data["regions"], table_z, task)
            ax.plot(demo_pos[:, 0], demo_pos[:, 1], demo_pos[:, 2], color=DEMO, linewidth=1.6, linestyle="--", alpha=0.32)
            ax.plot(adapted_pos[:progress_idx, 0], adapted_pos[:progress_idx, 1], adapted_pos[:progress_idx, 2], color=DHB, linewidth=2.8)
            tip = adapted_pos[progress_idx - 1]
            ax.scatter(*tip, s=48, c="#f8fafc", marker="o", depthshade=False)
            if frame_idx > 14:
                add_key_frames(ax, adapted_pos[:progress_idx], adapted_quat[:progress_idx], DHB)
            equalize(ax, combined)
            elev, azim = task.camera
            ax.view_init(elev=elev, azim=azim - 14 + frame_idx * 0.22)
            ax.set_title(task.short_label, pad=8, fontsize=10, color=TEXT)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.text2D(0.04, 0.92, "DHB retarget", transform=ax.transAxes, color=DHB, fontsize=7, fontweight="bold")
        fig.suptitle(
            "Single demonstrations, re-grounded to new scene placements",
            color=TEXT,
            fontsize=16,
            fontweight="bold",
            y=0.95,
        )
        fig.text(
            0.5,
            0.03,
            "Real LIBERO end-effector trajectories with solver-backed retargeting.",
            ha="center",
            color=MUTED,
            fontsize=10,
        )
        frames.append(render_canvas(fig))
    plt.close(fig)
    imageio.mimwrite(HERO_PATH, frames, fps=16, codec="libx264", quality=7)


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

    fig = plt.figure(figsize=(14.5, 5.1), facecolor="white")
    outer = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.95, 1.05], wspace=0.22)

    axA = fig.add_subplot(outer[0, 0])
    axA.set_title("(a) Scene-grounded geometric retargeting", fontsize=16, fontweight="bold", pad=10)
    for name, color, label, alpha in [
        (task.context_region, CONTEXT, "context", 0.14),
        (task.source_region, SOURCE, "source bowl", 0.16),
        (task.target_region, TARGET, "target plate", 0.16),
    ]:
        if not name or name not in regions:
            continue
        x0, y0, x1, y1 = regions[name]["bounds"]
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=color, edgecolor=color, alpha=alpha, linewidth=1.6)
        axA.add_patch(rect)
        axA.text((x0 + x1) / 2, y1 + 0.014, label, color=color, fontsize=9, ha="center")
    axA.plot(raw_replay[:, 0], raw_replay[:, 1], color="#cbd5e1", linewidth=2.5, linestyle="--", label="raw replay")
    axA.plot(adapted_pos[:, 0], adapted_pos[:, 1], color=DHB, linewidth=3.2, label="DHB adapted")
    axA.plot(demo_shifted[:, 0], demo_shifted[:, 1], color="#16a34a", linewidth=2.2, linestyle=":", label="translated demo")
    axA.scatter(*adapted_pos[0, :2], c=SOURCE, s=70, zorder=5)
    axA.scatter(*adapted_pos[-1, :2], c=TARGET, s=110, marker="*", zorder=5)
    axA.annotate(
        "new scene placement",
        xy=adapted_pos[0, :2],
        xytext=(adapted_pos[0, 0] - 0.02, adapted_pos[0, 1] - 0.035),
        arrowprops=dict(arrowstyle="->", color="#334155", lw=1.5),
        fontsize=9,
        color="#334155",
    )
    axA.text(
        0.02,
        0.04,
        "Real LIBERO bowl-to-plate demo,\nused here as an exact translation diagnostic",
        transform=axA.transAxes,
        fontsize=9,
        color="#334155",
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#f8fafc", edgecolor="#cbd5e1"),
    )
    axA.set_xlabel("x (m)")
    axA.set_ylabel("y (m)")
    axA.grid(alpha=0.18)
    axA.legend(loc="lower left", fontsize=8, framealpha=0.95)
    axA.set_aspect("equal", adjustable="box")

    innerB = outer[0, 1].subgridspec(2, 1, hspace=0.34)
    axB1 = fig.add_subplot(innerB[0, 0])
    axB1.set_title("(b) Stable tokens, better compression", fontsize=16, fontweight="bold", pad=10)
    labels = ["FAST raw", "delta-EEF", "DHB + FAST"]
    instability = [93.9, 18.2, 0.0]
    colors = ["#ef4444", "#f59e0b", "#22c55e"]
    bars = axB1.bar(labels, instability, color=colors, width=0.62)
    axB1.set_ylabel("% token IDs changed")
    axB1.set_ylim(0, 100)
    axB1.axhline(0, color="#64748b", linewidth=0.8)
    axB1.grid(axis="y", alpha=0.2)
    for bar, val in zip(bars, instability):
        axB1.text(bar.get_x() + bar.get_width() / 2, val + 3, f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axB1.text(0.98, 0.92, "40 random SE(3) transforms", transform=axB1.transAxes, ha="right", va="top", fontsize=8, color="#475569")

    axB2 = fig.add_subplot(innerB[1, 0])
    comp_labels = ["FAST raw", "DHB + FAST", "DHB + OAT"]
    compression = [20.4, 49.8, 10.0]
    comp_colors = ["#94a3b8", "#2563eb", "#7c3aed"]
    bars2 = axB2.bar(comp_labels, compression, color=comp_colors, width=0.62)
    axB2.set_ylabel("compression ratio")
    axB2.grid(axis="y", alpha=0.2)
    axB2.set_ylim(0, 56)
    for bar, val in zip(bars2, compression):
        axB2.text(bar.get_x() + bar.get_width() / 2, val + 1.1, f"{val:.1f}x", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axB2.text(0.03, 0.93, "Same skill, fewer tokens", transform=axB2.transAxes, ha="left", va="top", fontsize=8, color="#475569")

    axC = fig.add_subplot(outer[0, 2])
    axC.set_title("(c) State-aware DHB action decoding", fontsize=16, fontweight="bold", pad=10)
    axC.set_xlim(0, 1)
    axC.set_ylim(0, 1)
    axC.axis("off")
    axC.text(0.03, 0.92, "Instead of regressing delta actions directly,", fontsize=10, color="#334155", ha="left")
    axC.text(0.03, 0.87, "predict invariant motion then decode relative to the current EE state.", fontsize=10, color="#334155", ha="left")

    draw_round_box(axC, (0.05, 0.68), (0.27, 0.12), "#e2e8f0", text="obs + language", text_color="#0f172a")
    draw_round_box(axC, (0.37, 0.68), (0.22, 0.12), "#dbeafe", text="DHB tokens", text_color="#1d4ed8")
    draw_round_box(axC, (0.64, 0.68), (0.27, 0.12), "#dcfce7", text="current EE state", text_color="#166534")
    draw_round_box(axC, (0.21, 0.46), (0.56, 0.12), "#ede9fe", text="DHBActionHead predicts invariant motion pattern", text_color="#5b21b6", fontsize=10)
    draw_round_box(axC, (0.25, 0.23), (0.48, 0.12), "#eff6ff", text="decode relative to present robot pose", text_color="#1d4ed8", fontsize=10)
    draw_round_box(axC, (0.33, 0.05), (0.34, 0.10), "#f8fafc", ec="#cbd5e1", text="robot action", text_color="#0f172a", fontsize=11)

    arrow_kw = dict(arrowstyle="-|>", lw=1.7, color="#64748b")
    axC.annotate("", xy=(0.49, 0.58), xytext=(0.19, 0.68), arrowprops=arrow_kw)
    axC.annotate("", xy=(0.49, 0.58), xytext=(0.48, 0.68), arrowprops=arrow_kw)
    axC.annotate("", xy=(0.49, 0.58), xytext=(0.78, 0.68), arrowprops=arrow_kw)
    axC.annotate("", xy=(0.49, 0.35), xytext=(0.49, 0.46), arrowprops=arrow_kw)
    axC.annotate("", xy=(0.50, 0.15), xytext=(0.49, 0.23), arrowprops=arrow_kw)

    draw_round_box(axC, (0.69, 0.25), (0.24, 0.15), "#f8fafc", ec="#bfdbfe")
    axC.text(0.81, 0.355, "Current readout", ha="center", va="center", fontsize=9, color="#1e3a8a", fontweight="bold")
    axC.text(0.81, 0.315, "SmolVLA + DHB head", ha="center", va="center", fontsize=8, color="#334155")
    axC.text(0.81, 0.272, "2/3 @ 0 cm", ha="center", va="center", fontsize=12, color="#059669", fontweight="bold")
    axC.text(0.81, 0.235, "1/3 @ +3 cm", ha="center", va="center", fontsize=9, color="#475569")

    fig.suptitle("DHB-XR: learn and tokenize motion shape without tying it to one workspace pose", fontsize=20, fontweight="bold", y=0.985)
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
