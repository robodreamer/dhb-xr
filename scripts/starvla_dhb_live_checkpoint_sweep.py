from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1"


def _step_num(path: Path) -> int:
    m = re.search(r"steps_(\d+)_pytorch_model\.pt$", path.name)
    if not m:
        raise ValueError(f"Could not parse checkpoint step from {path}")
    return int(m.group(1))


def _parse_case(text: str) -> tuple[int, float]:
    try:
        ep_str, offset_str = text.split(":", 1)
        return int(ep_str), float(offset_str)
    except Exception as exc:
        raise ValueError(f"Invalid case '{text}'. Expected EPISODE:OFFSET_CM") from exc


def _run_case(
    *,
    checkpoint: Path,
    config_yaml: str,
    task_suite_name: str,
    task_id: int,
    episode_idx: int,
    offset_cm: float,
    output_dir: Path,
    replan_every: int,
    max_first_offset_mm: float,
    max_step_mm: float,
    smooth_window: int,
    blend_overlap_alpha: float,
    blend_mode: str,
    commit_prefix_steps: int,
    adaptive_commit_threshold_mm: float,
    adaptive_commit_max_steps: int,
    delta_continuation_alpha: float,
    delta_continuation_steps: int,
    gripper_source: str,
    init_state_source: str,
    gripper_threshold: float,
    gripper_close_threshold: float,
    gripper_open_threshold: float,
    gripper_transition_debounce_steps: int,
    image_orientation: str,
    trajectory_source: str,
    direct_action_xyz_scale: tuple[float, float, float],
    direct_action_xyz_bias_mm: tuple[float, float, float],
    state_history_update_rate: str,
    state_gripper_source: str,
    disable_gripper_gates: bool,
    max_policy_steps: int,
    settle_steps: int,
    reach_only: bool,
    reach_threshold_xy_m: float,
    reach_threshold_z_m: float,
    reach_stop_on_success: bool,
) -> dict:
    summary_path = output_dir / (
        f"{checkpoint.stem}_ep{episode_idx}_offset{offset_cm:.1f}cm_summary.json"
    )
    cmd = [
        sys.executable,
        "scripts/starvla_dhb_embodik_live_rollout.py",
        "--config_yaml",
        config_yaml,
        "--checkpoint",
        str(checkpoint),
        "--task_suite_name",
        task_suite_name,
        "--task_id",
        str(task_id),
        "--episode_idx",
        str(episode_idx),
        "--offset-cm",
        str(offset_cm),
        "--replan_every",
        str(replan_every),
        "--max-first-offset-mm",
        str(max_first_offset_mm),
        "--max-step-mm",
        str(max_step_mm),
        "--smooth-window",
        str(smooth_window),
        "--blend-overlap-alpha",
        str(blend_overlap_alpha),
        "--blend-mode",
        blend_mode,
        "--commit-prefix-steps",
        str(commit_prefix_steps),
        "--adaptive-commit-threshold-mm",
        str(adaptive_commit_threshold_mm),
        "--adaptive-commit-max-steps",
        str(adaptive_commit_max_steps),
        "--delta-continuation-alpha",
        str(delta_continuation_alpha),
        "--delta-continuation-steps",
        str(delta_continuation_steps),
        "--gripper-source",
        gripper_source,
        "--init-state-source",
        init_state_source,
        "--gripper-threshold",
        str(gripper_threshold),
        "--gripper-close-threshold",
        str(gripper_close_threshold),
        "--gripper-open-threshold",
        str(gripper_open_threshold),
        "--gripper-transition-debounce-steps",
        str(gripper_transition_debounce_steps),
        "--image-orientation",
        image_orientation,
        "--trajectory-source",
        trajectory_source,
        "--direct-action-xyz-scale",
        str(direct_action_xyz_scale[0]),
        str(direct_action_xyz_scale[1]),
        str(direct_action_xyz_scale[2]),
        "--direct-action-xyz-bias-mm",
        str(direct_action_xyz_bias_mm[0]),
        str(direct_action_xyz_bias_mm[1]),
        str(direct_action_xyz_bias_mm[2]),
        "--state-history-update-rate",
        state_history_update_rate,
        "--state-gripper-source",
        state_gripper_source,
        "--max-policy-steps",
        str(max_policy_steps),
        "--settle-steps",
        str(settle_steps),
        "--save_json",
        str(summary_path),
        "--no-video",
    ]
    if disable_gripper_gates:
        cmd.append("--disable-gripper-gates")
    if reach_only:
        cmd.extend(
            [
                "--reach-only",
                "--reach-threshold-xy-m",
                str(reach_threshold_xy_m),
                "--reach-threshold-z-m",
                str(reach_threshold_z_m),
            ]
        )
        if reach_stop_on_success:
            cmd.append("--reach-stop-on-success")
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(REPO_ROOT / "vendor/starVLA"))
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("MUJOCO_GL", "egl")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=env)
    summary = json.loads(summary_path.read_text())
    chunks = summary.get("chunks")
    if isinstance(chunks, list):
        summary["chunks"] = f"<{len(chunks)} chunks saved to {summary_path}>"
    step_trace = summary.get("step_trace")
    if isinstance(step_trace, list):
        summary["step_trace"] = f"<{len(step_trace)} entries saved to {summary_path}>"
    return summary


def _aggregate(rows: list[dict]) -> dict:
    success_count = sum(1 for r in rows if r.get("success"))
    close_count = sum(1 for r in rows if r.get("close_step") is not None)
    min_bowl_xyz_vals = [r.get("min_bowl_xyz_m") for r in rows if r.get("min_bowl_xyz_m") is not None]
    min_bowl_xy_vals = [r.get("min_bowl_xy_m") for r in rows if r.get("min_bowl_xy_m") is not None]
    ik_fail_vals = [r.get("ik_fail_total", 0) for r in rows]
    accel_vals = [
        r.get("motion_metrics", {}).get("mean_accel_mm")
        for r in rows
        if r.get("motion_metrics", {}).get("mean_accel_mm") is not None
    ]
    hold_vals = [
        r.get("motion_metrics", {}).get("hold_fraction")
        for r in rows
        if r.get("motion_metrics", {}).get("hold_fraction") is not None
    ]
    return {
        "success_count": int(success_count),
        "close_count": int(close_count),
        "avg_min_bowl_xyz_m": None if not min_bowl_xyz_vals else float(sum(min_bowl_xyz_vals) / len(min_bowl_xyz_vals)),
        "avg_min_bowl_xy_m": None if not min_bowl_xy_vals else float(sum(min_bowl_xy_vals) / len(min_bowl_xy_vals)),
        "avg_ik_fail_total": None if not ik_fail_vals else float(sum(ik_fail_vals) / len(ik_fail_vals)),
        "avg_mean_accel_mm": None if not accel_vals else float(sum(accel_vals) / len(accel_vals)),
        "avg_hold_fraction": None if not hold_vals else float(sum(hold_vals) / len(hold_vals)),
    }


def _sort_key(item: dict, *, reach_only: bool = False):
    agg = item["aggregate"]
    primary_dist = agg["avg_min_bowl_xy_m"] if reach_only else agg["avg_min_bowl_xyz_m"]
    secondary_dist = agg["avg_min_bowl_xyz_m"] if reach_only else agg["avg_min_bowl_xy_m"]
    return (
        -agg["success_count"],
        -agg["close_count"],
        float("inf") if primary_dist is None else primary_dist,
        float("inf") if secondary_dist is None else secondary_dist,
        float("inf") if agg["avg_mean_accel_mm"] is None else agg["avg_mean_accel_mm"],
        float("inf") if agg["avg_ik_fail_total"] is None else agg["avg_ik_fail_total"],
        item["step"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", default="configs/libero_dhbxr_qwen.yaml")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        action="append",
        default=None,
        help="Only evaluate checkpoints with these step numbers. Repeatable.",
    )
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--case", action="append", default=None, help="EPISODE_IDX:OFFSET_CM, repeatable")
    parser.add_argument("--replan_every", type=int, default=4)
    parser.add_argument("--max-first-offset-mm", type=float, default=35.0)
    parser.add_argument("--max-step-mm", type=float, default=20.0)
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--blend-overlap-alpha", type=float, default=0.6)
    parser.add_argument("--blend-mode", choices=("constant", "ramp"), default="constant")
    parser.add_argument("--commit-prefix-steps", type=int, default=0)
    parser.add_argument("--adaptive-commit-threshold-mm", type=float, default=0.0)
    parser.add_argument("--adaptive-commit-max-steps", type=int, default=0)
    parser.add_argument("--delta-continuation-alpha", type=float, default=0.0)
    parser.add_argument("--delta-continuation-steps", type=int, default=0)
    parser.add_argument("--gripper-source", choices=("demo", "model"), default="demo")
    parser.add_argument("--init-state-source", choices=("demo", "benchmark"), default="demo")
    parser.add_argument("--gripper-threshold", type=float, default=0.5)
    parser.add_argument("--gripper-close-threshold", type=float, default=0.6)
    parser.add_argument("--gripper-open-threshold", type=float, default=0.4)
    parser.add_argument("--gripper-transition-debounce-steps", type=int, default=1)
    parser.add_argument("--image-orientation", choices=("raw", "flipudlr"), default="raw")
    parser.add_argument("--trajectory-source", choices=("dhb", "direct_action"), default="dhb")
    parser.add_argument(
        "--state-history-update-rate",
        choices=("policy", "substep"),
        default="policy",
        help=(
            "Timing contract for policy state history: one update per VLA action "
            "step, or every low-level controller substep for streaming proprioception."
        ),
    )
    parser.add_argument(
        "--state-gripper-source",
        choices=("observed", "commanded"),
        default="observed",
    )
    parser.add_argument(
        "--disable-gripper-gates",
        action="store_true",
        help=(
            "Forward rollout-side gripper gate disabling. Use this for learned "
            "gripper evaluations so object-specific close/release gates do not "
            "mask model behavior."
        ),
    )
    parser.add_argument(
        "--direct-action-xyz-scale",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("SX", "SY", "SZ"),
        help="Task-agnostic XYZ delta scale for --trajectory-source direct_action.",
    )
    parser.add_argument(
        "--direct-action-xyz-bias-mm",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("BX", "BY", "BZ"),
        help="Task-agnostic per-step XYZ delta bias in millimeters for direct-action deltas.",
    )
    parser.add_argument(
        "--max-policy-steps",
        type=int,
        default=0,
        help=(
            "Closed-loop policy execution cap passed to the rollout script. "
            "0 preserves the episode demo length; use a larger value for delayed "
            "offset-adaptation rollouts."
        ),
    )
    parser.add_argument("--settle-steps", type=int, default=60)
    parser.add_argument("--reach-only", action="store_true")
    parser.add_argument("--reach-threshold-xy-m", type=float, default=0.05)
    parser.add_argument("--reach-threshold-z-m", type=float, default=0.08)
    parser.add_argument("--reach-stop-on-success", action="store_true")
    parser.add_argument(
        "--sweep-name",
        default=None,
        help=(
            "Optional child-output subdirectory under live_checkpoint_sweep. "
            "When omitted and --output-json is set, the output JSON stem is used "
            "so reruns with different protocols do not overwrite per-case files."
        ),
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    cases = args.case or ["0:0", "1:3"]
    parsed_cases = [_parse_case(c) for c in cases]
    run_dir = Path(args.run_dir)
    checkpoints = sorted(run_dir.glob("checkpoints/steps_*_pytorch_model.pt"), key=_step_num)
    if args.checkpoint_step:
        wanted_steps = {int(step) for step in args.checkpoint_step}
        checkpoints = [path for path in checkpoints if _step_num(path) in wanted_steps]
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found under {run_dir}")

    sweep_name = args.sweep_name
    if sweep_name is None and args.output_json:
        sweep_name = Path(args.output_json).stem
    sweep_dir = run_dir / "live_checkpoint_sweep"
    if sweep_name:
        safe_sweep_name = "".join(
            ch if ch.isalnum() or ch in "._-" else "_" for ch in sweep_name
        ).strip("_")
        if safe_sweep_name:
            sweep_dir = sweep_dir / safe_sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for checkpoint in checkpoints:
        step = _step_num(checkpoint)
        ckpt_dir = sweep_dir / f"step_{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for episode_idx, offset_cm in parsed_cases:
            row = _run_case(
                checkpoint=checkpoint,
                config_yaml=args.config_yaml,
                task_suite_name=args.task_suite_name,
                task_id=args.task_id,
                episode_idx=episode_idx,
                offset_cm=offset_cm,
                output_dir=ckpt_dir,
                replan_every=args.replan_every,
                max_first_offset_mm=args.max_first_offset_mm,
                max_step_mm=args.max_step_mm,
                smooth_window=args.smooth_window,
                blend_overlap_alpha=args.blend_overlap_alpha,
                blend_mode=args.blend_mode,
                commit_prefix_steps=args.commit_prefix_steps,
                adaptive_commit_threshold_mm=args.adaptive_commit_threshold_mm,
                adaptive_commit_max_steps=args.adaptive_commit_max_steps,
                delta_continuation_alpha=args.delta_continuation_alpha,
                delta_continuation_steps=args.delta_continuation_steps,
                gripper_source=args.gripper_source,
                init_state_source=args.init_state_source,
                gripper_threshold=float(args.gripper_threshold),
                gripper_close_threshold=float(args.gripper_close_threshold),
                gripper_open_threshold=float(args.gripper_open_threshold),
                gripper_transition_debounce_steps=max(1, int(args.gripper_transition_debounce_steps)),
                image_orientation=args.image_orientation,
                trajectory_source=args.trajectory_source,
                direct_action_xyz_scale=tuple(float(v) for v in args.direct_action_xyz_scale),
                direct_action_xyz_bias_mm=tuple(float(v) for v in args.direct_action_xyz_bias_mm),
                state_history_update_rate=args.state_history_update_rate,
                state_gripper_source=args.state_gripper_source,
                disable_gripper_gates=bool(args.disable_gripper_gates),
                max_policy_steps=int(args.max_policy_steps),
                settle_steps=args.settle_steps,
                reach_only=args.reach_only,
                reach_threshold_xy_m=args.reach_threshold_xy_m,
                reach_threshold_z_m=args.reach_threshold_z_m,
                reach_stop_on_success=args.reach_stop_on_success,
            )
            rows.append(row)
        results.append(
            {
                "checkpoint": str(checkpoint),
                "step": int(step),
                "cases": rows,
                "aggregate": _aggregate(rows),
            }
        )

    ranked = sorted(results, key=lambda item: _sort_key(item, reach_only=args.reach_only))
    payload = {
        "config_yaml": args.config_yaml,
        "run_dir": str(run_dir),
        "checkpoint_steps": None if not args.checkpoint_step else [int(s) for s in args.checkpoint_step],
        "task_suite_name": args.task_suite_name,
        "task_id": int(args.task_id),
        "cases": [{"episode_idx": ep, "offset_cm": offset} for ep, offset in parsed_cases],
        "settings": {
            "replan_every": int(args.replan_every),
            "max_first_offset_mm": float(args.max_first_offset_mm),
            "max_step_mm": float(args.max_step_mm),
            "smooth_window": int(args.smooth_window),
            "blend_overlap_alpha": float(args.blend_overlap_alpha),
            "blend_mode": args.blend_mode,
            "commit_prefix_steps": int(args.commit_prefix_steps),
            "adaptive_commit_threshold_mm": float(args.adaptive_commit_threshold_mm),
            "adaptive_commit_max_steps": int(args.adaptive_commit_max_steps),
            "delta_continuation_alpha": float(args.delta_continuation_alpha),
            "delta_continuation_steps": int(args.delta_continuation_steps),
            "gripper_source": args.gripper_source,
            "gripper_threshold": float(args.gripper_threshold),
            "gripper_close_threshold": float(args.gripper_close_threshold),
            "gripper_open_threshold": float(args.gripper_open_threshold),
            "gripper_transition_debounce_steps": int(args.gripper_transition_debounce_steps),
            "image_orientation": args.image_orientation,
            "trajectory_source": args.trajectory_source,
            "state_history_update_rate": args.state_history_update_rate,
            "state_gripper_source": args.state_gripper_source,
            "disable_gripper_gates": bool(args.disable_gripper_gates),
            "direct_action_xyz_scale": [float(v) for v in args.direct_action_xyz_scale],
            "direct_action_xyz_bias_mm": [float(v) for v in args.direct_action_xyz_bias_mm],
            "init_state_source": args.init_state_source,
            "max_policy_steps": int(args.max_policy_steps),
            "settle_steps": int(args.settle_steps),
            "reach_only": bool(args.reach_only),
            "reach_threshold_xy_m": float(args.reach_threshold_xy_m),
            "reach_threshold_z_m": float(args.reach_threshold_z_m),
            "reach_stop_on_success": bool(args.reach_stop_on_success),
        },
        "ranked": ranked,
    }

    output_json = Path(args.output_json) if args.output_json else sweep_dir / "summary.json"
    output_json.write_text(json.dumps(payload, indent=2))

    md_lines = [
        "# StarVLA Live Checkpoint Sweep",
        "",
        f"- run dir: `{run_dir}`",
        f"- config: `{args.config_yaml}`",
        f"- cases: {', '.join(f'ep{ep}/offset{offset:.1f}cm' for ep, offset in parsed_cases)}",
        "",
        "| Rank | Step | Successes | Closes | Avg min bowl XYZ (m) | Avg min bowl XY (m) | Avg accel (mm) | Avg hold frac | Avg IK fails |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, item in enumerate(ranked, start=1):
        agg = item["aggregate"]
        md_lines.append(
            "| {rank} | {step} | {succ} | {close} | {xyz:.4f} | {xy:.4f} | {accel:.1f} | {hold:.3f} | {ik:.1f} |".format(
                rank=rank,
                step=item["step"],
                succ=agg["success_count"],
                close=agg["close_count"],
                xyz=float("nan") if agg["avg_min_bowl_xyz_m"] is None else agg["avg_min_bowl_xyz_m"],
                xy=float("nan") if agg["avg_min_bowl_xy_m"] is None else agg["avg_min_bowl_xy_m"],
                accel=float("nan") if agg["avg_mean_accel_mm"] is None else agg["avg_mean_accel_mm"],
                hold=float("nan") if agg["avg_hold_fraction"] is None else agg["avg_hold_fraction"],
                ik=float("nan") if agg["avg_ik_fail_total"] is None else agg["avg_ik_fail_total"],
            )
        )
    output_md = Path(args.output_md) if args.output_md else sweep_dir / "summary.md"
    output_md.write_text("\n".join(md_lines) + "\n")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
