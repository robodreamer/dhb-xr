"""
Build and save a CasADi function for DHB-DR decode (for use with cusadi GPU batch).

Run: python -m dhb_xr.optimization.export_casadi_decode [--out path] [--length T]

Creates a .casadi file that can be moved to cusadi's src/casadi_functions/ and
compiled with: python run_codegen.py --fn=fn_dhb_decode

Requires: pip install dhb_xr[optimization] (casadi). Optional: spatial_casadi for rotations.
"""

from __future__ import annotations

import argparse
import os

try:
    import casadi as ca
except ImportError:
    ca = None


def _euler_to_rot_casadi(angles: "ca.SX") -> "ca.SX":
    """Euler XYZ (extrinsic) to 3x3 rotation matrix. angles: (3,1)."""
    rx, ry, rz = angles[0], angles[1], angles[2]
    cx, sx = ca.cos(rx), ca.sin(rx)
    cy, sy = ca.cos(ry), ca.sin(ry)
    cz, sz = ca.cos(rz), ca.sin(rz)
    Rx = ca.vertcat(
        ca.horzcat(1, 0, 0),
        ca.horzcat(0, cx, -sx),
        ca.horzcat(0, sx, cx),
    )
    Ry = ca.vertcat(
        ca.horzcat(cy, 0, sy),
        ca.horzcat(0, 1, 0),
        ca.horzcat(-sy, 0, cy),
    )
    Rz = ca.vertcat(
        ca.horzcat(cz, -sz, 0),
        ca.horzcat(sz, cz, 0),
        ca.horzcat(0, 0, 1),
    )
    return Rz @ Ry @ Rx


def _axis_angle_to_rot_casadi(rvec: "ca.SX") -> "ca.SX":
    """Rodrigues: rotation vector (3,1) -> 3x3 rotation matrix."""
    th = ca.sqrt(rvec[0] ** 2 + rvec[1] ** 2 + rvec[2] ** 2 + 1e-20)
    k = rvec / th
    K = ca.vertcat(
        ca.horzcat(0, -k[2], k[1]),
        ca.horzcat(k[2], 0, -k[0]),
        ca.horzcat(-k[1], k[0], 0),
    )
    return ca.SX.eye(3) + ca.sin(th) * K + (1 - ca.cos(th)) * (K @ K)


DEFAULT_DECODE_SAMPLES = 80
SUPPORTED_DECODE_SAMPLES = (50, 80, 100, 150, 200)


def decode_filename(samples: int) -> str:
    """Canonical fixed-horizon linear decode filename."""
    return f"fn_dhb_decode_linear_{samples}.casadi"


def build_decode_step_casadi():
    """
    Build a single-step decode: (linear_frame_4x4, angular_rot_3x3, linear_inv_4, angular_inv_4)
    -> (next_linear_frame_4x4, next_angular_rot_3x3, position_3, quat_4).
    Returns a CasADi Function.
    """
    lin_frame = ca.SX.sym("lin_frame", 4, 4)
    ang_rot = ca.SX.sym("ang_rot", 3, 3)
    lin_inv = ca.SX.sym("lin_inv", 4)
    ang_inv = ca.SX.sym("ang_inv", 4)

    # Linear step (DHB-DR: magnitude + euler)
    mag_lin = lin_inv[0]
    euler_lin = lin_inv[1:4]
    R_lin = _euler_to_rot_casadi(euler_lin)
    t_lin = ca.vertcat(mag_lin, 0, 0)
    T_lin = ca.vertcat(ca.horzcat(R_lin, t_lin), ca.horzcat(0, 0, 0, 1))
    next_lin_frame = lin_frame @ T_lin
    pos = next_lin_frame[:3, 3]

    # Angular step
    mag_ang = ang_inv[0]
    rvec_local = ang_rot @ ca.vertcat(mag_ang, 0, 0)
    R_ang = _euler_to_rot_casadi(ang_inv[1:4])
    next_ang_rot = ang_rot @ R_ang
    # Quat from rotation: full rot_to_quat in CasADi is verbose; output identity stub for step
    next_quat = ca.vertcat(1, 0, 0, 0)

    fn = ca.Function(
        "fn_dhb_decode_step",
        [lin_frame, ang_rot, lin_inv, ang_inv],
        [next_lin_frame, next_ang_rot, pos, next_quat],
    )
    return fn


def build_linear_decode_casadi(samples: int = DEFAULT_DECODE_SAMPLES):
    """
    Build a fixed-horizon linear DHB decode function for CusADi.

    The function maps ``samples - 1`` linear invariants plus the initial pose to
    ``samples`` Cartesian positions. This matches the example/runtime contract:
    the first output row is the initial position, then each invariant advances
    the local linear frame once.
    """
    if samples < 2:
        raise ValueError("samples must be at least 2")

    controls = samples - 1
    linear_invariants = ca.SX.sym("linear_invariants", controls * 4)
    init_pos = ca.SX.sym("init_pos", 3)
    init_rot = ca.SX.sym("init_rot", 9)

    linear_frame_rot = ca.reshape(init_rot, 3, 3)
    linear_frame_pos = init_pos
    positions = [linear_frame_pos]

    for i in range(controls):
        base = 4 * i
        linear_invariant = linear_invariants[base : base + 4]
        mag_lin = linear_invariant[0]
        euler_lin = linear_invariant[1:4]
        positions.append(
            linear_frame_pos + linear_frame_rot @ ca.vertcat(mag_lin, 0, 0)
        )
        linear_frame_rot = linear_frame_rot @ _euler_to_rot_casadi(euler_lin)
        linear_frame_pos = positions[-1]

    return ca.Function(
        f"fn_dhb_decode_linear_{samples}",
        [linear_invariants, init_pos, init_rot],
        [ca.horzcat(*positions).T],
    )


def main():
    parser = argparse.ArgumentParser(description="Export CasADi DHB decode for cusadi")
    parser.add_argument(
        "--out",
        default=None,
        help="Output .casadi path. Only valid for a single --length value.",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Directory for canonical fixed-horizon output filenames.",
    )
    parser.add_argument(
        "--length",
        type=int,
        nargs="+",
        default=[0],
        help=(
            "Build one or more fixed-horizon linear decode functions with this "
            f"many output samples. Common choices: {SUPPORTED_DECODE_SAMPLES}."
        ),
    )
    args = parser.parse_args()
    if ca is None:
        raise RuntimeError("casadi is required: pip install dhb_xr[optimization]")

    lengths = args.length
    if args.out is not None and len(lengths) != 1:
        raise ValueError("--out can only be used with a single --length value")

    if lengths == [0]:
        out_path = args.out or os.path.join(args.out_dir, "fn_dhb_decode_step.casadi")
        fn = build_decode_step_casadi()
        fn.save(out_path)
        print(f"Saved {out_path}")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    for length in lengths:
        fn = build_linear_decode_casadi(length)
        out_path = args.out or os.path.join(args.out_dir, decode_filename(length))
        fn.save(out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
