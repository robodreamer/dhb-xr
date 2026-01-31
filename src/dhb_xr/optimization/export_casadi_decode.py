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


def main():
    parser = argparse.ArgumentParser(description="Export CasADi DHB decode for cusadi")
    parser.add_argument("--out", default="fn_dhb_decode_step.casadi", help="Output .casadi path")
    parser.add_argument("--length", type=int, default=0, help="If >0, build full decode of T steps (not implemented yet)")
    args = parser.parse_args()
    if ca is None:
        raise RuntimeError("casadi is required: pip install dhb_xr[optimization]")
    fn = build_decode_step_casadi()
    out_path = args.out
    fn.save(out_path)
    print(f"Saved {out_path}")
    if args.length > 0:
        print("Full trajectory decode (--length) not implemented; use decode_step in a loop with cusadi.")


if __name__ == "__main__":
    main()
