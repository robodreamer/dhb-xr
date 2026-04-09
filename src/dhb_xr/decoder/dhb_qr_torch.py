"""
Differentiable DHB-QR decoder in PyTorch.
Enables end-to-end training with geodesic orientation loss through the full decode.
"""

import torch
import torch.nn.functional as F


def quat_to_rot_torch(q: torch.Tensor) -> torch.Tensor:
    """Convert wxyz quaternion to 3x3 rotation matrix. Batched.
    Args: q: (..., 4) wxyz
    Returns: (..., 3, 3)
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
        2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    return R


def axis_angle_to_rot_torch(rvec: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle vector to rotation matrix. Batched.
    Args: rvec: (..., 3)
    Returns: (..., 3, 3)
    """
    angle = torch.norm(rvec, dim=-1, keepdim=True).clamp(min=1e-10)  # (..., 1)
    axis = rvec / angle  # (..., 3)
    K = torch.zeros(*rvec.shape[:-1], 3, 3, device=rvec.device, dtype=rvec.dtype)
    K[..., 0, 1] = -axis[..., 2]
    K[..., 0, 2] = axis[..., 1]
    K[..., 1, 0] = axis[..., 2]
    K[..., 1, 2] = -axis[..., 0]
    K[..., 2, 0] = -axis[..., 1]
    K[..., 2, 1] = axis[..., 0]
    I = torch.eye(3, device=rvec.device, dtype=rvec.dtype).expand_as(K)
    angle_unsq = angle.unsqueeze(-1)  # (..., 1, 1)
    R = I + torch.sin(angle_unsq) * K + (1 - torch.cos(angle_unsq)) * (K @ K)
    return R


def rot_to_quat_torch(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to wxyz quaternion. Batched.
    Uses Shepperd's method for numerical stability.
    Args: R: (..., 3, 3)
    Returns: (..., 4) wxyz
    """
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    s = torch.sqrt(torch.clamp(trace + 1, min=1e-10)) * 2  # s = 4*w
    w = 0.25 * s
    x = (R[..., 2, 1] - R[..., 1, 2]) / s
    y = (R[..., 0, 2] - R[..., 2, 0]) / s
    z = (R[..., 1, 0] - R[..., 0, 1]) / s
    q = torch.stack([w, x, y, z], dim=-1)
    q = q / torch.norm(q, dim=-1, keepdim=True).clamp(min=1e-10)
    return q


def decode_dhb_qr_torch(
    linear_inv: torch.Tensor,
    angular_inv: torch.Tensor,
    init_pos: torch.Tensor,
    init_quat: torch.Tensor,
    drop_padded: bool = True,
) -> dict:
    """Differentiable DHB-QR decoder.

    Args:
        linear_inv: (B, T, 5) — magnitude + wxyz quaternion per step
        angular_inv: (B, T, 5) — magnitude + wxyz quaternion per step
        init_pos: (B, 3)
        init_quat: (B, 4) wxyz

    Returns:
        {'positions': (B, T', 3), 'quaternions': (B, T', 4)}
    """
    B, T, _ = linear_inv.shape
    device = linear_inv.device

    lin_mag = linear_inv[..., 0]          # (B, T)
    lin_quat = linear_inv[..., 1:5]       # (B, T, 4) wxyz
    ang_mag = angular_inv[..., 0]         # (B, T)
    ang_quat = angular_inv[..., 1:5]      # (B, T, 4) wxyz

    # Normalize quaternions
    lin_quat = lin_quat / torch.norm(lin_quat, dim=-1, keepdim=True).clamp(min=1e-10)
    ang_quat = ang_quat / torch.norm(ang_quat, dim=-1, keepdim=True).clamp(min=1e-10)

    positions = []
    quaternions = []

    # Initialize frame
    pos = init_pos.clone()                 # (B, 3)
    ang_rot = quat_to_rot_torch(init_quat)  # (B, 3, 3)
    frame_R = torch.eye(3, device=device, dtype=linear_inv.dtype).unsqueeze(0).expand(B, -1, -1).clone()
    frame_pos = init_pos.clone()

    for i in range(T):
        positions.append(frame_pos.clone())

        # Linear frame update
        R_lin = quat_to_rot_torch(lin_quat[:, i])  # (B, 3, 3)
        direction = torch.zeros(B, 3, device=device, dtype=linear_inv.dtype)
        direction[:, 0] = lin_mag[:, i]
        trans = torch.bmm(R_lin, direction.unsqueeze(-1)).squeeze(-1)  # (B, 3)

        # Update position
        frame_pos = frame_pos + trans

        # Angular frame update
        ang_dir = torch.zeros(B, 3, device=device, dtype=angular_inv.dtype)
        ang_dir[:, 0] = ang_mag[:, i]
        rvec = torch.bmm(quat_to_rot_torch(ang_quat[:, i]), ang_dir.unsqueeze(-1)).squeeze(-1)
        R_ang = axis_angle_to_rot_torch(rvec)
        ang_rot = torch.bmm(ang_rot, R_ang.transpose(-1, -2))

        quaternions.append(rot_to_quat_torch(ang_rot))

    positions = torch.stack(positions, dim=1)      # (B, T, 3)
    quaternions = torch.stack(quaternions, dim=1)   # (B, T, 4)

    if drop_padded and T > 2:
        positions = positions[:, 2:]
        quaternions = quaternions[:, 2:]

    return {"positions": positions, "quaternions": quaternions}


def geodesic_loss(q_pred: torch.Tensor, q_gt: torch.Tensor) -> torch.Tensor:
    """Geodesic distance between quaternion sequences.
    Args: q_pred, q_gt: (B, T, 4) wxyz
    Returns: scalar loss (mean geodesic distance in radians)
    """
    dot = torch.abs(torch.sum(q_pred * q_gt, dim=-1))  # (B, T)
    dot = torch.clamp(dot, 0.0, 1.0)
    angle = 2 * torch.acos(dot)  # radians
    return angle.mean()
