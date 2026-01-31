"""PyTorch geometry utilities for batched SO(3)/SE(3) operations."""

try:
    import torch
except ImportError:
    torch = None

if torch is not None:

    def quat_to_rot_torch(q: "torch.Tensor") -> "torch.Tensor":
        """(..., 4) wxyz -> (..., 3, 3)."""
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        R = torch.stack(
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - w * z),
                2 * (x * z + w * y),
                2 * (x * y + w * z),
                1 - 2 * (x * x + z * z),
                2 * (y * z - w * x),
                2 * (x * z - w * y),
                2 * (y * z + w * x),
                1 - 2 * (x * x + y * y),
            ],
            dim=-1,
        )
        return R.reshape(*q.shape[:-1], 3, 3)

    def rot_to_quat_torch(R: "torch.Tensor") -> "torch.Tensor":
        """(..., 3, 3) -> (..., 4) wxyz."""
        trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        w = torch.sqrt(1 + trace + 1e-8) / 2
        x = (R[..., 2, 1] - R[..., 1, 2]) / (4 * w + 1e-8)
        y = (R[..., 0, 2] - R[..., 2, 0]) / (4 * w + 1e-8)
        z = (R[..., 1, 0] - R[..., 0, 1]) / (4 * w + 1e-8)
        return torch.stack([w, x, y, z], dim=-1)

    def rot_to_axis_angle_torch(R: "torch.Tensor") -> "torch.Tensor":
        """(..., 3, 3) -> (..., 3) rotation vector."""
        angle = torch.acos(
            torch.clamp((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2, -1, 1)
        )
        denom = torch.sqrt(
            (R[..., 2, 1] - R[..., 1, 2]) ** 2
            + (R[..., 0, 2] - R[..., 2, 0]) ** 2
            + (R[..., 1, 0] - R[..., 0, 1]) ** 2
            + 1e-12
        )
        scale = torch.where(
            angle.abs() < 1e-6,
            torch.ones_like(angle),
            angle / denom,
        )
        x = (R[..., 2, 1] - R[..., 1, 2]) * scale
        y = (R[..., 0, 2] - R[..., 2, 0]) * scale
        z = (R[..., 1, 0] - R[..., 0, 1]) * scale
        return torch.stack([x, y, z], dim=-1)

    def axis_angle_to_rot_torch(v: "torch.Tensor") -> "torch.Tensor":
        """(..., 3) -> (..., 3, 3). Rodrigues."""
        angle = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        axis = v / angle
        K = torch.zeros(*v.shape[:-1], 3, 3, device=v.device, dtype=v.dtype)
        K[..., 0, 1] = -axis[..., 2]
        K[..., 0, 2] = axis[..., 1]
        K[..., 1, 0] = axis[..., 2]
        K[..., 1, 2] = -axis[..., 0]
        K[..., 2, 0] = -axis[..., 1]
        K[..., 2, 1] = axis[..., 0]
        I = torch.eye(3, device=v.device, dtype=v.dtype).expand_as(K)
        return I + torch.sin(angle).unsqueeze(-1) * K + (1 - torch.cos(angle)).unsqueeze(-1) * (K @ K)

else:
    quat_to_rot_torch = None
    rot_to_quat_torch = None
    rot_to_axis_angle_torch = None
    axis_angle_to_rot_torch = None
