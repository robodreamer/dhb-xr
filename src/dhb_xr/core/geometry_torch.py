"""PyTorch geometry utilities for batched SO(3)/SE(3) operations."""

try:
    import torch
except ImportError:
    torch = None

if torch is not None:

    def quat_to_rot_torch(q: "torch.Tensor") -> "torch.Tensor":
        """(..., 4) wxyz -> (..., 3, 3)."""
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        rotation = torch.stack(
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
        return rotation.reshape(*q.shape[:-1], 3, 3)

    def rot_to_quat_torch(rotation: "torch.Tensor") -> "torch.Tensor":
        """(..., 3, 3) -> (..., 4) wxyz."""
        trace = rotation[..., 0, 0] + rotation[..., 1, 1] + rotation[..., 2, 2]
        w = torch.sqrt(1 + trace + 1e-8) / 2
        x = (rotation[..., 2, 1] - rotation[..., 1, 2]) / (4 * w + 1e-8)
        y = (rotation[..., 0, 2] - rotation[..., 2, 0]) / (4 * w + 1e-8)
        z = (rotation[..., 1, 0] - rotation[..., 0, 1]) / (4 * w + 1e-8)
        return torch.stack([w, x, y, z], dim=-1)

    def rot_to_axis_angle_torch(rotation: "torch.Tensor") -> "torch.Tensor":
        """(..., 3, 3) -> (..., 3) rotation vector."""
        angle = torch.acos(
            torch.clamp(
                (rotation[..., 0, 0] + rotation[..., 1, 1] + rotation[..., 2, 2] - 1) / 2,
                -1,
                1,
            )
        )
        denom = torch.sqrt(
            (rotation[..., 2, 1] - rotation[..., 1, 2]) ** 2
            + (rotation[..., 0, 2] - rotation[..., 2, 0]) ** 2
            + (rotation[..., 1, 0] - rotation[..., 0, 1]) ** 2
            + 1e-12
        )
        scale = torch.where(
            angle.abs() < 1e-6,
            torch.ones_like(angle),
            angle / denom,
        )
        x = (rotation[..., 2, 1] - rotation[..., 1, 2]) * scale
        y = (rotation[..., 0, 2] - rotation[..., 2, 0]) * scale
        z = (rotation[..., 1, 0] - rotation[..., 0, 1]) * scale
        return torch.stack([x, y, z], dim=-1)

    def axis_angle_to_rot_torch(v: "torch.Tensor") -> "torch.Tensor":
        """(..., 3) -> (..., 3, 3). Rodrigues."""
        angle = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        axis = v / angle
        skew = torch.zeros(*v.shape[:-1], 3, 3, device=v.device, dtype=v.dtype)
        skew[..., 0, 1] = -axis[..., 2]
        skew[..., 0, 2] = axis[..., 1]
        skew[..., 1, 0] = axis[..., 2]
        skew[..., 1, 2] = -axis[..., 0]
        skew[..., 2, 0] = -axis[..., 1]
        skew[..., 2, 1] = axis[..., 0]
        identity = torch.eye(3, device=v.device, dtype=v.dtype).expand_as(skew)
        return (
            identity
            + torch.sin(angle).unsqueeze(-1) * skew
            + (1 - torch.cos(angle)).unsqueeze(-1) * (skew @ skew)
        )

    def euler_xyz_to_rot_torch(angles: "torch.Tensor") -> "torch.Tensor":
        """(..., 3) extrinsic XYZ Euler angles -> (..., 3, 3)."""
        rx, ry, rz = angles.unbind(dim=-1)
        cx, sx = torch.cos(rx), torch.sin(rx)
        cy, sy = torch.cos(ry), torch.sin(ry)
        cz, sz = torch.cos(rz), torch.sin(rz)
        zeros = torch.zeros_like(rx)
        ones = torch.ones_like(rx)
        rotation_x = torch.stack(
            [
                ones,
                zeros,
                zeros,
                zeros,
                cx,
                -sx,
                zeros,
                sx,
                cx,
            ],
            dim=-1,
        ).reshape(*angles.shape[:-1], 3, 3)
        rotation_y = torch.stack(
            [
                cy,
                zeros,
                sy,
                zeros,
                ones,
                zeros,
                -sy,
                zeros,
                cy,
            ],
            dim=-1,
        ).reshape(*angles.shape[:-1], 3, 3)
        rotation_z = torch.stack(
            [
                cz,
                -sz,
                zeros,
                sz,
                cz,
                zeros,
                zeros,
                zeros,
                ones,
            ],
            dim=-1,
        ).reshape(*angles.shape[:-1], 3, 3)
        return rotation_z @ rotation_y @ rotation_x

    def axis_angle_to_quat_torch(v: "torch.Tensor") -> "torch.Tensor":
        """(..., 3) rotation vector -> normalized (..., 4) wxyz."""
        angle = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
        half_angle = 0.5 * angle
        scale = torch.where(
            angle > 1e-10,
            torch.sin(half_angle) / angle,
            0.5 - angle * angle / 48.0,
        )
        quaternion = torch.cat([torch.cos(half_angle), v * scale], dim=-1)
        return quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(
            1e-12
        )

    def quat_multiply_torch(first: "torch.Tensor", second: "torch.Tensor") -> "torch.Tensor":
        """Hamilton product of broadcast-compatible wxyz quaternions."""
        w1, x1, y1, z1 = first.unbind(dim=-1)
        w2, x2, y2, z2 = second.unbind(dim=-1)
        return torch.stack(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dim=-1,
        )

else:
    quat_to_rot_torch = None
    rot_to_quat_torch = None
    rot_to_axis_angle_torch = None
    axis_angle_to_rot_torch = None
    euler_xyz_to_rot_torch = None
    axis_angle_to_quat_torch = None
    quat_multiply_torch = None
