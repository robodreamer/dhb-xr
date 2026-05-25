"""Tests for the differentiable PyTorch DHB-QR decoder."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dhb_xr.decoder.dhb_qr import decode_dhb_qr
from dhb_xr.decoder.dhb_qr_torch import decode_dhb_qr_torch, geodesic_loss
from dhb_xr.core.types import EncodingMethod


def _qr_invariants(batch: int = 2, steps: int = 6) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    linear = np.zeros((batch, steps, 5), dtype=np.float32)
    angular = np.zeros((batch, steps, 5), dtype=np.float32)
    linear[..., 0] = rng.uniform(0.01, 0.05, size=(batch, steps)).astype(np.float32)
    angular[..., 0] = rng.uniform(-0.08, 0.08, size=(batch, steps)).astype(np.float32)
    linear_quat = rng.normal(size=(batch, steps, 4)).astype(np.float32)
    angular_quat = rng.normal(size=(batch, steps, 4)).astype(np.float32)
    linear[..., 1:] = linear_quat / np.linalg.norm(linear_quat, axis=-1, keepdims=True)
    angular[..., 1:] = angular_quat / np.linalg.norm(angular_quat, axis=-1, keepdims=True)
    return linear, angular


def test_decode_dhb_qr_torch_matches_numpy_decoder():
    linear, angular = _qr_invariants(batch=2, steps=6)
    init_pos = np.array([[0.1, -0.2, 0.3], [-0.1, 0.2, 0.4]], dtype=np.float32)
    init_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (2, 1))

    torch_out = decode_dhb_qr_torch(
        torch.from_numpy(linear),
        torch.from_numpy(angular),
        torch.from_numpy(init_pos),
        torch.from_numpy(init_quat),
        drop_padded=True,
    )

    for batch_idx in range(2):
        numpy_out = decode_dhb_qr(
            linear[batch_idx],
            angular[batch_idx],
            {"position": init_pos[batch_idx], "quaternion": init_quat[batch_idx]},
            method=EncodingMethod.POSITION,
            drop_padded=True,
        )
        np.testing.assert_allclose(
            torch_out["positions"][batch_idx].detach().numpy(),
            numpy_out["positions"],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.abs(torch_out["quaternions"][batch_idx].detach().numpy()),
            np.abs(numpy_out["quaternions"]),
            atol=1e-6,
        )


def test_decode_dhb_qr_torch_shapes_without_dropping_padding():
    linear, angular = _qr_invariants(batch=1, steps=4)
    out = decode_dhb_qr_torch(
        torch.from_numpy(linear),
        torch.from_numpy(angular),
        torch.zeros(1, 3),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        drop_padded=False,
    )

    assert out["positions"].shape == (1, 4, 3)
    assert out["quaternions"].shape == (1, 4, 4)


def test_decode_dhb_qr_torch_supports_gradients():
    linear, angular = _qr_invariants(batch=1, steps=5)
    linear_t = torch.from_numpy(linear).requires_grad_(True)
    angular_t = torch.from_numpy(angular).requires_grad_(True)
    out = decode_dhb_qr_torch(
        linear_t,
        angular_t,
        torch.zeros(1, 3),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )

    target_q = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=out["quaternions"].dtype)
    target_q = target_q.expand_as(out["quaternions"])
    loss = out["positions"].pow(2).mean() + geodesic_loss(out["quaternions"], target_q)
    loss.backward()

    assert linear_t.grad is not None
    assert angular_t.grad is not None
    assert torch.isfinite(linear_t.grad).all()
    assert torch.isfinite(angular_t.grad).all()
