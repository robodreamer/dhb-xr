"""Batched DHB encoder in PyTorch: wrapper over numpy encode for GPU-friendly batch API."""

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

if torch is not None:

    import numpy as np
    from dhb_xr.encoder.dhb_dr import encode_dhb_dr
    from dhb_xr.core.types import DHBMethod, EncodingMethod

    class DHBEncoderTorch(nn.Module):
        """
        Batched DHB-DR encoder.
        positions (B, N, 3), quaternions (B, N, 4) wxyz -> invariants (B, N-2, 8).
        Uses numpy encode_dhb_dr per batch item and stacks; supports .to(device) for moving data to GPU.
        """

        def __init__(self, dhb_method: str = "double_reflection"):
            super().__init__()
            self.dhb_method = DHBMethod.DOUBLE_REFLECTION if dhb_method == "double_reflection" else DHBMethod.ORIGINAL
            self.k = 4 if self.dhb_method == DHBMethod.DOUBLE_REFLECTION else 3

        def forward(
            self,
            positions: torch.Tensor,
            quaternions: torch.Tensor,
        ) -> torch.Tensor:
            B, N, _ = positions.shape
            assert quaternions.shape == (B, N, 4)
            device = positions.device
            dtype = positions.dtype
            inv_list = []
            for b in range(B):
                pos = positions[b].detach().cpu().numpy()
                quat = quaternions[b].detach().cpu().numpy()
                out = encode_dhb_dr(
                    pos, quat,
                    method=EncodingMethod.POSITION,
                    use_default_initial_frames=True,
                    dhb_method=self.dhb_method,
                )
                lin = out["linear_motion_invariants"]
                ang = out["angular_motion_invariants"]
                inv_list.append(torch.from_numpy(np.concatenate([lin, ang], axis=1)))
            out_t = torch.stack(inv_list, dim=0).to(device=device, dtype=dtype)
            return out_t

else:
    DHBEncoderTorch = None
