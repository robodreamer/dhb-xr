"""Batched DHB decoder in PyTorch: wrapper over numpy decode."""

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

if torch is not None:

    import numpy as np
    from dhb_xr.decoder.dhb_dr import decode_dhb_dr
    from dhb_xr.core.types import DHBMethod

    class DHBDecoderTorch(nn.Module):
        """
        Batched DHB-DR decoder.
        invariants (B, N-2, 8), initial_poses list of dict -> positions (B, N', 3), quaternions (B, N', 4).
        """

        def __init__(self, dhb_method: str = "double_reflection"):
            super().__init__()
            self.dhb_method = DHBMethod.DOUBLE_REFLECTION if dhb_method == "double_reflection" else DHBMethod.ORIGINAL

        def forward(
            self,
            invariants: torch.Tensor,
            initial_poses: list,
        ) -> tuple:
            B = invariants.shape[0]
            k = 4 if self.dhb_method == DHBMethod.DOUBLE_REFLECTION else 3
            assert invariants.shape[2] == 2 * k
            pos_list = []
            quat_list = []
            for b in range(B):
                inv = invariants[b].detach().cpu().numpy()
                lin = inv[:, :k]
                ang = inv[:, k:]
                init = initial_poses[b] if isinstance(initial_poses[b], dict) else {
                    "position": initial_poses[b]["position"].cpu().numpy() if hasattr(initial_poses[b]["position"], "cpu") else np.asarray(initial_poses[b]["position"]),
                    "quaternion": initial_poses[b]["quaternion"].cpu().numpy() if hasattr(initial_poses[b]["quaternion"], "cpu") else np.asarray(initial_poses[b]["quaternion"]),
                }
                out = decode_dhb_dr(lin, ang, init, method=EncodingMethod.POSITION, dhb_method=self.dhb_method, drop_padded=True)
                pos_list.append(torch.from_numpy(out["positions"]))
                quat_list.append(torch.from_numpy(out["quaternions"]))
            positions = torch.stack(pos_list, dim=0)
            quaternions = torch.stack(quat_list, dim=0)
            return positions, quaternions

else:
    DHBDecoderTorch = None
