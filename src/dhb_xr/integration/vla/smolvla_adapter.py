"""
SmolVLA + DHBActionHead adapter for VLA inference with invariant actions.

Replaces SmolVLA's default action head with DHBActionHead for SE(3)-
invariant action prediction. At inference:

    Image + Task → SmolVLA VLM → DHBActionHead → DHB invariants
    → decode from current EE pose → SE(3) trajectory → embodik IK

Usage:
    from dhb_xr.integration.vla.smolvla_adapter import SmolVLADHBAdapter

    adapter = SmolVLADHBAdapter.from_pretrained(
        "HuggingFaceVLA/smolvla_libero",
        dhb_checkpoint="path/to/dhb_head.pt"
    )
    actions = adapter.predict(obs_dict, task_description)
"""
from __future__ import annotations

from typing import Dict, Optional
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from dhb_xr.integration.vla.action_head import DHBActionHead

    class SmolVLADHBAdapter(nn.Module):
        """
        Adapter that replaces SmolVLA's action head with DHBActionHead.

        This module:
        1. Uses SmolVLA's VLM (vision + language encoder) as a frozen backbone
        2. Replaces the action expert with DHBActionHead
        3. Predicts DHB invariants → decodes to SE(3) trajectory from current pose
        4. Optionally applies embodik IK for joint-space commands

        Args:
            vlm_dim: Dimension of SmolVLA's VLM hidden states
            action_horizon: Number of timesteps to predict
            freeze_vlm: Whether to freeze the VLM backbone during training
        """

        def __init__(
            self,
            vlm_dim: int = 576,  # SmolVLM2-500M hidden dim
            action_horizon: int = 50,
            freeze_vlm: bool = True,
            hidden_dim: int = 256,
            num_heads: int = 8,
            num_layers: int = 4,
        ):
            super().__init__()

            self.vlm_dim = vlm_dim
            self.freeze_vlm = freeze_vlm

            # DHB action head
            self.dhb_head = DHBActionHead(
                vlm_dim=vlm_dim,
                action_horizon=action_horizon,
                invariant_dim=8,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
            )

            # Placeholder for the VLM backbone (loaded from SmolVLA)
            self.vlm = None

        @classmethod
        def from_pretrained(
            cls,
            smolvla_path: str = "HuggingFaceVLA/smolvla_libero",
            dhb_checkpoint: Optional[str] = None,
            **kwargs,
        ) -> "SmolVLADHBAdapter":
            """
            Load SmolVLA VLM backbone and optional DHB head checkpoint.

            Args:
                smolvla_path: HuggingFace model ID or local path
                dhb_checkpoint: Path to trained DHB action head weights
            """
            adapter = cls(**kwargs)

            # Load SmolVLA VLM (lazy — requires lerobot installed)
            try:
                from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
                import lerobot.policies.smolvla.processor_smolvla  # noqa
                policy = SmolVLAPolicy.from_pretrained(smolvla_path)
                adapter.vlm = policy.model  # The VLM backbone
                if adapter.freeze_vlm:
                    for param in adapter.vlm.parameters():
                        param.requires_grad = False
            except ImportError:
                print("Warning: lerobot not installed. VLM backbone not loaded.")
                print("Use adapter.dhb_head directly with pre-computed VLM features.")

            # Load DHB head checkpoint if provided
            if dhb_checkpoint and Path(dhb_checkpoint).exists():
                state_dict = torch.load(dhb_checkpoint, map_location="cpu")
                adapter.dhb_head.load_state_dict(state_dict)

            return adapter

        def predict_invariants(
            self,
            vlm_features: torch.Tensor,
            state: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Predict DHB invariants from VLM features.

            Args:
                vlm_features: (B, S, vlm_dim) from VLM encoder
                state: (B, 8) current EE state

            Returns:
                (B, T, 8) predicted DHB invariants
            """
            result = self.dhb_head.predict(vlm_features, state=state)
            return result["pred_invariants"]

        def predict_trajectory(
            self,
            vlm_features: torch.Tensor,
            initial_pose: Dict[str, torch.Tensor],
            state: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            """
            Predict SE(3) trajectory from VLM features + current pose.

            Args:
                vlm_features: (B, S, vlm_dim) from VLM encoder
                initial_pose: {'position': (B,3), 'quaternion': (B,4) wxyz}
                state: (B, 8) current EE state

            Returns:
                dict with 'positions' (B, T, 3) and 'quaternions' (B, T, 4)
            """
            result = self.dhb_head.predict(
                vlm_features, state=state, initial_pose=initial_pose
            )
            return {
                "positions": result["positions"],
                "quaternions": result["quaternions"],
                "pred_invariants": result["pred_invariants"],
            }

        def save_dhb_head(self, path: str):
            """Save only the DHB action head weights."""
            torch.save(self.dhb_head.state_dict(), path)

        def get_optim_params(self) -> dict:
            """Return only trainable parameters (DHB head, not frozen VLM)."""
            return {"params": self.dhb_head.parameters()}
