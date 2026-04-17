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

import numpy as np

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

        @classmethod
        def from_dhb_head(
            cls,
            dhb_head: nn.Module,
            *,
            vlm_dim: Optional[int] = None,
            action_horizon: Optional[int] = None,
        ) -> "SmolVLADHBAdapter":
            """Wrap an already-constructed DHB head in the tested adapter path."""
            inferred_vlm_dim = vlm_dim or getattr(dhb_head, "vlm_dim", 576)
            inferred_horizon = action_horizon or getattr(dhb_head, "action_horizon", 50)
            adapter = cls(vlm_dim=inferred_vlm_dim, action_horizon=inferred_horizon)
            adapter.dhb_head = dhb_head
            return adapter

        @staticmethod
        def observation_to_state(
            obs_dict: Dict,
            *,
            device: Optional[torch.device | str] = None,
        ) -> torch.Tensor:
            """Convert a LIBERO-style observation dict into the DHB 8D state tensor."""
            from scipy.spatial.transform import Rotation

            eef_pos = np.asarray(obs_dict.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32)
            eef_quat = np.asarray(
                obs_dict.get("robot0_eef_quat", np.array([0.0, 0.0, 0.0, 1.0])),
                dtype=np.float32,
            )
            aa = Rotation.from_quat(eef_quat).as_rotvec().astype(np.float32)
            gripper = np.asarray(obs_dict.get("robot0_gripper_qpos", np.zeros(2)), dtype=np.float32)
            state = np.concatenate([eef_pos, aa, gripper], dtype=np.float32)
            return torch.from_numpy(state).unsqueeze(0).to(device=device)

        @staticmethod
        def observation_to_initial_pose(
            obs_dict: Dict,
            *,
            device: Optional[torch.device | str] = None,
        ) -> Dict[str, torch.Tensor]:
            """Convert a LIBERO-style observation dict into DHB decode initial pose tensors."""
            eef_pos = np.asarray(obs_dict.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32)
            eef_quat_xyzw = np.asarray(
                obs_dict.get("robot0_eef_quat", np.array([0.0, 0.0, 0.0, 1.0])),
                dtype=np.float32,
            )
            q_wxyz = np.array(
                [eef_quat_xyzw[3], eef_quat_xyzw[0], eef_quat_xyzw[1], eef_quat_xyzw[2]],
                dtype=np.float32,
            )
            return {
                "position": torch.from_numpy(eef_pos).unsqueeze(0).to(device=device),
                "quaternion": torch.from_numpy(q_wxyz).unsqueeze(0).to(device=device),
            }

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

        def predict_invariants_from_observation(
            self,
            vlm_features: torch.Tensor,
            obs_dict: Dict,
        ) -> torch.Tensor:
            """Predict invariants from VLM features using state extracted from observation."""
            state = self.observation_to_state(obs_dict, device=vlm_features.device)
            return self.predict_invariants(vlm_features, state=state)

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

        def predict_trajectory_from_observation(
            self,
            vlm_features: torch.Tensor,
            obs_dict: Dict,
        ) -> Dict[str, torch.Tensor]:
            """Predict trajectory using both state and initial pose from observation."""
            state = self.observation_to_state(obs_dict, device=vlm_features.device)
            initial_pose = self.observation_to_initial_pose(obs_dict, device=vlm_features.device)
            return self.predict_trajectory(vlm_features, initial_pose=initial_pose, state=state)

        def save_dhb_head(self, path: str):
            """Save only the DHB action head weights."""
            torch.save(self.dhb_head.state_dict(), path)

        def get_optim_params(self) -> dict:
            """Return only trainable parameters (DHB head, not frozen VLM)."""
            return {"params": self.dhb_head.parameters()}
