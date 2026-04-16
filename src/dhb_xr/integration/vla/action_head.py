"""
DHB-XR Action Head for Vision-Language-Action models.

Implements the DHBActionHead described in Section 4.4 of the CoRL paper:

    h_VLM → cross-attn → Î ∈ R^{T×8} → OAT → 8 tokens → decode → Î
    → decode_dhb → τ̂ ∈ SE(3)^T

Training loss (aligned with LEGATO manuscript Section 7):
    L = λ_inv * L_inv(Î, I*) + λ_act * L_act(â, a*)
    L_inv: invariant_matching_loss (weighted L2 for DHB-DR, geodesic for DHB-QR)
    L_act: L1 action loss (robust to outlier actions)

Advancement over LEGATO: dual-head predicts BOTH invariants (for SE(3) geometric
regularization) AND raw actions (for direct closed-loop execution). The invariant
loss ensures the internal representation is SE(3)-aware; the action loss ensures
the output is directly executable by the controller without lossy decode.

The head is a drop-in replacement for FAST, OFT, or DiT action heads.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from dhb_xr.core.geometry_torch import (
        quat_to_rot_torch,
        rot_to_quat_torch,
    )

    class DHBActionHead(nn.Module):
        """
        Modular action head that predicts DHB invariants from VLM hidden states.

        Architecture:
            1. Cross-attention: VLM embeddings → T query positions → invariant features
            2. Invariant prediction: Linear projection to 8-dim (4 linear + 4 angular)
            3. Optional OAT tokenization: compress to K discrete tokens
            4. DHB decode: invariants + initial pose → SE(3) trajectory

        Args:
            vlm_dim: VLM hidden state dimension (e.g., 768 for SmolVLM2-500M)
            action_horizon: Number of action timesteps T to predict
            invariant_dim: DHB invariant dimension (8 = 4 linear + 4 angular)
            hidden_dim: Internal hidden dimension
            num_heads: Number of attention heads for cross-attention
            num_layers: Number of transformer decoder layers
            dropout: Dropout rate
            lambda_inv: Weight for invariant MSE loss
            lambda_act: Weight for action L1 loss (requires DHB decode)
        """

        def __init__(
            self,
            vlm_dim: int = 768,
            action_horizon: int = 50,
            invariant_dim: int = 8,
            hidden_dim: int = 256,
            num_heads: int = 8,
            num_layers: int = 4,
            dropout: float = 0.1,
            lambda_inv: float = 1.0,
            lambda_act: float = 0.1,
            use_oat: bool = False,
            oat_num_registers: int = 8,
            oat_fsq_levels: Optional[list] = None,
            action_dim: Optional[int] = None,
        ):
            super().__init__()

            self.action_horizon = action_horizon
            self.invariant_dim = invariant_dim
            self.lambda_inv = lambda_inv
            self.use_oat = use_oat
            self.lambda_act = lambda_act
            self.action_dim = action_dim

            # Learnable query positions for each timestep
            self.query_embed = nn.Embedding(action_horizon, hidden_dim)

            # Project VLM hidden states to cross-attention key/value space
            self.vlm_proj = nn.Linear(vlm_dim, hidden_dim)

            # Cross-attention decoder: queries attend to VLM context
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=num_layers,
            )

            # Project decoder output to invariant space
            self.invariant_head = nn.Linear(hidden_dim, invariant_dim)

            # Optional: state projection for conditioning on current EE state
            self.state_proj = nn.Linear(8, hidden_dim)  # eef_pos(3) + aa(3) + grip(2)

            # Optional: OAT tokenizer for discrete token prediction
            self.oat = None
            if use_oat:
                try:
                    from dhb_xr.tokenization.oat_tokenizer import OATTokenizer
                    self.oat = OATTokenizer(
                        input_dim=invariant_dim,
                        input_horizon=action_horizon,
                        emb_dim=hidden_dim,
                        latent_dim=4,
                        num_registers=oat_num_registers,
                        fsq_levels=oat_fsq_levels or [8, 5, 5, 5],
                        encoder_depth=2,
                        decoder_depth=2,
                        num_heads=num_heads,
                    )
                except ImportError:
                    pass

            # Optional: direct action prediction head (dual-head mode)
            self.action_head = None
            if action_dim is not None:
                self.action_head = nn.Linear(hidden_dim, action_dim)

        def forward(
            self,
            vlm_hidden: torch.Tensor,
            state: Optional[torch.Tensor] = None,
            target_invariants: Optional[torch.Tensor] = None,
            target_actions: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            """
            Forward pass: predict DHB invariants from VLM hidden states.

            Args:
                vlm_hidden: (B, S, vlm_dim) VLM encoder output
                state: (B, 8) current EE state [eef_pos, axis_angle, gripper]
                target_invariants: (B, T, 8) ground-truth DHB invariants (for training)
                target_actions: (B, T, 7) ground-truth actions (for training)

            Returns:
                dict with:
                    'pred_invariants': (B, T, 8) predicted DHB invariants
                    'loss': scalar training loss (if targets provided)
                    'loss_inv': invariant MSE component
                    'loss_act': action L1 component (if target_actions provided)
            """
            B = vlm_hidden.shape[0]

            # Project VLM hidden states
            memory = self.vlm_proj(vlm_hidden)  # (B, S, hidden_dim)

            # Build query sequence
            query_pos = torch.arange(self.action_horizon, device=vlm_hidden.device)
            queries = self.query_embed(query_pos).unsqueeze(0).expand(B, -1, -1)

            # Optionally condition on current state
            if state is not None:
                state_feat = self.state_proj(state).unsqueeze(1)  # (B, 1, hidden_dim)
                memory = torch.cat([state_feat, memory], dim=1)

            # Cross-attention decode
            decoded = self.decoder(queries, memory)  # (B, T, hidden_dim)

            # Project to invariant space
            pred_inv = self.invariant_head(decoded)  # (B, T, 8)

            # Optional OAT tokenization: compress to discrete tokens
            if self.oat is not None and self.training:
                oat_loss = self.oat(pred_inv)
                tokens = self.oat.tokenize(pred_inv)
                pred_inv_recon = self.oat.detokenize(tokens)
            else:
                oat_loss = None
                pred_inv_recon = pred_inv

            result = {"pred_invariants": pred_inv, "pred_invariants_recon": pred_inv_recon}
            if self.oat is not None:
                result["oat_loss"] = oat_loss

            # Optional: predict raw actions via action_head
            pred_actions = None
            if self.action_head is not None:
                pred_actions = self.action_head(decoded)  # (B, T, action_dim)
                result["pred_actions"] = pred_actions

            # ── Training loss (aligned with LEGATO manuscript Section 7) ──
            #
            # L_total = lambda_inv * L_inv + lambda_act * L_act [+ 0.1 * L_oat]
            #
            # L_inv: Weighted invariant matching (L2 for DHB-DR, geodesic for DHB-QR)
            #        Uses invariant_matching_loss_torch from dhb_xr.losses when available.
            # L_act: Action L1 loss (robust to outlier actions, per manuscript spec)
            #
            # Advancement over LEGATO: dual-head predicts BOTH invariants (for SE(3)
            # geometric regularization) AND raw actions (for direct closed-loop execution).
            # The invariant loss ensures the internal representation is SE(3)-aware;
            # the action loss ensures the output is directly executable by the controller.
            if target_invariants is not None:
                # Use the project's invariant matching loss (supports DHB-QR geodesic)
                try:
                    from dhb_xr.losses.invariant_loss import invariant_matching_loss_torch
                    loss_inv = invariant_matching_loss_torch(
                        pred_inv, target_invariants, method="dhb_dr"
                    )
                except ImportError:
                    loss_inv = F.mse_loss(pred_inv, target_invariants)
                result["loss_inv"] = loss_inv
                total = self.lambda_inv * loss_inv
                if oat_loss is not None:
                    total = total + 0.1 * oat_loss

                if target_actions is not None and pred_actions is not None:
                    # Dual-head mode: L1 action loss + invariant regularization
                    loss_act = F.l1_loss(pred_actions, target_actions)
                    result["loss_act"] = loss_act
                    total = self.lambda_act * loss_act + self.lambda_inv * loss_inv
                    if oat_loss is not None:
                        total = total + 0.1 * oat_loss
                elif target_actions is not None:
                    # Legacy path: no action_head, placeholder loss
                    loss_act = torch.tensor(0.0, device=vlm_hidden.device)
                    result["loss_act"] = loss_act

                result["loss"] = total

            return result

        def predict(
            self,
            vlm_hidden: torch.Tensor,
            state: Optional[torch.Tensor] = None,
            initial_pose: Optional[Dict[str, torch.Tensor]] = None,
        ) -> Dict[str, torch.Tensor]:
            """
            Inference: predict invariants and optionally decode to trajectory.

            Args:
                vlm_hidden: (B, S, vlm_dim) VLM encoder output
                state: (B, 8) current EE state
                initial_pose: dict with 'position' (B, 3) and 'quaternion' (B, 4) wxyz
                              for DHB decode from current pose

            Returns:
                dict with 'pred_invariants' and optionally 'positions', 'quaternions'
            """
            with torch.no_grad():
                result = self.forward(vlm_hidden, state)

            # If action_head exists, pred_actions is already in result
            # (populated by forward())

            if initial_pose is not None:
                # Decode invariants to SE(3) trajectory from current pose
                # This uses the numpy DHB decoder (not differentiable)
                pred_inv = result["pred_invariants"]
                B = pred_inv.shape[0]

                positions_list = []
                quaternions_list = []

                for b in range(B):
                    inv_np = pred_inv[b].cpu().numpy()
                    lin_inv = inv_np[:, :4]  # linear invariants
                    ang_inv = inv_np[:, 4:]  # angular invariants

                    pos = initial_pose["position"][b].cpu().numpy()
                    quat = initial_pose["quaternion"][b].cpu().numpy()

                    from dhb_xr.decoder.dhb_dr import decode_dhb_dr
                    from dhb_xr.core.types import EncodingMethod

                    dec = decode_dhb_dr(
                        lin_inv, ang_inv,
                        initial_pose={"position": pos, "quaternion": quat},
                        method=EncodingMethod.POSITION,
                    )
                    positions_list.append(dec["positions"])
                    quaternions_list.append(dec["quaternions"])

                import numpy as np
                result["positions"] = torch.from_numpy(
                    np.stack(positions_list)
                ).to(pred_inv.device)
                result["quaternions"] = torch.from_numpy(
                    np.stack(quaternions_list)
                ).to(pred_inv.device)

            return result

        def get_optim_params(self) -> dict:
            """Return optimizer parameter groups."""
            return {"params": self.parameters()}
