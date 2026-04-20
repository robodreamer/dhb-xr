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

from typing import Dict, Optional, Sequence, Tuple

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
            state_dim: int = 8,
            state_history_window: int = 1,
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
            terminal_loss_weight: float = 1.0,
            contact_dim: int = 0,
            lambda_contact: float = 0.0,
            contact_pos_weight: float | Sequence[float] = 1.0,
            contact_loss_type: str = "bce",
            contact_class_weight: float | Sequence[float] = 1.0,
            contact_phase_dim: int = 0,
            lambda_contact_phase: float = 0.0,
            contact_phase_loss_type: str = "bce",
            contact_phase_class_weight: float | Sequence[float] = 1.0,
            contact_phase_pos_weight: float | Sequence[float] = 1.0,
            gripper_state_dim: int = 0,
            lambda_gripper_state: float = 0.0,
            gripper_state_loss_type: str = "cross_entropy",
            gripper_state_class_weight: float | Sequence[float] = 1.0,
            gripper_state_pos_weight: float | Sequence[float] = 1.0,
            contact_mode: str = "event",
            contact_use_vlm_context: bool = True,
            contact_use_state: bool = True,
            contact_condition_on_phase: bool = False,
            gripper_state_condition_on_phase: bool = False,
            action_loss_ignore_gripper: bool = False,
        ):
            super().__init__()

            self.action_horizon = action_horizon
            self.invariant_dim = invariant_dim
            self.state_dim = int(state_dim)
            self.state_history_window = max(1, int(state_history_window))
            self.lambda_inv = lambda_inv
            self.use_oat = use_oat
            self.lambda_act = lambda_act
            self.action_dim = action_dim
            self.terminal_loss_weight = terminal_loss_weight
            self.contact_dim = contact_dim
            self.lambda_contact = lambda_contact
            self.contact_pos_weight = contact_pos_weight
            self.contact_loss_type = contact_loss_type
            self.contact_class_weight = contact_class_weight
            self.contact_phase_dim = contact_phase_dim
            self.lambda_contact_phase = lambda_contact_phase
            self.contact_phase_loss_type = contact_phase_loss_type
            self.contact_phase_class_weight = contact_phase_class_weight
            self.contact_phase_pos_weight = contact_phase_pos_weight
            self.gripper_state_dim = gripper_state_dim
            self.lambda_gripper_state = lambda_gripper_state
            self.gripper_state_loss_type = gripper_state_loss_type
            self.gripper_state_class_weight = gripper_state_class_weight
            self.gripper_state_pos_weight = gripper_state_pos_weight
            self.contact_mode = contact_mode
            self.contact_use_vlm_context = contact_use_vlm_context
            self.contact_use_state = contact_use_state
            self.contact_condition_on_phase = contact_condition_on_phase
            self.gripper_state_condition_on_phase = gripper_state_condition_on_phase
            self.action_loss_ignore_gripper = action_loss_ignore_gripper

            if self.contact_mode not in {"event", "gripper_state"}:
                raise ValueError(f"Unsupported contact_mode={self.contact_mode!r}")
            if self.contact_mode == "gripper_state":
                if contact_loss_type == "cross_entropy" and contact_dim not in {0, 2}:
                    raise ValueError(
                        "gripper_state mode with cross_entropy expects contact_dim=2 "
                        f"for [open, closed], got {contact_dim}"
                    )
                if contact_loss_type != "cross_entropy" and contact_dim not in {0, 1}:
                    raise ValueError(
                        "gripper_state mode with BCE expects contact_dim=1, "
                        f"got {contact_dim}"
                    )
            if gripper_state_dim:
                if gripper_state_loss_type == "cross_entropy" and gripper_state_dim != 2:
                    raise ValueError(
                        "gripper_state_loss_type='cross_entropy' expects gripper_state_dim=2 "
                        f"for [open, closed], got {gripper_state_dim}"
                    )
                if gripper_state_loss_type != "cross_entropy" and gripper_state_dim != 1:
                    raise ValueError(
                        "gripper_state BCE mode expects gripper_state_dim=1, "
                        f"got {gripper_state_dim}"
                    )

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
            self.state_proj = nn.Linear(
                self.state_dim * self.state_history_window,
                hidden_dim,
            )  # [history, eef_pos(3) + aa(3) + grip(2)]

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

            self.contact_head = None
            self.contact_phase_head = None
            self.contact_phase_proj = None
            self.gripper_state_head = None
            self.contact_cross_attn = None
            if (
                (contact_dim and contact_dim > 0)
                or (contact_phase_dim and contact_phase_dim > 0)
                or (gripper_state_dim and gripper_state_dim > 0)
            ):
                contact_in_dim = hidden_dim
                if contact_use_vlm_context:
                    self.contact_cross_attn = nn.MultiheadAttention(
                        embed_dim=hidden_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        batch_first=True,
                    )
                    contact_in_dim += hidden_dim
                if contact_use_state:
                    contact_in_dim += hidden_dim
                event_in_dim = contact_in_dim
                if contact_condition_on_phase and contact_phase_dim and contact_phase_dim > 0:
                    event_in_dim += hidden_dim
                gripper_state_in_dim = contact_in_dim
                if gripper_state_condition_on_phase and contact_phase_dim and contact_phase_dim > 0:
                    gripper_state_in_dim += hidden_dim
                self.contact_head = nn.Sequential(
                    nn.Linear(event_in_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, contact_dim),
                ) if contact_dim and contact_dim > 0 else None
                self.contact_phase_proj = nn.Sequential(
                    nn.Linear(contact_in_dim, hidden_dim),
                    nn.GELU(),
                ) if contact_phase_dim and contact_phase_dim > 0 else None
                self.contact_phase_head = (
                    nn.Linear(hidden_dim, contact_phase_dim)
                    if contact_phase_dim and contact_phase_dim > 0
                    else None
                )
                self.gripper_state_head = nn.Sequential(
                    nn.Linear(gripper_state_in_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, gripper_state_dim),
                ) if gripper_state_dim and gripper_state_dim > 0 else None

        def _flatten_state_history(self, state: torch.Tensor) -> torch.Tensor:
            """Flatten state history to a fixed-size token input.

            Accepts either:
            - (B, D): single state vector or already-flattened history
            - (B, H, D): state-history window, oldest -> newest
            """
            if state.ndim == 2:
                if state.shape[-1] == self.state_dim * self.state_history_window:
                    return state
                if state.shape[-1] == self.state_dim:
                    if self.state_history_window == 1:
                        return state
                    state = state.unsqueeze(1).expand(-1, self.state_history_window, -1)
                else:
                    raise ValueError(
                        f"Unexpected flat state shape {tuple(state.shape)} for "
                        f"state_dim={self.state_dim}, state_history_window={self.state_history_window}"
                    )

            if state.ndim != 3:
                raise ValueError(f"State tensor must have ndim 2 or 3, got shape {tuple(state.shape)}")

            B, H, D = state.shape
            if D != self.state_dim:
                raise ValueError(
                    f"Expected state feature dim {self.state_dim}, got {D} for shape {tuple(state.shape)}"
                )
            if H < self.state_history_window:
                pad = state[:, :1, :].expand(-1, self.state_history_window - H, -1)
                state = torch.cat([pad, state], dim=1)
            elif H > self.state_history_window:
                state = state[:, -self.state_history_window :, :]
            return state.reshape(B, self.state_dim * self.state_history_window)

        def _invariant_loss(self, pred_inv: torch.Tensor, target_invariants: torch.Tensor) -> torch.Tensor:
            """Compute invariant loss with optional terminal-step upweighting."""
            per_step_mse = (pred_inv - target_invariants).pow(2).mean(dim=-1)  # (B, T)
            if self.terminal_loss_weight is None or self.terminal_loss_weight <= 1.0:
                return per_step_mse.mean()

            weights = torch.linspace(
                1.0,
                float(self.terminal_loss_weight),
                self.action_horizon,
                device=pred_inv.device,
                dtype=pred_inv.dtype,
            )
            weights = weights / weights.mean()
            return (per_step_mse * weights.unsqueeze(0)).mean()

        def forward(
            self,
            vlm_hidden: torch.Tensor,
            state: Optional[torch.Tensor] = None,
            target_invariants: Optional[torch.Tensor] = None,
            target_actions: Optional[torch.Tensor] = None,
            target_contact: Optional[torch.Tensor] = None,
            target_contact_phase: Optional[torch.Tensor] = None,
            target_gripper_state: Optional[torch.Tensor] = None,
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
            memory_vlm = self.vlm_proj(vlm_hidden)  # (B, S, hidden_dim)
            memory = memory_vlm

            # Build query sequence
            query_pos = torch.arange(self.action_horizon, device=vlm_hidden.device)
            queries = self.query_embed(query_pos).unsqueeze(0).expand(B, -1, -1)

            # Optionally condition on current state
            state_feat = None
            if state is not None:
                state_flat = self._flatten_state_history(state)
                state_feat = self.state_proj(state_flat).unsqueeze(1)  # (B, 1, hidden_dim)
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

            pred_contact_logits = None
            pred_contact_phase_logits = None
            pred_gripper_state_logits = None
            if self.contact_head is not None or self.contact_phase_head is not None or self.gripper_state_head is not None:
                contact_inputs = [decoded]
                if self.contact_use_vlm_context:
                    contact_vlm_ctx, _ = self.contact_cross_attn(
                        decoded,
                        memory_vlm,
                        memory_vlm,
                        need_weights=False,
                    )
                    contact_inputs.append(contact_vlm_ctx)
                if self.contact_use_state:
                    if state_feat is not None:
                        contact_inputs.append(state_feat.expand(-1, self.action_horizon, -1))
                    else:
                        zeros = torch.zeros_like(decoded)
                        contact_inputs.append(zeros)
                contact_features = torch.cat(contact_inputs, dim=-1)
                phase_features = None
                if self.contact_phase_head is not None:
                    phase_features = self.contact_phase_proj(contact_features)
                    pred_contact_phase_logits = self.contact_phase_head(phase_features)
                    if self.contact_phase_loss_type == "cross_entropy":
                        if pred_contact_phase_logits.shape[-1] != 3:
                            raise ValueError(
                                "contact_phase_loss_type='cross_entropy' requires contact_phase_dim=3 "
                                f"for [none, close_phase, open_phase], got "
                                f"{pred_contact_phase_logits.shape[-1]}"
                            )
                        result["pred_contact_phase_probs"] = pred_contact_phase_logits.softmax(dim=-1)
                    else:
                        result["pred_contact_phase_probs"] = pred_contact_phase_logits.sigmoid()
                    result["pred_contact_phase_logits"] = pred_contact_phase_logits
                if self.contact_head is not None:
                    event_features = contact_features
                    if (
                        self.contact_condition_on_phase
                        and pred_contact_phase_logits is not None
                        and self.contact_phase_dim > 0
                        and phase_features is not None
                    ):
                        event_features = torch.cat([event_features, phase_features], dim=-1)
                    pred_contact_logits = self.contact_head(event_features)  # (B, T, contact_dim)
                    result["pred_contact_logits"] = pred_contact_logits
                    result["contact_mode"] = self.contact_mode
                    if self.contact_loss_type == "cross_entropy":
                        result["pred_contact_probs"] = pred_contact_logits.softmax(dim=-1)
                    else:
                        result["pred_contact_probs"] = pred_contact_logits.sigmoid()
                if self.gripper_state_head is not None:
                    gripper_state_features = contact_features
                    if self.gripper_state_condition_on_phase and phase_features is not None:
                        gripper_state_features = torch.cat([gripper_state_features, phase_features], dim=-1)
                    pred_gripper_state_logits = self.gripper_state_head(gripper_state_features)
                    result["pred_gripper_state_logits"] = pred_gripper_state_logits
                    if self.gripper_state_loss_type == "cross_entropy":
                        result["pred_gripper_state_probs"] = pred_gripper_state_logits.softmax(dim=-1)
                    else:
                        result["pred_gripper_state_probs"] = pred_gripper_state_logits.sigmoid()

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
            total = None
            if target_invariants is not None:
                loss_inv = self._invariant_loss(pred_inv, target_invariants)
                result["loss_inv"] = loss_inv
                total = self.lambda_inv * loss_inv
                if oat_loss is not None:
                    total = total + 0.1 * oat_loss

                if target_actions is not None and pred_actions is not None:
                    # Dual-head mode: L1 action loss + invariant regularization
                    if self.action_loss_ignore_gripper and pred_actions.shape[-1] >= 7:
                        loss_act = F.l1_loss(pred_actions[..., :6], target_actions[..., :6])
                    else:
                        loss_act = F.l1_loss(pred_actions, target_actions)
                    result["loss_act"] = loss_act
                    total = self.lambda_act * loss_act + self.lambda_inv * loss_inv
                    if oat_loss is not None:
                        total = total + 0.1 * oat_loss
                elif target_actions is not None:
                    # Legacy path: no action_head, placeholder loss
                    loss_act = torch.tensor(0.0, device=vlm_hidden.device)
                    result["loss_act"] = loss_act

            elif target_actions is not None and pred_actions is not None:
                if self.action_loss_ignore_gripper and pred_actions.shape[-1] >= 7:
                    loss_act = F.l1_loss(pred_actions[..., :6], target_actions[..., :6])
                else:
                    loss_act = F.l1_loss(pred_actions, target_actions)
                result["loss_act"] = loss_act
                total = self.lambda_act * loss_act

            if target_contact is not None and pred_contact_logits is not None:
                if self.contact_loss_type == "cross_entropy":
                    class_weight_values = None
                    if not isinstance(self.contact_class_weight, (str, bytes)):
                        try:
                            class_weight_values = list(self.contact_class_weight)
                        except TypeError:
                            class_weight_values = None

                    if class_weight_values is not None:
                        class_weight = torch.as_tensor(
                            class_weight_values,
                            device=pred_contact_logits.device,
                            dtype=pred_contact_logits.dtype,
                        )
                        if class_weight.numel() == 1:
                            class_weight = class_weight.expand(pred_contact_logits.shape[-1])
                        elif class_weight.numel() != pred_contact_logits.shape[-1]:
                            raise ValueError(
                                "contact_class_weight must be scalar or match contact_dim; "
                                f"got {class_weight.numel()} values for contact_dim="
                                f"{pred_contact_logits.shape[-1]}"
                            )
                    else:
                        class_weight = torch.full(
                            (pred_contact_logits.shape[-1],),
                            float(self.contact_class_weight),
                            device=pred_contact_logits.device,
                            dtype=pred_contact_logits.dtype,
                        )

                    if target_contact.ndim == pred_contact_logits.ndim:
                        target_contact_ce = target_contact.argmax(dim=-1)
                    else:
                        target_contact_ce = target_contact
                    target_contact_ce = target_contact_ce.to(device=pred_contact_logits.device, dtype=torch.long)
                    loss_contact = F.cross_entropy(
                        pred_contact_logits.reshape(-1, pred_contact_logits.shape[-1]),
                        target_contact_ce.reshape(-1),
                        weight=class_weight,
                    )
                else:
                    pos_weight_values = None
                    if not isinstance(self.contact_pos_weight, (str, bytes)):
                        try:
                            pos_weight_values = list(self.contact_pos_weight)
                        except TypeError:
                            pos_weight_values = None

                    if pos_weight_values is not None:
                        pos_weight = torch.as_tensor(
                            pos_weight_values,
                            device=pred_contact_logits.device,
                            dtype=pred_contact_logits.dtype,
                        )
                        if pos_weight.numel() == 1:
                            pos_weight = pos_weight.expand(pred_contact_logits.shape[-1])
                        elif pos_weight.numel() != pred_contact_logits.shape[-1]:
                            raise ValueError(
                                "contact_pos_weight must be scalar or match contact_dim; "
                                f"got {pos_weight.numel()} values for contact_dim="
                                f"{pred_contact_logits.shape[-1]}"
                            )
                    else:
                        pos_weight = torch.full(
                            (pred_contact_logits.shape[-1],),
                            float(self.contact_pos_weight),
                            device=pred_contact_logits.device,
                            dtype=pred_contact_logits.dtype,
                        )
                    loss_contact = F.binary_cross_entropy_with_logits(
                        pred_contact_logits,
                        target_contact,
                        pos_weight=pos_weight,
                    )
                result["loss_contact"] = loss_contact
                total = self.lambda_contact * loss_contact if total is None else total + self.lambda_contact * loss_contact

            if target_contact_phase is not None and pred_contact_phase_logits is not None:
                if self.contact_phase_loss_type == "cross_entropy":
                    phase_class_weight_values = None
                    if not isinstance(self.contact_phase_class_weight, (str, bytes)):
                        try:
                            phase_class_weight_values = list(self.contact_phase_class_weight)
                        except TypeError:
                            phase_class_weight_values = None

                    if phase_class_weight_values is not None:
                        phase_class_weight = torch.as_tensor(
                            phase_class_weight_values,
                            device=pred_contact_phase_logits.device,
                            dtype=pred_contact_phase_logits.dtype,
                        )
                        if phase_class_weight.numel() == 1:
                            phase_class_weight = phase_class_weight.expand(pred_contact_phase_logits.shape[-1])
                        elif phase_class_weight.numel() != pred_contact_phase_logits.shape[-1]:
                            raise ValueError(
                                "contact_phase_class_weight must be scalar or match categorical phase classes; "
                                f"got {phase_class_weight.numel()} values for phase logits="
                                f"{pred_contact_phase_logits.shape[-1]}"
                            )
                    else:
                        phase_class_weight = torch.full(
                            (pred_contact_phase_logits.shape[-1],),
                            float(self.contact_phase_class_weight),
                            device=pred_contact_phase_logits.device,
                            dtype=pred_contact_phase_logits.dtype,
                        )

                    if target_contact_phase.ndim == pred_contact_phase_logits.ndim:
                        target_contact_phase_ce = target_contact_phase.argmax(dim=-1)
                    else:
                        target_contact_phase_ce = target_contact_phase
                    target_contact_phase_ce = target_contact_phase_ce.to(
                        device=pred_contact_phase_logits.device,
                        dtype=torch.long,
                    )
                    loss_contact_phase = F.cross_entropy(
                        pred_contact_phase_logits.reshape(-1, pred_contact_phase_logits.shape[-1]),
                        target_contact_phase_ce.reshape(-1),
                        weight=phase_class_weight,
                    )
                else:
                    phase_pos_weight_values = None
                    if not isinstance(self.contact_phase_pos_weight, (str, bytes)):
                        try:
                            phase_pos_weight_values = list(self.contact_phase_pos_weight)
                        except TypeError:
                            phase_pos_weight_values = None

                    if phase_pos_weight_values is not None:
                        phase_pos_weight = torch.as_tensor(
                            phase_pos_weight_values,
                            device=pred_contact_phase_logits.device,
                            dtype=pred_contact_phase_logits.dtype,
                        )
                        if phase_pos_weight.numel() == 1:
                            phase_pos_weight = phase_pos_weight.expand(pred_contact_phase_logits.shape[-1])
                        elif phase_pos_weight.numel() != pred_contact_phase_logits.shape[-1]:
                            raise ValueError(
                                "contact_phase_pos_weight must be scalar or match contact_phase_dim; "
                                f"got {phase_pos_weight.numel()} values for contact_phase_dim="
                                f"{pred_contact_phase_logits.shape[-1]}"
                            )
                    else:
                        phase_pos_weight = torch.full(
                            (pred_contact_phase_logits.shape[-1],),
                            float(self.contact_phase_pos_weight),
                            device=pred_contact_phase_logits.device,
                            dtype=pred_contact_phase_logits.dtype,
                        )
                    loss_contact_phase = F.binary_cross_entropy_with_logits(
                        pred_contact_phase_logits,
                        target_contact_phase,
                        pos_weight=phase_pos_weight,
                    )
                result["loss_contact_phase"] = loss_contact_phase
                total = (
                    self.lambda_contact_phase * loss_contact_phase
                    if total is None
                    else total + self.lambda_contact_phase * loss_contact_phase
                )

            if target_gripper_state is not None and pred_gripper_state_logits is not None:
                if self.gripper_state_loss_type == "cross_entropy":
                    class_weight_values = None
                    if not isinstance(self.gripper_state_class_weight, (str, bytes)):
                        try:
                            class_weight_values = list(self.gripper_state_class_weight)
                        except TypeError:
                            class_weight_values = None

                    if class_weight_values is not None:
                        class_weight = torch.as_tensor(
                            class_weight_values,
                            device=pred_gripper_state_logits.device,
                            dtype=pred_gripper_state_logits.dtype,
                        )
                        if class_weight.numel() == 1:
                            class_weight = class_weight.expand(pred_gripper_state_logits.shape[-1])
                        elif class_weight.numel() != pred_gripper_state_logits.shape[-1]:
                            raise ValueError(
                                "gripper_state_class_weight must be scalar or match gripper_state_dim; "
                                f"got {class_weight.numel()} values for gripper_state_dim="
                                f"{pred_gripper_state_logits.shape[-1]}"
                            )
                    else:
                        class_weight = torch.full(
                            (pred_gripper_state_logits.shape[-1],),
                            float(self.gripper_state_class_weight),
                            device=pred_gripper_state_logits.device,
                            dtype=pred_gripper_state_logits.dtype,
                        )

                    if target_gripper_state.ndim == pred_gripper_state_logits.ndim:
                        target_gripper_state_ce = target_gripper_state.argmax(dim=-1)
                    else:
                        target_gripper_state_ce = target_gripper_state
                    target_gripper_state_ce = target_gripper_state_ce.to(
                        device=pred_gripper_state_logits.device,
                        dtype=torch.long,
                    )
                    loss_gripper_state = F.cross_entropy(
                        pred_gripper_state_logits.reshape(-1, pred_gripper_state_logits.shape[-1]),
                        target_gripper_state_ce.reshape(-1),
                        weight=class_weight,
                    )
                else:
                    pos_weight_values = None
                    if not isinstance(self.gripper_state_pos_weight, (str, bytes)):
                        try:
                            pos_weight_values = list(self.gripper_state_pos_weight)
                        except TypeError:
                            pos_weight_values = None

                    if pos_weight_values is not None:
                        pos_weight = torch.as_tensor(
                            pos_weight_values,
                            device=pred_gripper_state_logits.device,
                            dtype=pred_gripper_state_logits.dtype,
                        )
                        if pos_weight.numel() == 1:
                            pos_weight = pos_weight.expand(pred_gripper_state_logits.shape[-1])
                        elif pos_weight.numel() != pred_gripper_state_logits.shape[-1]:
                            raise ValueError(
                                "gripper_state_pos_weight must be scalar or match gripper_state_dim; "
                                f"got {pos_weight.numel()} values for gripper_state_dim="
                                f"{pred_gripper_state_logits.shape[-1]}"
                            )
                    else:
                        pos_weight = torch.full(
                            (pred_gripper_state_logits.shape[-1],),
                            float(self.gripper_state_pos_weight),
                            device=pred_gripper_state_logits.device,
                            dtype=pred_gripper_state_logits.dtype,
                        )
                    loss_gripper_state = F.binary_cross_entropy_with_logits(
                        pred_gripper_state_logits,
                        target_gripper_state,
                        pos_weight=pos_weight,
                    )

                result["loss_gripper_state"] = loss_gripper_state
                total = (
                    self.lambda_gripper_state * loss_gripper_state
                    if total is None
                    else total + self.lambda_gripper_state * loss_gripper_state
                )

            if total is not None:
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
                    inv_np = pred_inv[b].to(dtype=torch.float32).cpu().numpy()
                    lin_inv = inv_np[:, :4]  # linear invariants
                    ang_inv = inv_np[:, 4:]  # angular invariants

                    pos = initial_pose["position"][b].to(dtype=torch.float32).cpu().numpy()
                    quat = initial_pose["quaternion"][b].to(dtype=torch.float32).cpu().numpy()

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
