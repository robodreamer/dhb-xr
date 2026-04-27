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
            oat_emb_dim: Optional[int] = None,
            oat_latent_dim: int = 4,
            oat_encoder_depth: int = 2,
            oat_decoder_depth: int = 2,
            oat_num_heads: Optional[int] = None,
            oat_init_seed: Optional[int] = None,
            oat_checkpoint_path: Optional[str] = None,
            freeze_oat: bool = False,
            supervise_oat_tokens: bool = False,
            lambda_oat_token: float = 0.0,
            lambda_oat_autoencode: float = 0.1,
            lambda_oat_register_diversity: float = 0.0,
            oat_autoencode_target: str = "pred",
            train_with_discrete_oat_tokens: bool = False,
            use_exported_oat_tokens: bool = False,
            oat_token_prefix_train_prob: float = 0.0,
            oat_token_prefix_min_k: int = 1,
            action_dim: Optional[int] = None,
            terminal_loss_weight: float = 1.0,
            prefix_loss_steps: int = 0,
            prefix_loss_weight: float = 1.0,
            invariant_loss_channel_scale: Optional[Sequence[float]] = None,
            lambda_invariant_delta: float = 0.0,
            head_query_residual_scale: float = 0.0,
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
            lambda_gripper_state_smooth: float = 0.0,
            lambda_gripper_state_margin: float = 0.0,
            gripper_state_closed_margin: float = 0.65,
            gripper_state_open_margin: float = 0.35,
            contact_mode: str = "event",
            contact_use_vlm_context: bool = True,
            contact_use_state: bool = True,
            contact_condition_on_phase: bool = False,
            gripper_state_condition_on_phase: bool = False,
            action_loss_ignore_gripper: bool = False,
            action_loss_target_shift: int = 0,
            action_loss_channel_weight: Optional[Sequence[float]] = None,
            use_action_residual_adapter: bool = False,
            action_residual_hidden_dim: Optional[int] = None,
            action_residual_scale: float = 1.0,
            lambda_action_residual_l2: float = 0.0,
            lambda_action_base_cumsum_l2: float = 0.0,
            action_base_cumsum_dims: int = 3,
            use_action_closed_residual_adapter: bool = False,
            action_closed_residual_hidden_dim: Optional[int] = None,
            action_closed_residual_scale: float = 1.0,
            action_closed_residual_gate_source: str = "state_gripper",
            action_closed_residual_gripper_threshold: float = 0.015,
            action_closed_residual_gate_temperature: float = 0.003,
            action_closed_residual_hard_gate: bool = False,
            action_closed_residual_zero_gripper_dim: bool = True,
            lambda_action_closed_residual_l2: float = 0.0,
        ):
            super().__init__()

            self.action_horizon = action_horizon
            self.invariant_dim = invariant_dim
            self.state_dim = int(state_dim)
            self.state_history_window = max(1, int(state_history_window))
            self.lambda_inv = lambda_inv
            self.use_oat = use_oat
            self.lambda_act = lambda_act
            self.supervise_oat_tokens = bool(supervise_oat_tokens)
            self.lambda_oat_token = float(lambda_oat_token)
            self.lambda_oat_autoencode = float(lambda_oat_autoencode)
            self.lambda_oat_register_diversity = float(lambda_oat_register_diversity)
            self.oat_autoencode_target = str(oat_autoencode_target)
            self.train_with_discrete_oat_tokens = bool(train_with_discrete_oat_tokens)
            self.use_exported_oat_tokens = bool(use_exported_oat_tokens)
            self.oat_token_prefix_train_prob = float(oat_token_prefix_train_prob)
            self.oat_token_prefix_min_k = int(oat_token_prefix_min_k)
            self.freeze_oat = bool(freeze_oat)
            self.action_dim = action_dim
            self.terminal_loss_weight = terminal_loss_weight
            self.prefix_loss_steps = max(0, int(prefix_loss_steps))
            self.prefix_loss_weight = float(prefix_loss_weight)
            self.lambda_invariant_delta = float(lambda_invariant_delta)
            self.head_query_residual_scale = float(head_query_residual_scale)
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
            self.lambda_gripper_state_smooth = float(lambda_gripper_state_smooth)
            self.lambda_gripper_state_margin = float(lambda_gripper_state_margin)
            self.gripper_state_closed_margin = float(gripper_state_closed_margin)
            self.gripper_state_open_margin = float(gripper_state_open_margin)
            self.contact_mode = contact_mode
            self.contact_use_vlm_context = contact_use_vlm_context
            self.contact_use_state = contact_use_state
            self.contact_condition_on_phase = contact_condition_on_phase
            self.gripper_state_condition_on_phase = gripper_state_condition_on_phase
            self.action_loss_ignore_gripper = action_loss_ignore_gripper
            self.action_loss_target_shift = max(0, int(action_loss_target_shift))
            self.use_action_residual_adapter = bool(use_action_residual_adapter)
            self.action_residual_scale = float(action_residual_scale)
            self.lambda_action_residual_l2 = float(lambda_action_residual_l2)
            self.lambda_action_base_cumsum_l2 = float(lambda_action_base_cumsum_l2)
            self.action_base_cumsum_dims = max(0, int(action_base_cumsum_dims))
            self.use_action_closed_residual_adapter = bool(use_action_closed_residual_adapter)
            self.action_closed_residual_scale = float(action_closed_residual_scale)
            self.action_closed_residual_gate_source = str(action_closed_residual_gate_source)
            self.action_closed_residual_gripper_threshold = float(action_closed_residual_gripper_threshold)
            self.action_closed_residual_gate_temperature = max(
                float(action_closed_residual_gate_temperature),
                1e-6,
            )
            self.action_closed_residual_hard_gate = bool(action_closed_residual_hard_gate)
            self.action_closed_residual_zero_gripper_dim = bool(action_closed_residual_zero_gripper_dim)
            self.lambda_action_closed_residual_l2 = float(lambda_action_closed_residual_l2)
            if action_loss_channel_weight is None:
                action_loss_weights = torch.ones(1, dtype=torch.float32)
                self.action_loss_uses_channel_weight = False
            else:
                action_loss_weights = torch.as_tensor(
                    list(action_loss_channel_weight),
                    dtype=torch.float32,
                )
                if action_loss_weights.ndim != 1:
                    raise ValueError("action_loss_channel_weight must be a 1D sequence")
                if action_dim is not None and action_loss_weights.numel() not in {
                    action_dim,
                    max(action_dim - 1, 1),
                }:
                    raise ValueError(
                        "action_loss_channel_weight must match action_dim or action_dim - 1 "
                        f"when gripper is ignored; got {action_loss_weights.numel()} values "
                        f"for action_dim={action_dim}"
                    )
                if torch.any(action_loss_weights < 0):
                    raise ValueError("action_loss_channel_weight values must be non-negative")
                if float(action_loss_weights.sum()) <= 0.0:
                    raise ValueError("action_loss_channel_weight must contain at least one positive value")
                self.action_loss_uses_channel_weight = True
            self.register_buffer(
                "action_loss_channel_weight",
                action_loss_weights.view(1, 1, -1),
            )
            self.register_buffer(
                "oat_norm_mean",
                torch.zeros(1, 1, invariant_dim),
            )
            self.register_buffer(
                "oat_norm_std",
                torch.ones(1, 1, invariant_dim),
            )
            if invariant_loss_channel_scale is None:
                inv_loss_scale = torch.ones(1, 1, invariant_dim)
                self.invariant_loss_uses_channel_scale = False
            else:
                inv_loss_scale = torch.as_tensor(
                    list(invariant_loss_channel_scale),
                    dtype=torch.float32,
                )
                if inv_loss_scale.numel() != invariant_dim:
                    raise ValueError(
                        "invariant_loss_channel_scale must match invariant_dim="
                        f"{invariant_dim}, got {inv_loss_scale.numel()} values"
                    )
                inv_loss_scale = inv_loss_scale.reshape(1, 1, invariant_dim).clamp_min(1e-6)
                self.invariant_loss_uses_channel_scale = True
            self.register_buffer("invariant_loss_channel_scale", inv_loss_scale)
            self.oat_uses_normalization = False

            if self.contact_mode not in {"event", "gripper_state"}:
                raise ValueError(f"Unsupported contact_mode={self.contact_mode!r}")
            if self.action_closed_residual_gate_source not in {"none", "state_gripper"}:
                raise ValueError(
                    "action_closed_residual_gate_source must be 'none' or 'state_gripper', "
                    f"got {self.action_closed_residual_gate_source!r}"
                )
            if self.oat_autoencode_target not in {"pred", "target", "both"}:
                raise ValueError(
                    "oat_autoencode_target must be 'pred', 'target', or 'both'"
                )
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
            self.committed_prefix_proj = nn.Linear(invariant_dim, hidden_dim)
            self.committed_prefix_bias = nn.Parameter(torch.zeros(1, 1, hidden_dim))

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
                    oat_checkpoint = None
                    if oat_checkpoint_path:
                        oat_checkpoint = torch.load(oat_checkpoint_path, map_location="cpu")
                    tokenizer_meta = (
                        oat_checkpoint.get("tokenizer", {})
                        if isinstance(oat_checkpoint, dict)
                        else {}
                    )
                    ckpt_normalization = (
                        oat_checkpoint.get("normalization")
                        if isinstance(oat_checkpoint, dict)
                        else None
                    )
                    ckpt_input_dim = tokenizer_meta.get("input_dim")
                    ckpt_input_horizon = tokenizer_meta.get("input_horizon")
                    if ckpt_input_dim is not None and int(ckpt_input_dim) != invariant_dim:
                        raise ValueError(
                            f"OAT checkpoint input_dim={ckpt_input_dim} does not match invariant_dim={invariant_dim}"
                        )
                    if ckpt_input_horizon is not None and int(ckpt_input_horizon) != action_horizon:
                        raise ValueError(
                            "OAT checkpoint input_horizon="
                            f"{ckpt_input_horizon} does not match action_horizon={action_horizon}"
                        )
                    tokenizer_kwargs = {
                        "input_dim": invariant_dim,
                        "input_horizon": action_horizon,
                        "emb_dim": int(tokenizer_meta.get("emb_dim", oat_emb_dim or hidden_dim)),
                        "latent_dim": int(tokenizer_meta.get("latent_dim", oat_latent_dim)),
                        "num_registers": int(tokenizer_meta.get("num_registers", oat_num_registers)),
                        "fsq_levels": tokenizer_meta.get("fsq_levels", oat_fsq_levels or [8, 5, 5, 5]),
                        "encoder_depth": int(tokenizer_meta.get("encoder_depth", oat_encoder_depth)),
                        "decoder_depth": int(tokenizer_meta.get("decoder_depth", oat_decoder_depth)),
                        "num_heads": int(tokenizer_meta.get("num_heads", oat_num_heads or num_heads)),
                    }
                    if oat_init_seed is None:
                        self.oat = OATTokenizer(**tokenizer_kwargs)
                    else:
                        with torch.random.fork_rng(devices=[]):
                            torch.manual_seed(int(oat_init_seed))
                            self.oat = OATTokenizer(**tokenizer_kwargs)
                    if oat_checkpoint is not None:
                        state_dict = (
                            oat_checkpoint.get("model_state_dict")
                            if isinstance(oat_checkpoint, dict)
                            else None
                        )
                        if state_dict is None:
                            state_dict = oat_checkpoint
                        self.oat.load_state_dict(state_dict, strict=True)
                        if ckpt_normalization is not None:
                            self._set_oat_normalization(ckpt_normalization)
                    if self.freeze_oat:
                        self.oat.requires_grad_(False)
                        self.oat.eval()
                except ImportError:
                    pass
            if self.oat is None and (
                self.supervise_oat_tokens or self.train_with_discrete_oat_tokens
            ):
                raise RuntimeError(
                    "Supervised/discrete OAT training requires use_oat=True and an importable OATTokenizer"
                )

            self.oat_token_query = None
            self.oat_token_decoder = None
            self.oat_token_head = None
            self.oat_token_context_proj = None
            if self.oat is not None and (
                self.supervise_oat_tokens or self.train_with_discrete_oat_tokens
            ):
                token_decoder_layer = nn.TransformerDecoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    batch_first=True,
                )
                self.oat_token_query = nn.Embedding(self.oat.num_registers, hidden_dim)
                self.oat_token_decoder = nn.TransformerDecoder(
                    token_decoder_layer,
                    num_layers=num_layers,
                )
                self.oat_token_head = nn.Linear(hidden_dim, self.oat.codebook_size)
                self.oat_token_context_proj = nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                nn.init.zeros_(self.oat_token_context_proj[-1].weight)
                nn.init.zeros_(self.oat_token_context_proj[-1].bias)

            # Optional: direct action prediction head (dual-head mode)
            self.action_head = None
            self.action_residual_adapter = None
            self.action_closed_residual_adapter = None
            if action_dim is not None:
                self.action_head = nn.Linear(hidden_dim, action_dim)
                if self.use_action_residual_adapter:
                    residual_hidden_dim = int(action_residual_hidden_dim or max(64, hidden_dim // 4))
                    self.action_residual_adapter = nn.Sequential(
                        nn.LayerNorm(hidden_dim),
                        nn.Linear(hidden_dim, residual_hidden_dim),
                        nn.GELU(),
                        nn.Linear(residual_hidden_dim, action_dim),
                    )
                    nn.init.zeros_(self.action_residual_adapter[-1].weight)
                    nn.init.zeros_(self.action_residual_adapter[-1].bias)
                if self.use_action_closed_residual_adapter:
                    closed_residual_hidden_dim = int(
                        action_closed_residual_hidden_dim or max(64, hidden_dim // 4)
                    )
                    self.action_closed_residual_adapter = nn.Sequential(
                        nn.LayerNorm(hidden_dim),
                        nn.Linear(hidden_dim, closed_residual_hidden_dim),
                        nn.GELU(),
                        nn.Linear(closed_residual_hidden_dim, action_dim),
                    )
                    nn.init.zeros_(self.action_closed_residual_adapter[-1].weight)
                    nn.init.zeros_(self.action_closed_residual_adapter[-1].bias)
            elif self.use_action_residual_adapter or self.use_action_closed_residual_adapter:
                raise ValueError("action residual adapters require action_dim to be set")

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

        def train(self, mode: bool = True):
            super().train(mode)
            if getattr(self, "freeze_oat", False) and self.oat is not None:
                self.oat.eval()
            return self

        def _set_oat_normalization(self, normalization: dict) -> None:
            if not normalization:
                return
            mean = torch.as_tensor(normalization["mean"], dtype=self.oat_norm_mean.dtype)
            std = torch.as_tensor(normalization["std"], dtype=self.oat_norm_std.dtype)
            if mean.numel() != self.invariant_dim or std.numel() != self.invariant_dim:
                raise ValueError(
                    "OAT normalization stats must match invariant_dim="
                    f"{self.invariant_dim}, got mean={mean.numel()} std={std.numel()}"
                )
            self.oat_norm_mean.copy_(mean.reshape(1, 1, -1))
            self.oat_norm_std.copy_(std.reshape(1, 1, -1).clamp_min(1e-8))
            self.oat_uses_normalization = True

        def _normalize_oat_invariants(self, invariants: torch.Tensor) -> torch.Tensor:
            if not self.oat_uses_normalization:
                return invariants
            mean = self.oat_norm_mean.to(device=invariants.device, dtype=invariants.dtype)
            std = self.oat_norm_std.to(device=invariants.device, dtype=invariants.dtype)
            return (invariants - mean) / std

        def _denormalize_oat_invariants(self, invariants: torch.Tensor) -> torch.Tensor:
            if not self.oat_uses_normalization:
                return invariants
            mean = self.oat_norm_mean.to(device=invariants.device, dtype=invariants.dtype)
            std = self.oat_norm_std.to(device=invariants.device, dtype=invariants.dtype)
            return invariants * std + mean

        def _tokenize_oat_invariants(self, invariants: torch.Tensor) -> torch.Tensor:
            if self.oat is None:
                raise RuntimeError("OAT tokenization requires an initialized OAT tokenizer")
            oat_input = self._normalize_oat_invariants(invariants)
            if self.freeze_oat:
                self.oat.eval()
                with torch.no_grad():
                    return self.oat.tokenize(oat_input)
            return self.oat.tokenize(oat_input)

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

        def _closed_residual_gate(
            self,
            state_flat: Optional[torch.Tensor],
            *,
            batch_size: int,
            device: torch.device,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            """Return a `(B, 1, 1)` gate for post-grasp residual corrections.

            The default gate uses the latest gripper qpos in the canonical state
            vector. It is intentionally object-agnostic: the adapter only sees
            whether the robot gripper is physically closed, not where any named
            object is in the scene.
            """
            if self.action_closed_residual_gate_source == "none":
                return torch.ones(batch_size, 1, 1, device=device, dtype=dtype)
            if state_flat is None:
                return torch.zeros(batch_size, 1, 1, device=device, dtype=dtype)

            latest_gripper_qpos = state_flat[:, -1].to(device=device, dtype=dtype)
            if self.action_closed_residual_hard_gate:
                return (
                    latest_gripper_qpos <= float(self.action_closed_residual_gripper_threshold)
                ).to(dtype=dtype).view(batch_size, 1, 1)
            closed_logit = (
                float(self.action_closed_residual_gripper_threshold) - latest_gripper_qpos
            ) / float(self.action_closed_residual_gate_temperature)
            return torch.sigmoid(closed_logit).view(batch_size, 1, 1)

        def _invariant_loss(self, pred_inv: torch.Tensor, target_invariants: torch.Tensor) -> torch.Tensor:
            """Compute invariant loss with optional terminal-step upweighting."""
            residual = self._scaled_invariant_residual(pred_inv, target_invariants)
            per_step_mse = residual.pow(2).mean(dim=-1)  # (B, T)
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

        def _scaled_invariant_residual(
            self,
            pred_inv: torch.Tensor,
            target_invariants: torch.Tensor,
        ) -> torch.Tensor:
            residual = pred_inv - target_invariants
            if not self.invariant_loss_uses_channel_scale:
                return residual
            scale = self.invariant_loss_channel_scale.to(
                device=residual.device,
                dtype=residual.dtype,
            )
            return residual / scale

        @staticmethod
        def _normalize_temporal_mask(
            loss_mask: Optional[torch.Tensor],
            *,
            batch_size: int,
            horizon: int,
            device: torch.device,
            dtype: torch.dtype,
        ) -> Optional[torch.Tensor]:
            if loss_mask is None:
                return None
            mask = loss_mask.to(device=device, dtype=dtype)
            if mask.ndim == 1:
                if mask.shape[0] != horizon:
                    raise ValueError(f"Expected 1D mask of length {horizon}, got {tuple(mask.shape)}")
                mask = mask.unsqueeze(0).expand(batch_size, -1)
            elif mask.ndim == 2:
                if mask.shape != (batch_size, horizon):
                    raise ValueError(
                        f"Expected 2D mask shape {(batch_size, horizon)}, got {tuple(mask.shape)}"
                    )
            elif mask.ndim == 3 and mask.shape[-1] == 1:
                if mask.shape[:2] != (batch_size, horizon):
                    raise ValueError(
                        f"Expected 3D mask shape {(batch_size, horizon, 1)}, got {tuple(mask.shape)}"
                    )
                mask = mask.squeeze(-1)
            else:
                raise ValueError(
                    f"Unsupported temporal mask shape {tuple(mask.shape)} for batch={batch_size} horizon={horizon}"
                )
            return mask

        @staticmethod
        def _masked_temporal_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
            if mask is None:
                return values.mean()
            weighted = values * mask
            denom = mask.sum().clamp_min(1.0)
            return weighted.sum() / denom

        @staticmethod
        def _masked_weighted_temporal_mean(
            values: torch.Tensor,
            mask: Optional[torch.Tensor],
            weights: Optional[torch.Tensor],
        ) -> torch.Tensor:
            if weights is None:
                return DHBActionHead._masked_temporal_mean(values, mask)

            weights = weights.to(device=values.device, dtype=values.dtype)
            while weights.ndim < values.ndim:
                weights = weights.unsqueeze(0)
            combined = weights if mask is None else weights * mask
            denom = combined.sum().clamp_min(1.0)
            return (values * combined).sum() / denom

        @staticmethod
        def _closed_binary_target(
            target: torch.Tensor,
            logits: torch.Tensor,
        ) -> torch.Tensor:
            """Normalize gripper-state targets for single-logit BCE heads.

            Dataset labels are usually categorical open/closed IDs with shape
            `(B, T)`. A BCE gripper-state head predicts the closed probability
            with shape `(B, T, 1)`, so categorical labels must be converted to a
            float closed-indicator channel before loss computation.
            """
            if logits.shape[-1] != 1:
                return target
            if target.ndim == logits.ndim - 1:
                return target.unsqueeze(-1)
            if target.ndim == logits.ndim and target.shape[-1] == 2:
                return target[..., 1:2]
            return target

        def _align_action_loss_inputs(
            self,
            pred_actions: torch.Tensor,
            target_actions: torch.Tensor,
            temporal_loss_mask: Optional[torch.Tensor],
            temporal_loss_weights: Optional[torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
            """Align auxiliary direct-action loss to DHB decode query slots.

            DHB-DR decode drops padded prefix frames, so decoded waypoint 0 is
            produced after consuming earlier invariant slots.  A positive shift
            supervises action target k on prediction slot k + shift and ignores
            the leading prediction slots for the auxiliary action loss.
            """
            shift = self.action_loss_target_shift
            pred_horizon = pred_actions.shape[1]
            target_horizon = target_actions.shape[1]
            if shift >= pred_horizon:
                raise ValueError(
                    "action_loss_target_shift must be smaller than the predicted action horizon "
                    f"({shift} >= {pred_horizon})"
                )
            aligned_horizon = min(pred_horizon - shift, target_horizon)
            if aligned_horizon <= 0:
                raise ValueError(
                    "No overlapping timesteps remain after action loss alignment: "
                    f"pred_horizon={pred_horizon}, target_horizon={target_horizon}, shift={shift}"
                )

            pred_for_loss = pred_actions[:, shift : shift + aligned_horizon, :]
            target_for_loss = target_actions[:, :aligned_horizon, :]
            mask_for_loss = (
                temporal_loss_mask[:, :aligned_horizon]
                if temporal_loss_mask is not None
                else None
            )
            weights_for_loss = (
                temporal_loss_weights[:aligned_horizon]
                if temporal_loss_weights is not None
                else None
            )
            return pred_for_loss, target_for_loss, mask_for_loss, weights_for_loss

        def _action_l1_per_step(
            self,
            pred_actions: torch.Tensor,
            target_actions: torch.Tensor,
        ) -> torch.Tensor:
            """Return per-step action L1 with optional channel weighting."""
            diff = (pred_actions - target_actions).abs()
            channel_weight = None
            if self.action_loss_uses_channel_weight:
                channel_weight = self.action_loss_channel_weight.to(
                    device=diff.device,
                    dtype=diff.dtype,
                )

            if self.action_loss_ignore_gripper and diff.shape[-1] >= 7:
                diff = diff[..., :6]
                if channel_weight is not None and channel_weight.shape[-1] > diff.shape[-1]:
                    channel_weight = channel_weight[..., :6]

            if channel_weight is None:
                return diff.mean(dim=-1)

            if channel_weight.shape[-1] != diff.shape[-1]:
                raise ValueError(
                    "action_loss_channel_weight length must match the action dimensions used "
                    f"for loss; got {channel_weight.shape[-1]} weights for {diff.shape[-1]} dims"
                )
            denom = channel_weight.sum(dim=-1).clamp_min(torch.finfo(diff.dtype).eps)
            return (diff * channel_weight).sum(dim=-1) / denom

        def _normalize_token_prefix_keep_k(
            self,
            token_prefix_keep_k: Optional[int | Sequence[int]],
            *,
            batch_size: int,
            device: torch.device,
        ) -> Optional[list[int]]:
            if token_prefix_keep_k is None:
                return None
            if self.oat is None:
                raise ValueError("token_prefix_keep_k requires an initialized OAT tokenizer")

            if isinstance(token_prefix_keep_k, int):
                keep_k = [int(token_prefix_keep_k)] * batch_size
            else:
                keep_k = [int(v) for v in token_prefix_keep_k]
                if len(keep_k) != batch_size:
                    raise ValueError(
                        f"Expected {batch_size} token prefix lengths, got {len(keep_k)}"
                    )

            max_tokens = int(self.oat.num_registers)
            normalized = []
            for value in keep_k:
                if value < 1 or value > max_tokens:
                    raise ValueError(
                        f"token_prefix_keep_k values must be in [1, {max_tokens}], got {value}"
                    )
                normalized.append(value)
            return normalized

        def _decode_oat_latents(
            self,
            latents: torch.Tensor,
            *,
            token_prefix_keep_k: Optional[int | Sequence[int]] = None,
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            if self.oat is None:
                raise RuntimeError("OAT latent decode requires an initialized OAT tokenizer")

            keep_k = self._normalize_token_prefix_keep_k(
                token_prefix_keep_k,
                batch_size=latents.shape[0],
                device=latents.device,
            )
            if keep_k is not None:
                masked_latents = latents.clone()
                positions = torch.arange(masked_latents.shape[1], device=latents.device).unsqueeze(0)
                keep_k_tensor = torch.as_tensor(keep_k, device=latents.device, dtype=torch.long)
                mask = positions >= keep_k_tensor.unsqueeze(1)
                mask_token = getattr(self.oat.nested_dropout, "mask_token", None)
                if mask_token is None:
                    masked_latents[mask] = 0
                else:
                    masked_latents[mask] = mask_token.to(device=latents.device, dtype=latents.dtype)
            else:
                masked_latents = latents
                keep_k_tensor = None

            recon = self._denormalize_oat_invariants(self.oat.decoder(masked_latents))
            return recon, keep_k_tensor

        def _decode_oat_tokens(
            self,
            tokens: torch.Tensor,
            *,
            token_prefix_keep_k: Optional[int | Sequence[int]] = None,
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            if self.oat is None:
                raise RuntimeError("OAT token decode requires an initialized OAT tokenizer")
            latents = self.oat.quantizer.indices_to_embedding(tokens).to(
                device=tokens.device,
                dtype=self.query_embed.weight.dtype,
            )
            return self._decode_oat_latents(
                latents,
                token_prefix_keep_k=token_prefix_keep_k,
            )

        def _resolve_target_oat_tokens(
            self,
            target_oat_tokens: Optional[torch.Tensor],
            target_invariants: Optional[torch.Tensor],
            *,
            device: torch.device,
        ) -> Optional[torch.Tensor]:
            if self.oat is None or self.oat_token_head is None:
                return None

            resolved = target_oat_tokens if self.use_exported_oat_tokens else None
            if resolved is None and target_invariants is not None:
                with torch.no_grad():
                    resolved = self._tokenize_oat_invariants(target_invariants.detach())
            if resolved is None:
                return None

            resolved = resolved.to(device=device)
            if resolved.ndim == 3:
                if not resolved.dtype.is_floating_point:
                    if resolved.shape[-1] != self.oat.num_registers:
                        raise ValueError(
                            "integer target_oat_tokens with shape (B, T, K) must use K="
                            f"{self.oat.num_registers}, got {tuple(resolved.shape)}"
                        )
                    resolved = resolved[:, 0, :]
                else:
                    if resolved.shape[-1] == self.oat.num_registers:
                        raise ValueError(
                            "floating target_oat_tokens shaped like a windowed tensor (B, T, K) "
                            "is ambiguous. Pass integer token IDs, one-hot (B, K, V), or omit "
                            "exported tokens so labels are derived from target_invariants."
                        )
                    if (
                        resolved.shape[1] == self.oat.num_registers
                        and resolved.shape[-1] == self.oat.codebook_size
                    ):
                        resolved = resolved.argmax(dim=-1)
                    else:
                        raise ValueError(
                            "floating target_oat_tokens must be one-hot with shape "
                            f"(B, K, {self.oat.codebook_size}), "
                            f"got {tuple(resolved.shape)}"
                        )
            if resolved.ndim != 2:
                raise ValueError(
                    "target_oat_tokens must have shape (B, K) or one-hot shape (B, K, V), "
                    f"got {tuple(resolved.shape)}"
                )
            if resolved.shape[1] != self.oat.num_registers:
                raise ValueError(
                    f"Expected {self.oat.num_registers} OAT tokens per sample, got {resolved.shape[1]}"
                )
            return resolved.to(dtype=torch.long)

        def _predict_oat_token_branch(
            self,
            decoded: torch.Tensor,
            *,
            token_prefix_keep_k: Optional[int | Sequence[int]] = None,
        ) -> Dict[str, torch.Tensor]:
            if self.oat is None or self.oat_token_head is None or self.oat_token_decoder is None:
                return {}

            batch_size = decoded.shape[0]
            token_query_pos = torch.arange(self.oat.num_registers, device=decoded.device)
            token_queries = self.oat_token_query(token_query_pos).unsqueeze(0).expand(batch_size, -1, -1)
            if self.oat_token_context_proj is not None:
                token_context = self.oat_token_context_proj(decoded.mean(dim=1))
                token_queries = token_queries + token_context.unsqueeze(1)
            token_features = self.oat_token_decoder(token_queries, decoded)
            token_logits = self.oat_token_head(token_features)
            token_probs = token_logits.softmax(dim=-1)
            pred_tokens = token_logits.argmax(dim=-1)

            hard_selector = F.one_hot(
                pred_tokens,
                num_classes=token_logits.shape[-1],
            ).to(dtype=token_probs.dtype)
            selector = token_probs + (hard_selector - token_probs).detach()
            codebook = self.oat.quantizer.implicit_codebook.to(
                device=token_logits.device,
                dtype=token_logits.dtype,
            )
            token_latents = torch.einsum("bkv,vd->bkd", selector, codebook)
            pred_inv_token_recon, keep_k_tensor = self._decode_oat_latents(
                token_latents,
                token_prefix_keep_k=token_prefix_keep_k,
            )

            result = {
                "pred_oat_token_logits": token_logits,
                "pred_oat_tokens": pred_tokens,
                "pred_invariants_token_recon": pred_inv_token_recon,
            }
            if keep_k_tensor is not None:
                result["token_prefix_keep_k"] = keep_k_tensor
            return result

        def _sample_oat_train_prefix_keep_k(
            self,
            *,
            batch_size: int,
            device: torch.device,
            explicit_keep_k: Optional[int | Sequence[int]],
        ) -> Optional[int | Sequence[int] | torch.Tensor]:
            if explicit_keep_k is not None:
                return explicit_keep_k
            if (
                self.oat is None
                or not self.training
                or self.oat_token_prefix_train_prob <= 0.0
                or torch.rand((), device=device).item() >= self.oat_token_prefix_train_prob
            ):
                return None
            min_k = max(1, min(int(self.oat_token_prefix_min_k), int(self.oat.num_registers)))
            return torch.randint(
                low=min_k,
                high=int(self.oat.num_registers) + 1,
                size=(batch_size,),
                device=device,
            )

        def forward(
            self,
            vlm_hidden: torch.Tensor,
            state: Optional[torch.Tensor] = None,
            target_invariants: Optional[torch.Tensor] = None,
            target_actions: Optional[torch.Tensor] = None,
            target_oat_tokens: Optional[torch.Tensor] = None,
            target_contact: Optional[torch.Tensor] = None,
            target_contact_phase: Optional[torch.Tensor] = None,
            target_gripper_state: Optional[torch.Tensor] = None,
            committed_prefix_invariants: Optional[torch.Tensor] = None,
            committed_prefix_mask: Optional[torch.Tensor] = None,
            loss_mask: Optional[torch.Tensor] = None,
            token_prefix_keep_k: Optional[int | Sequence[int]] = None,
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
            base_queries = self.query_embed(query_pos).unsqueeze(0).expand(B, -1, -1)
            queries = base_queries

            temporal_loss_mask = self._normalize_temporal_mask(
                loss_mask,
                batch_size=B,
                horizon=self.action_horizon,
                device=vlm_hidden.device,
                dtype=queries.dtype,
            )

            # Optionally condition on current state
            state_feat = None
            state_flat = None
            if state is not None:
                state_flat = self._flatten_state_history(state)
                state_feat = self.state_proj(state_flat).unsqueeze(1)  # (B, 1, hidden_dim)
                memory = torch.cat([state_feat, memory], dim=1)

            prefix_mask = self._normalize_temporal_mask(
                committed_prefix_mask,
                batch_size=B,
                horizon=self.action_horizon,
                device=vlm_hidden.device,
                dtype=queries.dtype,
            )
            if committed_prefix_invariants is not None:
                if committed_prefix_invariants.shape[:2] != (B, self.action_horizon):
                    raise ValueError(
                        "committed_prefix_invariants must have shape "
                        f"(B, T, {self.invariant_dim}), got {tuple(committed_prefix_invariants.shape)}"
                    )
                prefix_values = committed_prefix_invariants.to(device=vlm_hidden.device, dtype=queries.dtype)
                prefix_hidden = self.committed_prefix_proj(prefix_values)
                if prefix_mask is None:
                    prefix_mask = (
                        prefix_values.abs().sum(dim=-1) > 0
                    ).to(device=vlm_hidden.device, dtype=queries.dtype)
                queries = queries + prefix_hidden * prefix_mask.unsqueeze(-1)
                queries = queries + prefix_mask.unsqueeze(-1) * self.committed_prefix_bias
                result_prefix_mask = prefix_mask
            else:
                result_prefix_mask = prefix_mask

            # Cross-attention decode
            decoded = self.decoder(queries, memory)  # (B, T, hidden_dim)
            head_features = decoded
            if self.head_query_residual_scale != 0.0:
                head_features = head_features + base_queries * self.head_query_residual_scale

            # Project to invariant space
            pred_inv = self.invariant_head(head_features)  # (B, T, 8)

            # Optional OAT tokenization: compress to discrete tokens
            if self.oat is not None and self.training and self.lambda_oat_autoencode > 0.0:
                oat_loss_terms = []
                if self.oat_autoencode_target in {"pred", "both"}:
                    oat_loss_terms.append(self.oat(self._normalize_oat_invariants(pred_inv)))
                if self.oat_autoencode_target in {"target", "both"} and target_invariants is not None:
                    oat_loss_terms.append(
                        self.oat(
                            self._normalize_oat_invariants(
                                target_invariants.detach().to(dtype=pred_inv.dtype)
                            )
                        )
                    )
                oat_loss = sum(oat_loss_terms) / len(oat_loss_terms) if oat_loss_terms else None
                tokens = self._tokenize_oat_invariants(pred_inv)
                pred_inv_recon, _ = self._decode_oat_tokens(tokens)
            else:
                oat_loss = None
                pred_inv_recon = pred_inv

            oat_register_diversity_loss = None
            if self.oat is not None and self.training and self.lambda_oat_register_diversity > 0.0:
                diversity_terms = []
                if self.oat_autoencode_target in {"pred", "both"}:
                    diversity_terms.append(
                        self.oat.register_diversity_loss(
                            self._normalize_oat_invariants(pred_inv)
                        )
                    )
                if self.oat_autoencode_target in {"target", "both"} and target_invariants is not None:
                    diversity_terms.append(
                        self.oat.register_diversity_loss(
                            self._normalize_oat_invariants(
                                target_invariants.detach().to(dtype=pred_inv.dtype)
                            )
                        )
                    )
                if diversity_terms:
                    oat_register_diversity_loss = sum(diversity_terms) / len(diversity_terms)

            result = {"pred_invariants": pred_inv, "pred_invariants_recon": pred_inv_recon}
            if self.oat is not None:
                result["oat_loss"] = oat_loss
                if oat_register_diversity_loss is not None:
                    result["loss_oat_register_diversity"] = oat_register_diversity_loss
            train_token_prefix_keep_k = self._sample_oat_train_prefix_keep_k(
                batch_size=B,
                device=vlm_hidden.device,
                explicit_keep_k=token_prefix_keep_k,
            )
            token_result = self._predict_oat_token_branch(
                head_features,
                token_prefix_keep_k=train_token_prefix_keep_k,
            )
            if token_result:
                result.update(token_result)
            result["pred_invariants_execute"] = pred_inv
            if result_prefix_mask is not None:
                result["committed_prefix_mask"] = result_prefix_mask

            # Optional: predict raw actions via action_head
            pred_actions = None
            pred_action_residual = None
            pred_action_closed_residual = None
            pred_action_closed_residual_raw = None
            if self.action_head is not None:
                pred_actions_base = self.action_head(head_features)  # (B, T, action_dim)
                pred_actions = pred_actions_base
                if (
                    self.action_residual_adapter is not None
                    or self.action_closed_residual_adapter is not None
                ):
                    result["pred_actions_base"] = pred_actions_base
                if self.action_residual_adapter is not None:
                    pred_action_residual = (
                        self.action_residual_adapter(head_features) * self.action_residual_scale
                    )
                    pred_actions = pred_actions_base + pred_action_residual
                    result["pred_action_residual"] = pred_action_residual
                if self.action_closed_residual_adapter is not None:
                    closed_gate = self._closed_residual_gate(
                        state_flat,
                        batch_size=B,
                        device=head_features.device,
                        dtype=head_features.dtype,
                    )
                    pred_action_closed_residual_raw = (
                        self.action_closed_residual_adapter(head_features)
                        * self.action_closed_residual_scale
                    )
                    if (
                        self.action_closed_residual_zero_gripper_dim
                        and pred_action_closed_residual_raw.shape[-1] >= 7
                    ):
                        pred_action_closed_residual_raw = pred_action_closed_residual_raw.clone()
                        pred_action_closed_residual_raw[..., 6] = 0.0
                    pred_action_closed_residual = pred_action_closed_residual_raw * closed_gate
                    pred_actions = pred_actions + pred_action_closed_residual
                    result["pred_action_closed_residual_raw"] = pred_action_closed_residual_raw
                    result["pred_action_closed_residual"] = pred_action_closed_residual
                    result["action_closed_residual_gate"] = closed_gate.expand(
                        -1,
                        self.action_horizon,
                        -1,
                    )
                result["pred_actions"] = pred_actions

            pred_contact_logits = None
            pred_contact_phase_logits = None
            pred_gripper_state_logits = None
            if self.contact_head is not None or self.contact_phase_head is not None or self.gripper_state_head is not None:
                contact_inputs = [head_features]
                if self.contact_use_vlm_context:
                    contact_vlm_ctx, _ = self.contact_cross_attn(
                        head_features,
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
            resolved_target_oat_tokens = self._resolve_target_oat_tokens(
                target_oat_tokens,
                target_invariants,
                device=vlm_hidden.device,
            )
            pred_inv_for_loss = pred_inv
            if self.train_with_discrete_oat_tokens and "pred_invariants_token_recon" in result:
                pred_inv_for_loss = result["pred_invariants_token_recon"]

            total = None
            if target_invariants is not None:
                invariant_residual = self._scaled_invariant_residual(
                    pred_inv_for_loss,
                    target_invariants,
                )
                per_step_mse = invariant_residual.pow(2).mean(dim=-1)
                temporal_loss_weights = None
                uses_terminal_weights = self.terminal_loss_weight is not None and self.terminal_loss_weight > 1.0
                uses_prefix_weights = self.prefix_loss_steps > 0 and self.prefix_loss_weight != 1.0
                if uses_terminal_weights or uses_prefix_weights:
                    if uses_terminal_weights:
                        temporal_loss_weights = torch.linspace(
                            1.0,
                            float(self.terminal_loss_weight),
                            self.action_horizon,
                            device=pred_inv.device,
                            dtype=pred_inv.dtype,
                        )
                    else:
                        temporal_loss_weights = torch.ones(
                            self.action_horizon,
                            device=pred_inv.device,
                            dtype=pred_inv.dtype,
                        )
                    if uses_prefix_weights:
                        prefix_steps = min(self.prefix_loss_steps, self.action_horizon)
                        temporal_loss_weights[:prefix_steps] *= self.prefix_loss_weight
                    temporal_loss_weights = temporal_loss_weights / temporal_loss_weights.mean()
                loss_inv = self._masked_weighted_temporal_mean(
                    per_step_mse,
                    temporal_loss_mask,
                    temporal_loss_weights,
                )
                result["loss_inv"] = loss_inv
                with torch.no_grad():
                    raw_per_step_mse = (pred_inv_for_loss - target_invariants).pow(2).mean(dim=-1)
                    result["loss_inv_raw"] = self._masked_weighted_temporal_mean(
                        raw_per_step_mse,
                        temporal_loss_mask,
                        temporal_loss_weights,
                    )
                total = self.lambda_inv * loss_inv
                loss_inv_delta = None
                if self.lambda_invariant_delta > 0.0 and self.action_horizon > 1:
                    pred_delta = pred_inv_for_loss[:, 1:, :] - pred_inv_for_loss[:, :-1, :]
                    target_delta = target_invariants[:, 1:, :] - target_invariants[:, :-1, :]
                    delta_residual = self._scaled_invariant_residual(pred_delta, target_delta)
                    per_step_delta_mse = delta_residual.pow(2).mean(dim=-1)
                    delta_mask = None
                    if temporal_loss_mask is not None:
                        delta_mask = temporal_loss_mask[:, 1:] * temporal_loss_mask[:, :-1]
                    delta_weights = temporal_loss_weights[1:] if temporal_loss_weights is not None else None
                    loss_inv_delta = self._masked_weighted_temporal_mean(
                        per_step_delta_mse,
                        delta_mask,
                        delta_weights,
                    )
                    result["loss_inv_delta"] = loss_inv_delta
                    total = total + self.lambda_invariant_delta * loss_inv_delta
                if oat_loss is not None:
                    total = total + self.lambda_oat_autoencode * oat_loss
                if oat_register_diversity_loss is not None:
                    total = total + self.lambda_oat_register_diversity * oat_register_diversity_loss

                if target_actions is not None and pred_actions is not None:
                    # Dual-head mode: L1 action loss + invariant regularization
                    (
                        pred_actions_for_loss,
                        target_actions_for_loss,
                        action_loss_mask,
                        action_loss_weights,
                    ) = self._align_action_loss_inputs(
                        pred_actions,
                        target_actions,
                        temporal_loss_mask,
                        temporal_loss_weights,
                    )
                    per_step_l1 = self._action_l1_per_step(
                        pred_actions_for_loss,
                        target_actions_for_loss,
                    )
                    loss_act = self._masked_weighted_temporal_mean(
                        per_step_l1,
                        action_loss_mask,
                        action_loss_weights,
                    )
                    result["loss_act"] = loss_act
                    total = self.lambda_act * loss_act + self.lambda_inv * loss_inv
                    if loss_inv_delta is not None:
                        total = total + self.lambda_invariant_delta * loss_inv_delta
                    if oat_loss is not None:
                        total = total + self.lambda_oat_autoencode * oat_loss
                    if oat_register_diversity_loss is not None:
                        total = total + self.lambda_oat_register_diversity * oat_register_diversity_loss
                elif target_actions is not None:
                    # Legacy path: no action_head, placeholder loss
                    loss_act = torch.tensor(0.0, device=vlm_hidden.device)
                    result["loss_act"] = loss_act

            elif target_actions is not None and pred_actions is not None:
                (
                    pred_actions_for_loss,
                    target_actions_for_loss,
                    action_loss_mask,
                    _,
                ) = self._align_action_loss_inputs(
                    pred_actions,
                    target_actions,
                    temporal_loss_mask,
                    None,
                )
                per_step_l1 = self._action_l1_per_step(
                    pred_actions_for_loss,
                    target_actions_for_loss,
                )
                loss_act = self._masked_temporal_mean(per_step_l1, action_loss_mask)
                result["loss_act"] = loss_act
                total = self.lambda_act * loss_act
                if oat_register_diversity_loss is not None:
                    total = total + self.lambda_oat_register_diversity * oat_register_diversity_loss

            if (
                pred_action_residual is not None
                and self.lambda_action_residual_l2 > 0.0
                and target_actions is not None
            ):
                per_step_residual_l2 = pred_action_residual.pow(2).mean(dim=-1)
                loss_action_residual_l2 = self._masked_temporal_mean(
                    per_step_residual_l2,
                    temporal_loss_mask,
                )
                result["loss_action_residual_l2"] = loss_action_residual_l2
                total = (
                    self.lambda_action_residual_l2 * loss_action_residual_l2
                    if total is None
                    else total + self.lambda_action_residual_l2 * loss_action_residual_l2
                )

            if (
                pred_actions is not None
                and "pred_actions_base" in result
                and self.lambda_action_base_cumsum_l2 > 0.0
                and target_actions is not None
            ):
                base_actions = result["pred_actions_base"].detach()
                num_dims = min(
                    self.action_base_cumsum_dims,
                    pred_actions.shape[-1],
                    base_actions.shape[-1],
                )
                if num_dims > 0:
                    cumulative_delta = (
                        pred_actions[..., :num_dims] - base_actions[..., :num_dims]
                    ).cumsum(dim=1)
                    per_step_base_cumsum_l2 = cumulative_delta.pow(2).mean(dim=-1)
                    loss_action_base_cumsum_l2 = self._masked_temporal_mean(
                        per_step_base_cumsum_l2,
                        temporal_loss_mask,
                    )
                    result["loss_action_base_cumsum_l2"] = loss_action_base_cumsum_l2
                    total = (
                        self.lambda_action_base_cumsum_l2 * loss_action_base_cumsum_l2
                        if total is None
                        else total + self.lambda_action_base_cumsum_l2 * loss_action_base_cumsum_l2
                    )

            if (
                pred_action_closed_residual_raw is not None
                and self.lambda_action_closed_residual_l2 > 0.0
                and target_actions is not None
            ):
                per_step_closed_residual_l2 = pred_action_closed_residual_raw.pow(2).mean(dim=-1)
                loss_action_closed_residual_l2 = self._masked_temporal_mean(
                    per_step_closed_residual_l2,
                    temporal_loss_mask,
                )
                result["loss_action_closed_residual_l2"] = loss_action_closed_residual_l2
                total = (
                    self.lambda_action_closed_residual_l2 * loss_action_closed_residual_l2
                    if total is None
                    else total + self.lambda_action_closed_residual_l2 * loss_action_closed_residual_l2
                )

            if (
                resolved_target_oat_tokens is not None
                and "pred_oat_token_logits" in result
            ):
                token_logits = result["pred_oat_token_logits"]
                loss_oat_token = F.cross_entropy(
                    token_logits.reshape(-1, token_logits.shape[-1]),
                    resolved_target_oat_tokens.reshape(-1),
                )
                result["loss_oat_token"] = loss_oat_token
                pred_oat_tokens = result.get("pred_oat_tokens")
                if pred_oat_tokens is not None:
                    with torch.no_grad():
                        token_matches = (pred_oat_tokens == resolved_target_oat_tokens).to(dtype=token_logits.dtype)
                        result["oat_token_accuracy"] = token_matches.mean()
                        result["oat_pred_unique_count"] = torch.as_tensor(
                            float(torch.unique(pred_oat_tokens).numel()),
                            device=token_logits.device,
                            dtype=token_logits.dtype,
                        )
                        result["oat_target_unique_count"] = torch.as_tensor(
                            float(torch.unique(resolved_target_oat_tokens).numel()),
                            device=token_logits.device,
                            dtype=token_logits.dtype,
                        )
                        result["oat_pred_unique_sequence_count"] = torch.as_tensor(
                            float(torch.unique(pred_oat_tokens, dim=0).shape[0]),
                            device=token_logits.device,
                            dtype=token_logits.dtype,
                        )
                total = (
                    self.lambda_oat_token * loss_oat_token
                    if total is None
                    else total + self.lambda_oat_token * loss_oat_token
                )

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
                    per_step_contact = F.cross_entropy(
                        pred_contact_logits.reshape(-1, pred_contact_logits.shape[-1]),
                        target_contact_ce.reshape(-1),
                        weight=class_weight,
                        reduction="none",
                    )
                    per_step_contact = per_step_contact.reshape(pred_contact_logits.shape[0], pred_contact_logits.shape[1])
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
                    per_step_contact = F.binary_cross_entropy_with_logits(
                        pred_contact_logits,
                        target_contact,
                        pos_weight=pos_weight,
                        reduction="none",
                    )
                    per_step_contact = per_step_contact.mean(dim=-1)
                loss_contact = self._masked_temporal_mean(per_step_contact, temporal_loss_mask)
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
                    per_step_contact_phase = F.cross_entropy(
                        pred_contact_phase_logits.reshape(-1, pred_contact_phase_logits.shape[-1]),
                        target_contact_phase_ce.reshape(-1),
                        weight=phase_class_weight,
                        reduction="none",
                    )
                    per_step_contact_phase = per_step_contact_phase.reshape(
                        pred_contact_phase_logits.shape[0],
                        pred_contact_phase_logits.shape[1],
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
                    per_step_contact_phase = F.binary_cross_entropy_with_logits(
                        pred_contact_phase_logits,
                        target_contact_phase,
                        pos_weight=phase_pos_weight,
                        reduction="none",
                    )
                    per_step_contact_phase = per_step_contact_phase.mean(dim=-1)
                loss_contact_phase = self._masked_temporal_mean(per_step_contact_phase, temporal_loss_mask)
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
                    per_step_gripper_state = F.cross_entropy(
                        pred_gripper_state_logits.reshape(-1, pred_gripper_state_logits.shape[-1]),
                        target_gripper_state_ce.reshape(-1),
                        weight=class_weight,
                        reduction="none",
                    )
                    per_step_gripper_state = per_step_gripper_state.reshape(
                        pred_gripper_state_logits.shape[0],
                        pred_gripper_state_logits.shape[1],
                    )
                else:
                    target_gripper_state_bce = self._closed_binary_target(
                        target_gripper_state,
                        pred_gripper_state_logits,
                    ).to(
                        device=pred_gripper_state_logits.device,
                        dtype=pred_gripper_state_logits.dtype,
                    )
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
                    per_step_gripper_state = F.binary_cross_entropy_with_logits(
                        pred_gripper_state_logits,
                        target_gripper_state_bce,
                        pos_weight=pos_weight,
                        reduction="none",
                    )
                    per_step_gripper_state = per_step_gripper_state.mean(dim=-1)

                loss_gripper_state = self._masked_temporal_mean(per_step_gripper_state, temporal_loss_mask)
                result["loss_gripper_state"] = loss_gripper_state
                total = (
                    self.lambda_gripper_state * loss_gripper_state
                    if total is None
                    else total + self.lambda_gripper_state * loss_gripper_state
                )
                needs_gripper_prob_loss = (
                    self.lambda_gripper_state_smooth > 0.0
                    or self.lambda_gripper_state_margin > 0.0
                )
                if needs_gripper_prob_loss:
                    if self.gripper_state_loss_type == "cross_entropy":
                        closed_probs = pred_gripper_state_logits.softmax(dim=-1)[..., 1]
                        target_closed = target_gripper_state_ce.to(
                            device=pred_gripper_state_logits.device,
                            dtype=pred_gripper_state_logits.dtype,
                        )
                    else:
                        closed_probs = torch.sigmoid(pred_gripper_state_logits).squeeze(-1)
                        target_closed = target_gripper_state_bce.squeeze(-1).to(
                            device=pred_gripper_state_logits.device,
                            dtype=pred_gripper_state_logits.dtype,
                        )

                    if self.lambda_gripper_state_smooth > 0.0 and pred_gripper_state_logits.shape[1] > 1:
                        same_state = (target_closed[:, 1:] - target_closed[:, :-1]).abs() < 0.5
                        smooth_mask = same_state.to(dtype=pred_gripper_state_logits.dtype)
                        if temporal_loss_mask is not None:
                            smooth_mask = smooth_mask * temporal_loss_mask[:, 1:] * temporal_loss_mask[:, :-1]
                        per_step_smooth = (closed_probs[:, 1:] - closed_probs[:, :-1]).pow(2)
                        loss_gripper_state_smooth = self._masked_temporal_mean(per_step_smooth, smooth_mask)
                        result["loss_gripper_state_smooth"] = loss_gripper_state_smooth
                        total = (
                            self.lambda_gripper_state_smooth * loss_gripper_state_smooth
                            if total is None
                            else total + self.lambda_gripper_state_smooth * loss_gripper_state_smooth
                        )

                    if self.lambda_gripper_state_margin > 0.0:
                        closed_target = target_closed >= 0.5
                        closed_penalty = F.relu(self.gripper_state_closed_margin - closed_probs).pow(2)
                        open_penalty = F.relu(closed_probs - self.gripper_state_open_margin).pow(2)
                        per_step_margin = torch.where(closed_target, closed_penalty, open_penalty)
                        loss_gripper_state_margin = self._masked_temporal_mean(
                            per_step_margin,
                            temporal_loss_mask,
                        )
                        result["loss_gripper_state_margin"] = loss_gripper_state_margin
                        total = (
                            self.lambda_gripper_state_margin * loss_gripper_state_margin
                            if total is None
                            else total + self.lambda_gripper_state_margin * loss_gripper_state_margin
                        )

            if total is not None:
                result["loss"] = total

            return result

        def predict(
            self,
            vlm_hidden: torch.Tensor,
            state: Optional[torch.Tensor] = None,
            initial_pose: Optional[Dict[str, torch.Tensor]] = None,
            committed_prefix_invariants: Optional[torch.Tensor] = None,
            committed_prefix_mask: Optional[torch.Tensor] = None,
            use_discrete_tokens: bool = False,
            token_prefix_keep_k: Optional[int | Sequence[int]] = None,
            emit_oat_tokens: bool = False,
            tokenize_continuous_head: bool = False,
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
                result = self.forward(
                    vlm_hidden,
                    state,
                    committed_prefix_invariants=committed_prefix_invariants,
                    committed_prefix_mask=committed_prefix_mask,
                    token_prefix_keep_k=token_prefix_keep_k,
                )

            pred_inv_for_decode = result["pred_invariants"].detach()
            if self.oat is not None and (use_discrete_tokens or emit_oat_tokens or token_prefix_keep_k is not None):
                if self.oat_token_head is None or tokenize_continuous_head:
                    pred_inv_base = result["pred_invariants"].detach()
                    pred_tokens = self._tokenize_oat_invariants(pred_inv_base)
                    result["pred_oat_tokens"] = pred_tokens
                    pred_inv_token_recon, keep_k_tensor = self._decode_oat_tokens(
                        pred_tokens,
                        token_prefix_keep_k=token_prefix_keep_k,
                    )
                    pred_inv_token_recon = pred_inv_token_recon.detach()
                    result["pred_invariants_token_recon"] = pred_inv_token_recon
                    if keep_k_tensor is not None:
                        result["token_prefix_keep_k"] = keep_k_tensor
                if use_discrete_tokens:
                    pred_inv_for_decode = result["pred_invariants_token_recon"].detach()
                    result["pred_invariants_execute"] = pred_inv_for_decode
                else:
                    result["pred_invariants_execute"] = result["pred_invariants"].detach()
            else:
                for key in ("pred_oat_token_logits", "pred_oat_tokens", "pred_invariants_token_recon", "token_prefix_keep_k"):
                    result.pop(key, None)
                result["pred_invariants_execute"] = result["pred_invariants"].detach()

            result.pop("pred_oat_token_logits", None)

            # If action_head exists, pred_actions is already in result
            # (populated by forward())

            if initial_pose is not None:
                # Decode invariants to SE(3) trajectory from current pose
                # This uses the numpy DHB decoder (not differentiable)
                pred_inv = pred_inv_for_decode
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
