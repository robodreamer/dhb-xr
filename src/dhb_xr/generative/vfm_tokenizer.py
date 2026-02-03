"""
VFM Token Generator: End-to-end integration of tokenizer + flow matching.

Provides a unified interface for:
1. Encoding trajectories to continuous latent space
2. Training flow matching on latent representations
3. Generating diverse trajectories via flow matching sampling
4. Decoding back to invariants and SE(3) poses

Supports both deterministic (FlowMatcher) and variational (VariationalFlowMatcher)
generation, with optional conditioning on goals, context, or task embeddings.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, List, Tuple, Any, Union

from .flow_matching import FlowMatcher, VariationalFlowMatcher
from .latent_encoder import LatentEncoder, CategoricalLatentEncoder
from .sampling import ode_solve


class VFMTokenGenerator(nn.Module):
    """
    End-to-end variational flow matching generator for DHB invariants.

    Combines:
    - Tokenizer encoder: invariants -> continuous latent z
    - Flow matcher: generative model over z
    - Tokenizer decoder: z -> invariants

    Example:
        >>> tokenizer = DHBTokenizer(invariant_dim=8, latent_dim=32, codebook_size=512)
        >>> flow_matcher = VariationalFlowMatcher(latent_dim=32, condition_dim=16)
        >>> generator = VFMTokenGenerator(tokenizer, flow_matcher)
        >>>
        >>> # Train on invariant sequences
        >>> loss = generator.loss(invariants)
        >>> loss.backward()
        >>>
        >>> # Generate diverse trajectories
        >>> samples = generator.generate_multimodal(num_samples=4, seq_len=50)
    """

    def __init__(
        self,
        tokenizer: nn.Module,
        flow_matcher: Union[FlowMatcher, VariationalFlowMatcher],
        latent_encoder: Optional[LatentEncoder] = None,
        condition_encoder: Optional[nn.Module] = None,
    ):
        """
        Args:
            tokenizer: DHBTokenizer or ResidualVQTokenizer with encode_continuous()
                and decode_from_latent() methods.
            flow_matcher: FlowMatcher or VariationalFlowMatcher for generation.
            latent_encoder: Optional external latent encoder (overrides flow_matcher's
                internal encoder for VariationalFlowMatcher).
            condition_encoder: Optional encoder for external conditions (e.g., goal pose,
                task embedding).
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.flow_matcher = flow_matcher
        self.latent_encoder = latent_encoder
        self.condition_encoder = condition_encoder

        # Check if variational
        self.is_variational = isinstance(flow_matcher, VariationalFlowMatcher)

        # Get dimensions from tokenizer
        self.latent_dim = getattr(tokenizer, "latent_dim", None)
        if self.latent_dim is None:
            raise ValueError("Tokenizer must have latent_dim attribute")

    def encode(self, invariants: Tensor) -> Tensor:
        """
        Encode invariants to continuous latent space.

        Args:
            invariants: Input invariant sequences (B, T, invariant_dim).

        Returns:
            Continuous latent z (B, T, latent_dim).
        """
        return self.tokenizer.encode_continuous(invariants)

    def decode(self, z: Tensor) -> Tensor:
        """
        Decode continuous latent to invariants.

        Args:
            z: Continuous latent (B, T, latent_dim).

        Returns:
            Reconstructed invariants (B, T, invariant_dim).
        """
        return self.tokenizer.decode_from_latent(z)

    def encode_condition(self, condition: Any) -> Optional[Tensor]:
        """
        Encode external condition (if condition_encoder is provided).

        Args:
            condition: External condition (format depends on encoder).

        Returns:
            Condition embedding or None.
        """
        if self.condition_encoder is None or condition is None:
            return None
        return self.condition_encoder(condition)

    def forward(
        self,
        invariants: Tensor,
        condition: Optional[Any] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass: encode, flow match, decode.

        Args:
            invariants: Input invariant sequences (B, T, invariant_dim).
            condition: Optional external condition.

        Returns:
            Dictionary with 'z', 'z_reconstructed', 'invariants_reconstructed'.
        """
        # Encode to latent
        z = self.encode(invariants)

        # Encode condition if provided
        cond = self.encode_condition(condition)

        # Pass through flow matcher (for analysis, not generation)
        # During training, use loss() instead
        z_recon = z  # No-op for now

        # Decode back to invariants
        invariants_recon = self.decode(z)

        return {
            "z": z,
            "z_reconstructed": z_recon,
            "invariants_reconstructed": invariants_recon,
        }

    def loss(
        self,
        invariants: Tensor,
        condition: Optional[Any] = None,
        tokenizer_weight: float = 1.0,
        flow_weight: float = 1.0,
        beta: float = 0.01,
    ) -> Dict[str, Tensor]:
        """
        Compute joint training loss.

        Combines:
        - Tokenizer reconstruction loss
        - Flow matching velocity loss
        - KL divergence (for variational models)

        Args:
            invariants: Input invariant sequences (B, T, invariant_dim).
            condition: Optional external condition.
            tokenizer_weight: Weight for tokenizer loss.
            flow_weight: Weight for flow matching loss.
            beta: KL divergence weight (for variational models).

        Returns:
            Dictionary with 'total', 'tokenizer', 'flow', and optionally 'kl'.
        """
        # Encode to latent
        z_1 = self.encode(invariants)  # "data" distribution

        # Tokenizer loss (reconstruction through VQ)
        indices, reconstructed, z, z_q = self.tokenizer(invariants)
        tokenizer_loss = self.tokenizer.loss(invariants, reconstructed, z, z_q)

        # Flow matching loss
        if self.is_variational:
            flow_losses = self.flow_matcher.loss(z_1, beta=beta)
            flow_loss = flow_losses["total"]
            kl_loss = flow_losses["kl"]
        else:
            flow_loss = self.flow_matcher.loss(z_1)
            kl_loss = torch.tensor(0.0, device=invariants.device)

        # Total loss
        total_loss = tokenizer_weight * tokenizer_loss + flow_weight * flow_loss

        return {
            "total": total_loss,
            "tokenizer": tokenizer_loss,
            "flow": flow_loss,
            "kl": kl_loss,
        }

    def generate(
        self,
        num_samples: int = 1,
        seq_len: int = 50,
        num_steps: int = 10,
        method: str = "euler",
        condition: Optional[Any] = None,
        w: Optional[Tensor] = None,
        device: str = "cpu",
        return_latent: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        """
        Generate invariant sequences via flow matching.

        Args:
            num_samples: Number of samples to generate.
            seq_len: Sequence length.
            num_steps: ODE integration steps.
            method: ODE solver ('euler' or 'rk4').
            condition: Optional external condition.
            w: Optional conditioning latent for variational models.
            device: Device for computation.
            return_latent: If True, also return the generated latent z.

        Returns:
            Generated invariants (num_samples, seq_len, invariant_dim),
            optionally with latent z.
        """
        # Generate latent via flow matching
        if self.is_variational:
            z = self.flow_matcher.sample(
                num_samples=num_samples,
                seq_len=seq_len,
                num_steps=num_steps,
                method=method,
                w=w,
                device=device,
            )
        else:
            z = self.flow_matcher.sample(
                num_samples=num_samples,
                seq_len=seq_len,
                num_steps=num_steps,
                method=method,
                device=device,
            )

        # Decode to invariants
        invariants = self.decode(z)

        if return_latent:
            return invariants, z
        return invariants

    def generate_multimodal(
        self,
        num_samples: int = 1,
        seq_len: int = 50,
        num_modes: int = 4,
        num_steps: int = 10,
        method: str = "euler",
        condition: Optional[Any] = None,
        device: str = "cpu",
    ) -> Tensor:
        """
        Generate diverse samples by sampling different conditioning latents.

        Args:
            num_samples: Number of samples per mode.
            seq_len: Sequence length.
            num_modes: Number of distinct modes to sample.
            num_steps: ODE integration steps.
            method: ODE solver method.
            condition: Optional external condition.
            device: Device for computation.

        Returns:
            Generated invariants (num_modes * num_samples, seq_len, invariant_dim).
        """
        if not self.is_variational:
            # For non-variational, just generate multiple samples
            return self.generate(
                num_samples=num_modes * num_samples,
                seq_len=seq_len,
                num_steps=num_steps,
                method=method,
                device=device,
            )

        # Generate with different w for each mode
        z = self.flow_matcher.sample_multimodal(
            num_samples=num_samples,
            seq_len=seq_len,
            num_modes=num_modes,
            num_steps=num_steps,
            method=method,
            device=device,
        )

        return self.decode(z)

    def generate_continuation(
        self,
        prefix_invariants: Tensor,
        continuation_len: int,
        num_modes: int = 1,
        num_steps: int = 10,
        method: str = "euler",
    ) -> Tensor:
        """
        Generate trajectory continuations from a prefix.

        Args:
            prefix_invariants: Prefix invariant sequence (B, prefix_len, invariant_dim).
            continuation_len: Length of continuation to generate.
            num_modes: Number of continuation modes (for variational).
            num_steps: ODE integration steps.
            method: ODE solver method.

        Returns:
            Continued trajectories (B * num_modes, prefix_len + continuation_len, invariant_dim).
        """
        B, prefix_len, _ = prefix_invariants.shape
        device = prefix_invariants.device
        total_len = prefix_len + continuation_len

        # Encode prefix to latent
        z_prefix = self.encode(prefix_invariants)

        if self.is_variational:
            # Sample continuations with different w
            z_full = self.flow_matcher.sample_continuation(
                z_prefix=z_prefix,
                prefix_len=prefix_len,
                total_len=total_len,
                num_modes=num_modes,
                num_steps=num_steps,
                method=method,
            )
        else:
            # Generate full sequences and keep prefix + continuation
            all_z = []
            for _ in range(num_modes):
                z = self.flow_matcher.sample(
                    num_samples=B,
                    seq_len=total_len,
                    num_steps=num_steps,
                    method=method,
                    device=device,
                )
                # Blend: use prefix from input, continuation from generated
                z_out = torch.cat([z_prefix, z[:, prefix_len:]], dim=1)
                all_z.append(z_out)
            z_full = torch.cat(all_z, dim=0)

        return self.decode(z_full)

    def interpolate_latent(
        self,
        invariants_a: Tensor,
        invariants_b: Tensor,
        num_interp: int = 5,
    ) -> Tensor:
        """
        Interpolate between two invariant sequences in latent space.

        Args:
            invariants_a: First sequence (1, T, invariant_dim).
            invariants_b: Second sequence (1, T, invariant_dim).
            num_interp: Number of interpolation steps.

        Returns:
            Interpolated invariants (num_interp, T, invariant_dim).
        """
        z_a = self.encode(invariants_a)
        z_b = self.encode(invariants_b)

        # Linear interpolation in latent space
        alphas = torch.linspace(0, 1, num_interp, device=z_a.device)
        z_interp = []
        for alpha in alphas:
            z = (1 - alpha) * z_a + alpha * z_b
            z_interp.append(z)

        z_interp = torch.cat(z_interp, dim=0)
        return self.decode(z_interp)


class ConditionalVFMGenerator(VFMTokenGenerator):
    """
    VFM generator with explicit goal/context conditioning.

    Extends VFMTokenGenerator with:
    - Goal pose conditioning
    - Task embedding conditioning
    - Context encoder for vision/language inputs
    """

    def __init__(
        self,
        tokenizer: nn.Module,
        flow_matcher: VariationalFlowMatcher,
        goal_dim: int = 7,  # position (3) + quaternion (4)
        context_dim: int = 0,
        hidden_dim: int = 128,
    ):
        """
        Args:
            tokenizer: DHBTokenizer or RVQ with encode_continuous/decode_from_latent.
            flow_matcher: VariationalFlowMatcher.
            goal_dim: Dimension of goal pose (default: 7 for pos + quat).
            context_dim: Dimension of external context (e.g., vision embedding).
            hidden_dim: Hidden dimension for condition encoding.
        """
        # Create condition encoder
        cond_input_dim = goal_dim + context_dim
        condition_encoder = nn.Sequential(
            nn.Linear(cond_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, flow_matcher.condition_dim),
        ) if cond_input_dim > 0 else None

        super().__init__(
            tokenizer=tokenizer,
            flow_matcher=flow_matcher,
            condition_encoder=condition_encoder,
        )

        self.goal_dim = goal_dim
        self.context_dim = context_dim

    def generate_to_goal(
        self,
        goal_pose: Tensor,
        num_samples: int = 1,
        seq_len: int = 50,
        num_steps: int = 10,
        method: str = "euler",
        context: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Generate trajectories conditioned on goal pose.

        Args:
            goal_pose: Goal pose (B, 7) with position and quaternion.
            num_samples: Samples per goal.
            seq_len: Sequence length.
            num_steps: ODE integration steps.
            method: ODE solver.
            context: Optional external context.

        Returns:
            Generated invariants reaching the goal.
        """
        B = goal_pose.shape[0]
        device = goal_pose.device

        # Build condition
        if context is not None:
            cond_input = torch.cat([goal_pose, context], dim=-1)
        else:
            cond_input = goal_pose

        # Encode condition to w
        w = self.condition_encoder(cond_input)

        # Expand for num_samples
        if num_samples > 1:
            w = w.unsqueeze(1).expand(-1, num_samples, -1)
            w = w.reshape(B * num_samples, -1)

        # Generate with goal conditioning
        z = self.flow_matcher.sample(
            num_samples=B * num_samples,
            seq_len=seq_len,
            num_steps=num_steps,
            method=method,
            w=w,
            device=device,
        )

        return self.decode(z)
