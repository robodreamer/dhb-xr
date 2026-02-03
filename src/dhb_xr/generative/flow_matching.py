"""
Variational Flow Matching for DHB-Token generation.

Implements:
- FlowMatcher: Base flow matching (FM/RFM) for deterministic velocity regression
- VariationalFlowMatcher: V-RFM with latent conditioning for multi-modal generation

The variational extension addresses the "mode averaging" problem in standard FM:
when multiple valid trajectories exist from the same state, a deterministic velocity
regressor averages modes, producing implausible intermediate behavior. By conditioning
on a latent variable w, V-RFM can represent and sample from multiple modes.

Reference: "Flow Matching for Generative Modeling" (Lipman et al., 2023)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Dict, Any

from .sampling import ode_solve


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for time t in [0, 1].

    Maps scalar time to a high-dimensional representation using
    sinusoidal functions at different frequencies.
    """

    def __init__(self, dim: int, max_period: float = 10000.0):
        """
        Args:
            dim: Output embedding dimension (should be even).
            max_period: Maximum period for sinusoidal functions.
        """
        super().__init__()
        self.dim = dim
        self.max_period = max_period

        # Precompute frequency factors
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer("freqs", freqs)

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: Time values (B,) or scalar, in [0, 1].

        Returns:
            Time embeddings (B, dim) or (dim,).
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)

        # Scale t to match frequency range
        args = t.unsqueeze(-1) * self.freqs * 2 * math.pi

        # Concatenate sin and cos
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return embedding


class VelocityNetwork(nn.Module):
    """
    Neural network predicting the velocity field v_theta(z_t, t, w).

    Architecture: MLP with residual connections, processing temporal sequences.
    Supports optional latent conditioning for variational flow matching.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        time_embed_dim: int = 64,
        condition_dim: int = 0,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        """
        Args:
            latent_dim: Dimension of the latent space z.
            hidden_dim: Hidden layer dimension.
            num_layers: Number of residual blocks.
            time_embed_dim: Dimension of time embedding.
            condition_dim: Dimension of conditioning latent w (0 for unconditional).
            dropout: Dropout rate.
            use_layer_norm: Whether to use layer normalization.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        # Input projection: z_t + time_embed + (optional) condition
        input_dim = latent_dim + time_embed_dim + condition_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Residual blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            block = nn.Sequential(
                nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Dropout(dropout),
            )
            self.blocks.append(block)

        # Output projection to velocity
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self,
        z_t: Tensor,
        t: Tensor | float,
        w: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Predict velocity at state z_t and time t.

        Args:
            z_t: Current state (B, T, D) or (B, D).
            t: Time (B,) or scalar in [0, 1].
            w: Optional conditioning latent (B, condition_dim).

        Returns:
            Predicted velocity v_theta (same shape as z_t).
        """
        # Handle scalar time
        if isinstance(t, float):
            t = torch.tensor([t], device=z_t.device, dtype=z_t.dtype)
        if t.dim() == 0:
            t = t.unsqueeze(0)

        # Get batch size and handle 2D/3D input
        is_3d = z_t.dim() == 3
        if is_3d:
            B, T, D = z_t.shape
            z_flat = z_t.reshape(B * T, D)
        else:
            B, D = z_t.shape
            T = 1
            z_flat = z_t

        # Time embedding (broadcast to all timesteps)
        t_embed = self.time_embed(t)  # (B, time_embed_dim)
        if is_3d:
            t_embed = t_embed.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
        else:
            t_embed = t_embed.expand(B, -1)

        # Build input
        inputs = [z_flat, t_embed]

        # Add conditioning if provided
        if w is not None:
            if is_3d:
                w_expanded = w.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
            else:
                w_expanded = w
            inputs.append(w_expanded)

        x = torch.cat(inputs, dim=-1)
        x = self.input_proj(x)

        # Residual blocks
        for block in self.blocks:
            x = x + block(x)

        # Output velocity
        v = self.output_proj(x)

        # Reshape back if needed
        if is_3d:
            v = v.reshape(B, T, D)

        return v


class FlowMatcher(nn.Module):
    """
    Base Flow Matching model for trajectory generation.

    Learns a velocity field that transports samples from noise (z_0)
    to data (z_1) along straight interpolation paths.

    Training objective:
        L = E[||v_theta(z_t, t) - (z_1 - z_0)||^2]

    where z_t = (1-t)*z_0 + t*z_1 is the linear interpolation.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        time_embed_dim: int = 64,
        dropout: float = 0.1,
    ):
        """
        Args:
            latent_dim: Dimension of the latent space.
            hidden_dim: Hidden layer dimension.
            num_layers: Number of residual blocks.
            time_embed_dim: Dimension of time embedding.
            dropout: Dropout rate.
        """
        super().__init__()
        self.latent_dim = latent_dim

        self.velocity_net = VelocityNetwork(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            time_embed_dim=time_embed_dim,
            condition_dim=0,
            dropout=dropout,
        )

    def forward(self, z_t: Tensor, t: Tensor | float) -> Tensor:
        """Predict velocity at state z_t and time t."""
        return self.velocity_net(z_t, t)

    def interpolate(self, z_0: Tensor, z_1: Tensor, t: Tensor | float) -> Tensor:
        """
        Linear interpolation between noise (z_0) and data (z_1).

        Args:
            z_0: Noise samples (B, T, D) or (B, D).
            z_1: Data samples (same shape).
            t: Interpolation time in [0, 1].

        Returns:
            Interpolated state z_t.
        """
        if isinstance(t, float):
            return (1 - t) * z_0 + t * z_1

        # Handle tensor t with proper broadcasting
        if z_0.dim() == 3:
            t = t.view(-1, 1, 1)
        else:
            t = t.view(-1, 1)

        return (1 - t) * z_0 + t * z_1

    def target_velocity(self, z_0: Tensor, z_1: Tensor) -> Tensor:
        """
        Compute target velocity for flow matching (straight path).

        The optimal transport direction is simply z_1 - z_0.
        """
        return z_1 - z_0

    def loss(
        self,
        z_1: Tensor,
        z_0: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute flow matching loss.

        Args:
            z_1: Data samples (B, T, D) or (B, D).
            z_0: Noise samples. If None, sampled from N(0, 1).

        Returns:
            Scalar loss value.
        """
        # Sample noise if not provided
        if z_0 is None:
            z_0 = torch.randn_like(z_1)

        # Sample random time
        B = z_1.shape[0]
        t = torch.rand(B, device=z_1.device, dtype=z_1.dtype)

        # Interpolate
        z_t = self.interpolate(z_0, z_1, t)

        # Target velocity
        v_target = self.target_velocity(z_0, z_1)

        # Predicted velocity
        v_pred = self.forward(z_t, t)

        # MSE loss
        loss = F.mse_loss(v_pred, v_target)

        return loss

    def sample(
        self,
        num_samples: int,
        seq_len: int,
        num_steps: int = 10,
        method: str = "euler",
        device: str = "cpu",
        return_trajectory: bool = False,
    ) -> Tensor | Tuple[Tensor, list]:
        """
        Generate samples by solving the flow ODE from noise to data.

        Args:
            num_samples: Number of samples to generate.
            seq_len: Sequence length T.
            num_steps: Number of ODE integration steps.
            method: ODE solver ('euler' or 'rk4').
            device: Device for computation.
            return_trajectory: If True, return intermediate states.

        Returns:
            Generated samples (num_samples, seq_len, latent_dim).
        """
        # Start from noise
        z_0 = torch.randn(num_samples, seq_len, self.latent_dim, device=device)

        # Define velocity function for ODE solver
        def velocity_fn(z: Tensor, t: float) -> Tensor:
            t_tensor = torch.tensor([t], device=z.device, dtype=z.dtype).expand(z.shape[0])
            return self.forward(z, t_tensor)

        # Solve ODE from t=0 to t=1
        return ode_solve(
            velocity_fn,
            z_0,
            t_start=0.0,
            t_end=1.0,
            num_steps=num_steps,
            method=method,
            return_trajectory=return_trajectory,
        )


class VariationalFlowMatcher(nn.Module):
    """
    Variational Flow Matching (V-RFM) for multi-modal trajectory generation.

    Extends FlowMatcher with a latent variable w that conditions the velocity
    field, enabling representation of multiple modes in the trajectory space.

    Training objective:
        L = E[||v_theta(z_t, t, w) - (z_1 - z_0)||^2] + beta * KL(q(w|z_t,t) || p(w))

    At inference, sampling different w values produces diverse trajectories.
    """

    def __init__(
        self,
        latent_dim: int,
        condition_dim: int = 16,
        hidden_dim: int = 256,
        num_layers: int = 4,
        time_embed_dim: int = 64,
        dropout: float = 0.1,
        prior_std: float = 1.0,
    ):
        """
        Args:
            latent_dim: Dimension of the trajectory latent space z.
            condition_dim: Dimension of the conditioning latent w.
            hidden_dim: Hidden layer dimension.
            num_layers: Number of residual blocks.
            time_embed_dim: Dimension of time embedding.
            dropout: Dropout rate.
            prior_std: Standard deviation of prior p(w) = N(0, prior_std^2).
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.prior_std = prior_std

        # Velocity network with conditioning
        self.velocity_net = VelocityNetwork(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            time_embed_dim=time_embed_dim,
            condition_dim=condition_dim,
            dropout=dropout,
        )

        # Posterior encoder q(w | z_t, t)
        # Simple MLP that maps (z_t, t) to mean and log_std of w
        self.posterior_net = nn.Sequential(
            nn.Linear(latent_dim + time_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, condition_dim * 2),  # mean and log_std
        )

        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

    def encode_posterior(
        self,
        z_t: Tensor,
        t: Tensor | float,
    ) -> Tuple[Tensor, Tensor]:
        """
        Encode posterior q(w | z_t, t).

        Args:
            z_t: Current state (B, T, D) or (B, D).
            t: Time (B,) or scalar.

        Returns:
            Tuple of (mean, log_std) for the posterior distribution.
        """
        # Handle scalar time
        if isinstance(t, float):
            t = torch.tensor([t], device=z_t.device, dtype=z_t.dtype)
        if t.dim() == 0:
            t = t.unsqueeze(0)

        # Pool over sequence if 3D
        if z_t.dim() == 3:
            z_pooled = z_t.mean(dim=1)  # (B, D)
        else:
            z_pooled = z_t

        # Time embedding
        t_embed = self.time_embed(t)  # (B, time_embed_dim)

        # Concatenate and encode
        x = torch.cat([z_pooled, t_embed], dim=-1)
        out = self.posterior_net(x)

        mean, log_std = out.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, min=-10, max=2)  # Stability

        return mean, log_std

    def sample_posterior(
        self,
        z_t: Tensor,
        t: Tensor | float,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Sample from posterior using reparameterization trick.

        Returns:
            Tuple of (w, mean, log_std).
        """
        mean, log_std = self.encode_posterior(z_t, t)
        std = torch.exp(log_std)

        # Reparameterization
        eps = torch.randn_like(mean)
        w = mean + std * eps

        return w, mean, log_std

    def kl_divergence(self, mean: Tensor, log_std: Tensor) -> Tensor:
        """
        Compute KL divergence from posterior to prior.

        KL(N(mean, std) || N(0, prior_std))
        """
        std = torch.exp(log_std)
        prior_var = self.prior_std ** 2

        kl = 0.5 * (
            (std ** 2 + mean ** 2) / prior_var
            - 1
            - 2 * log_std
            + 2 * math.log(self.prior_std)
        )

        return kl.sum(dim=-1).mean()

    def forward(
        self,
        z_t: Tensor,
        t: Tensor | float,
        w: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Predict velocity at state z_t and time t.

        If w is None during training, it's sampled from the posterior.
        """
        return self.velocity_net(z_t, t, w)

    def interpolate(self, z_0: Tensor, z_1: Tensor, t: Tensor | float) -> Tensor:
        """Linear interpolation between noise and data."""
        if isinstance(t, float):
            return (1 - t) * z_0 + t * z_1

        if z_0.dim() == 3:
            t = t.view(-1, 1, 1)
        else:
            t = t.view(-1, 1)

        return (1 - t) * z_0 + t * z_1

    def target_velocity(self, z_0: Tensor, z_1: Tensor) -> Tensor:
        """Target velocity (straight path)."""
        return z_1 - z_0

    def loss(
        self,
        z_1: Tensor,
        z_0: Optional[Tensor] = None,
        beta: float = 0.01,
    ) -> Dict[str, Tensor]:
        """
        Compute V-RFM loss with KL regularization.

        Args:
            z_1: Data samples (B, T, D) or (B, D).
            z_0: Noise samples. If None, sampled from N(0, 1).
            beta: Weight for KL divergence term.

        Returns:
            Dictionary with 'total', 'reconstruction', and 'kl' losses.
        """
        if z_0 is None:
            z_0 = torch.randn_like(z_1)

        B = z_1.shape[0]
        t = torch.rand(B, device=z_1.device, dtype=z_1.dtype)

        # Interpolate
        z_t = self.interpolate(z_0, z_1, t)

        # Sample from posterior
        w, mean, log_std = self.sample_posterior(z_t, t)

        # Target velocity
        v_target = self.target_velocity(z_0, z_1)

        # Predicted velocity (conditioned on w)
        v_pred = self.forward(z_t, t, w)

        # Reconstruction loss
        recon_loss = F.mse_loss(v_pred, v_target)

        # KL divergence
        kl_loss = self.kl_divergence(mean, log_std)

        # Total loss
        total_loss = recon_loss + beta * kl_loss

        return {
            "total": total_loss,
            "reconstruction": recon_loss,
            "kl": kl_loss,
        }

    def sample(
        self,
        num_samples: int,
        seq_len: int,
        num_steps: int = 10,
        method: str = "euler",
        w: Optional[Tensor] = None,
        device: str = "cpu",
        return_trajectory: bool = False,
    ) -> Tensor | Tuple[Tensor, list]:
        """
        Generate samples by solving the flow ODE.

        Args:
            num_samples: Number of samples to generate.
            seq_len: Sequence length T.
            num_steps: Number of ODE integration steps.
            method: ODE solver ('euler' or 'rk4').
            w: Conditioning latent. If None, sampled from prior.
            device: Device for computation.
            return_trajectory: If True, return intermediate states.

        Returns:
            Generated samples (num_samples, seq_len, latent_dim).
        """
        # Sample w from prior if not provided
        if w is None:
            w = torch.randn(num_samples, self.condition_dim, device=device) * self.prior_std

        # Start from noise
        z_0 = torch.randn(num_samples, seq_len, self.latent_dim, device=device)

        # Define velocity function
        def velocity_fn(z: Tensor, t: float) -> Tensor:
            t_tensor = torch.tensor([t], device=z.device, dtype=z.dtype).expand(z.shape[0])
            return self.forward(z, t_tensor, w)

        return ode_solve(
            velocity_fn,
            z_0,
            t_start=0.0,
            t_end=1.0,
            num_steps=num_steps,
            method=method,
            return_trajectory=return_trajectory,
        )

    def sample_multimodal(
        self,
        num_samples: int,
        seq_len: int,
        num_modes: int = 4,
        num_steps: int = 10,
        method: str = "euler",
        device: str = "cpu",
    ) -> Tensor:
        """
        Generate diverse samples by sampling different w values.

        Args:
            num_samples: Number of samples per mode.
            seq_len: Sequence length.
            num_modes: Number of distinct modes to sample.
            num_steps: ODE integration steps.
            method: ODE solver method.
            device: Device for computation.

        Returns:
            Generated samples (num_modes * num_samples, seq_len, latent_dim).
        """
        all_samples = []

        for _ in range(num_modes):
            # Sample a different w for each mode
            w = torch.randn(num_samples, self.condition_dim, device=device) * self.prior_std
            samples = self.sample(
                num_samples, seq_len, num_steps, method, w, device
            )
            all_samples.append(samples)

        return torch.cat(all_samples, dim=0)

    def sample_continuation(
        self,
        z_prefix: Tensor,
        prefix_len: int,
        total_len: int,
        num_modes: int = 1,
        num_steps: int = 10,
        method: str = "euler",
    ) -> Tensor:
        """
        Generate trajectory continuations from a prefix.

        Args:
            z_prefix: Prefix latent sequence (B, prefix_len, D).
            prefix_len: Length of prefix to condition on.
            total_len: Total output sequence length.
            num_modes: Number of continuation modes to generate.
            num_steps: ODE integration steps.
            method: ODE solver method.

        Returns:
            Continued sequences (B * num_modes, total_len, D).
        """
        B = z_prefix.shape[0]
        device = z_prefix.device
        continuation_len = total_len - prefix_len

        all_continuations = []

        for _ in range(num_modes):
            # Sample w from prior
            w = torch.randn(B, self.condition_dim, device=device) * self.prior_std

            # Start from noise for continuation part
            z_cont_noise = torch.randn(B, continuation_len, self.latent_dim, device=device)

            # Full sequence: prefix (fixed) + continuation (to be generated)
            # For now, simple approach: generate full sequence conditioned on w
            # and use the continuation part
            z_0 = torch.randn(B, total_len, self.latent_dim, device=device)

            def velocity_fn(z: Tensor, t: float) -> Tensor:
                t_tensor = torch.tensor([t], device=z.device, dtype=z.dtype).expand(z.shape[0])
                return self.forward(z, t_tensor, w)

            z_1 = ode_solve(velocity_fn, z_0, 0.0, 1.0, num_steps, method)

            # Blend: keep prefix, take generated continuation
            z_out = torch.cat([z_prefix, z_1[:, prefix_len:]], dim=1)
            all_continuations.append(z_out)

        return torch.cat(all_continuations, dim=0)
