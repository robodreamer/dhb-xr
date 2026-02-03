"""
Latent encoders for variational flow matching.

Provides inference networks q(w | z_t, t) for the variational formulation:
- LatentEncoder: Gaussian posterior (continuous w)
- CategoricalLatentEncoder: Categorical posterior with Gumbel-Softmax (discrete w)

The latent w captures "intent" or "mode" information that helps the velocity
network distinguish between multiple valid trajectory continuations.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Dict

from .flow_matching import SinusoidalTimeEmbedding


class LatentEncoder(nn.Module):
    """
    Gaussian latent encoder for variational flow matching.

    Maps (z_t, t) to a Gaussian distribution over the conditioning latent w.
    Uses an MLP with optional temporal attention for sequence inputs.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        time_embed_dim: int = 64,
        use_attention: bool = False,
        num_heads: int = 4,
    ):
        """
        Args:
            input_dim: Dimension of input z_t.
            latent_dim: Dimension of output latent w.
            hidden_dim: Hidden layer dimension.
            num_layers: Number of hidden layers.
            time_embed_dim: Dimension of time embedding.
            use_attention: Whether to use attention for sequence pooling.
            num_heads: Number of attention heads (if use_attention=True).
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.use_attention = use_attention

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        # Optional attention for sequence pooling
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=num_heads,
                batch_first=True,
            )
            self.pool_query = nn.Parameter(torch.randn(1, 1, input_dim))

        # MLP encoder
        layers = []
        in_dim = input_dim + time_embed_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else hidden_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
            ])
            in_dim = out_dim

        self.encoder = nn.Sequential(*layers)

        # Output heads for mean and log_std
        self.mean_head = nn.Linear(hidden_dim, latent_dim)
        self.log_std_head = nn.Linear(hidden_dim, latent_dim)

    def pool_sequence(self, z_t: Tensor) -> Tensor:
        """Pool sequence dimension to get batch-level representation."""
        if z_t.dim() == 2:
            return z_t  # Already (B, D)

        if self.use_attention:
            # Use attention pooling
            B, T, D = z_t.shape
            query = self.pool_query.expand(B, -1, -1)
            pooled, _ = self.attention(query, z_t, z_t)
            return pooled.squeeze(1)  # (B, D)
        else:
            # Simple mean pooling
            return z_t.mean(dim=1)

    def forward(
        self,
        z_t: Tensor,
        t: Tensor | float,
    ) -> Tuple[Tensor, Tensor]:
        """
        Encode to posterior parameters.

        Args:
            z_t: Input state (B, T, D) or (B, D).
            t: Time (B,) or scalar in [0, 1].

        Returns:
            Tuple of (mean, log_std) for q(w | z_t, t).
        """
        # Handle scalar time
        if isinstance(t, float):
            t = torch.tensor([t], device=z_t.device, dtype=z_t.dtype)
        if t.dim() == 0:
            t = t.unsqueeze(0)

        # Pool sequence
        z_pooled = self.pool_sequence(z_t)

        # Time embedding
        t_embed = self.time_embed(t)

        # Expand t_embed to batch size if needed
        if t_embed.shape[0] == 1 and z_pooled.shape[0] > 1:
            t_embed = t_embed.expand(z_pooled.shape[0], -1)

        # Concatenate and encode
        x = torch.cat([z_pooled, t_embed], dim=-1)
        h = self.encoder(x)

        # Output parameters
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, min=-10, max=2)

        return mean, log_std

    def sample(
        self,
        z_t: Tensor,
        t: Tensor | float,
        num_samples: int = 1,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Sample from posterior using reparameterization trick.

        Args:
            z_t: Input state.
            t: Time.
            num_samples: Number of samples per input.

        Returns:
            Tuple of (w, mean, log_std) where w is (B, num_samples, latent_dim)
            or (B, latent_dim) if num_samples=1.
        """
        mean, log_std = self.forward(z_t, t)
        std = torch.exp(log_std)

        if num_samples == 1:
            eps = torch.randn_like(mean)
            w = mean + std * eps
        else:
            # (B, num_samples, latent_dim)
            eps = torch.randn(mean.shape[0], num_samples, mean.shape[1], device=mean.device)
            w = mean.unsqueeze(1) + std.unsqueeze(1) * eps

        return w, mean, log_std

    def kl_divergence(
        self,
        mean: Tensor,
        log_std: Tensor,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
    ) -> Tensor:
        """
        Compute KL divergence from posterior to prior.

        KL(N(mean, std^2) || N(prior_mean, prior_std^2))
        """
        std = torch.exp(log_std)
        prior_var = prior_std ** 2

        kl = 0.5 * (
            (std ** 2 + (mean - prior_mean) ** 2) / prior_var
            - 1
            - 2 * log_std
            + 2 * math.log(prior_std)
        )

        return kl.sum(dim=-1).mean()


class CategoricalLatentEncoder(nn.Module):
    """
    Categorical latent encoder with Gumbel-Softmax for discrete modes.

    Useful when the multi-modality has a clear categorical structure
    (e.g., left vs. right grasp, different skill types).

    Uses Gumbel-Softmax for differentiable sampling during training
    and argmax during inference.
    """

    def __init__(
        self,
        input_dim: int,
        num_categories: int,
        embedding_dim: int = 16,
        hidden_dim: int = 256,
        num_layers: int = 2,
        time_embed_dim: int = 64,
        temperature: float = 1.0,
    ):
        """
        Args:
            input_dim: Dimension of input z_t.
            num_categories: Number of discrete categories (modes).
            embedding_dim: Dimension of category embeddings.
            hidden_dim: Hidden layer dimension.
            num_layers: Number of hidden layers.
            time_embed_dim: Dimension of time embedding.
            temperature: Gumbel-Softmax temperature.
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_categories = num_categories
        self.embedding_dim = embedding_dim
        self.temperature = temperature

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        # MLP encoder
        layers = []
        in_dim = input_dim + time_embed_dim
        for i in range(num_layers):
            out_dim = hidden_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
            ])
            in_dim = out_dim

        self.encoder = nn.Sequential(*layers)

        # Logits head
        self.logits_head = nn.Linear(hidden_dim, num_categories)

        # Category embeddings (for converting one-hot to continuous)
        self.category_embeddings = nn.Embedding(num_categories, embedding_dim)

    @property
    def latent_dim(self) -> int:
        """Return the effective latent dimension (embedding_dim)."""
        return self.embedding_dim

    def forward(
        self,
        z_t: Tensor,
        t: Tensor | float,
    ) -> Tensor:
        """
        Compute category logits.

        Args:
            z_t: Input state (B, T, D) or (B, D).
            t: Time (B,) or scalar.

        Returns:
            Logits (B, num_categories).
        """
        # Handle scalar time
        if isinstance(t, float):
            t = torch.tensor([t], device=z_t.device, dtype=z_t.dtype)
        if t.dim() == 0:
            t = t.unsqueeze(0)

        # Pool sequence
        if z_t.dim() == 3:
            z_pooled = z_t.mean(dim=1)
        else:
            z_pooled = z_t

        # Time embedding
        t_embed = self.time_embed(t)
        if t_embed.shape[0] == 1 and z_pooled.shape[0] > 1:
            t_embed = t_embed.expand(z_pooled.shape[0], -1)

        # Encode
        x = torch.cat([z_pooled, t_embed], dim=-1)
        h = self.encoder(x)

        # Logits
        logits = self.logits_head(h)

        return logits

    def sample(
        self,
        z_t: Tensor,
        t: Tensor | float,
        hard: bool = False,
        temperature: Optional[float] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Sample using Gumbel-Softmax.

        Args:
            z_t: Input state.
            t: Time.
            hard: If True, use straight-through estimator for hard samples.
            temperature: Override default temperature.

        Returns:
            Tuple of (w, probs, logits) where:
            - w: Category embedding (B, embedding_dim)
            - probs: Category probabilities (B, num_categories)
            - logits: Raw logits (B, num_categories)
        """
        if temperature is None:
            temperature = self.temperature

        logits = self.forward(z_t, t)
        probs = F.softmax(logits, dim=-1)

        # Gumbel-Softmax sampling
        if self.training:
            gumbel_probs = F.gumbel_softmax(logits, tau=temperature, hard=hard)
        else:
            # During inference, use argmax
            if hard:
                indices = logits.argmax(dim=-1)
                gumbel_probs = F.one_hot(indices, self.num_categories).float()
            else:
                gumbel_probs = probs

        # Convert to embedding
        # gumbel_probs: (B, num_categories)
        # category_embeddings: (num_categories, embedding_dim)
        w = gumbel_probs @ self.category_embeddings.weight

        return w, probs, logits

    def sample_prior(
        self,
        batch_size: int,
        device: str = "cpu",
        hard: bool = True,
    ) -> Tensor:
        """
        Sample from uniform categorical prior.

        Args:
            batch_size: Number of samples.
            device: Device for tensor.
            hard: If True, return embeddings for hard categories.

        Returns:
            Category embeddings (batch_size, embedding_dim).
        """
        if hard:
            # Sample uniform categories
            indices = torch.randint(
                0, self.num_categories, (batch_size,), device=device
            )
            w = self.category_embeddings(indices)
        else:
            # Soft uniform mixture
            probs = torch.ones(batch_size, self.num_categories, device=device)
            probs = probs / self.num_categories
            w = probs @ self.category_embeddings.weight

        return w

    def kl_divergence(
        self,
        probs: Tensor,
        prior_probs: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute KL divergence from posterior to prior.

        If prior_probs is None, uses uniform prior.
        """
        if prior_probs is None:
            # Uniform prior
            prior_probs = torch.ones_like(probs) / self.num_categories

        # KL = sum_k p_k * log(p_k / q_k)
        kl = probs * (torch.log(probs + 1e-10) - torch.log(prior_probs + 1e-10))

        return kl.sum(dim=-1).mean()

    def entropy(self, probs: Tensor) -> Tensor:
        """Compute entropy of the categorical distribution."""
        return -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()


class HybridLatentEncoder(nn.Module):
    """
    Hybrid encoder combining categorical and continuous latents.

    Useful for representing both discrete mode selection and
    continuous variations within each mode.
    """

    def __init__(
        self,
        input_dim: int,
        num_categories: int,
        continuous_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        time_embed_dim: int = 64,
        temperature: float = 1.0,
    ):
        """
        Args:
            input_dim: Dimension of input z_t.
            num_categories: Number of discrete modes.
            continuous_dim: Dimension of continuous latent per mode.
            hidden_dim: Hidden layer dimension.
            num_layers: Number of hidden layers.
            time_embed_dim: Dimension of time embedding.
            temperature: Gumbel-Softmax temperature.
        """
        super().__init__()
        self.num_categories = num_categories
        self.continuous_dim = continuous_dim

        # Categorical encoder
        self.categorical = CategoricalLatentEncoder(
            input_dim=input_dim,
            num_categories=num_categories,
            embedding_dim=hidden_dim // 4,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            time_embed_dim=time_embed_dim,
            temperature=temperature,
        )

        # Continuous encoder (shared across modes)
        self.continuous = LatentEncoder(
            input_dim=input_dim,
            latent_dim=continuous_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            time_embed_dim=time_embed_dim,
        )

        # Combine categorical embedding and continuous latent
        self.combine = nn.Linear(
            self.categorical.embedding_dim + continuous_dim,
            continuous_dim,
        )

    @property
    def latent_dim(self) -> int:
        """Return the combined latent dimension."""
        return self.continuous_dim

    def forward(
        self,
        z_t: Tensor,
        t: Tensor | float,
    ) -> Dict[str, Tensor]:
        """
        Encode to both categorical and continuous latents.

        Returns dictionary with all components.
        """
        # Categorical
        cat_logits = self.categorical(z_t, t)
        cat_probs = F.softmax(cat_logits, dim=-1)

        # Continuous
        cont_mean, cont_log_std = self.continuous(z_t, t)

        return {
            "categorical_logits": cat_logits,
            "categorical_probs": cat_probs,
            "continuous_mean": cont_mean,
            "continuous_log_std": cont_log_std,
        }

    def sample(
        self,
        z_t: Tensor,
        t: Tensor | float,
        hard: bool = False,
        temperature: Optional[float] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Sample combined latent.

        Returns:
            Tuple of (w, info_dict) where w is (B, latent_dim).
        """
        # Sample categorical
        cat_w, cat_probs, cat_logits = self.categorical.sample(
            z_t, t, hard=hard, temperature=temperature
        )

        # Sample continuous
        cont_w, cont_mean, cont_log_std = self.continuous.sample(z_t, t)

        # Combine
        combined = torch.cat([cat_w, cont_w], dim=-1)
        w = self.combine(combined)

        info = {
            "categorical_w": cat_w,
            "categorical_probs": cat_probs,
            "categorical_logits": cat_logits,
            "continuous_w": cont_w,
            "continuous_mean": cont_mean,
            "continuous_log_std": cont_log_std,
        }

        return w, info

    def kl_divergence(self, info: Dict[str, Tensor]) -> Tensor:
        """Compute total KL divergence."""
        cat_kl = self.categorical.kl_divergence(info["categorical_probs"])
        cont_kl = self.continuous.kl_divergence(
            info["continuous_mean"], info["continuous_log_std"]
        )
        return cat_kl + cont_kl
