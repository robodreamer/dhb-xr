"""
Generative models for DHB-Token trajectory generation.

This module provides variational flow matching (VFM) and rectified flow matching (V-RFM)
for multi-modal trajectory generation in the DHB invariant latent space.

Key components:
- FlowMatcher: Base flow matching model for continuous latent generation
- VariationalFlowMatcher: VFM with latent conditioning for multi-modal generation
- VFMTokenGenerator: End-to-end integration with DHB tokenizers

Example usage:
    >>> from dhb_xr.generative import VariationalFlowMatcher, VFMTokenGenerator
    >>> from dhb_xr.tokenization import DHBTokenizer
    >>>
    >>> # Create tokenizer and flow matcher
    >>> tokenizer = DHBTokenizer(invariant_dim=8, latent_dim=32, codebook_size=512)
    >>> flow_matcher = VariationalFlowMatcher(latent_dim=32, hidden_dim=128)
    >>>
    >>> # Create end-to-end generator
    >>> generator = VFMTokenGenerator(tokenizer, flow_matcher)
    >>>
    >>> # Generate multi-modal trajectories
    >>> invariants = generator.generate_multimodal(prefix_invariants, num_modes=4)
"""

from .flow_matching import (
    SinusoidalTimeEmbedding,
    VelocityNetwork,
    FlowMatcher,
    VariationalFlowMatcher,
)
from .sampling import euler_solve, rk4_solve, ode_solve
from .latent_encoder import (
    LatentEncoder,
    CategoricalLatentEncoder,
    HybridLatentEncoder,
)
from .vfm_tokenizer import (
    VFMTokenGenerator,
    ConditionalVFMGenerator,
)
from .training import (
    InvariantDataset,
    train_vfm_tokenizer,
    evaluate_model,
    compute_reconstruction_error,
    compute_generation_diversity,
    linear_kl_schedule,
    cyclical_kl_schedule,
)

__all__ = [
    # Flow matching
    "SinusoidalTimeEmbedding",
    "VelocityNetwork",
    "FlowMatcher",
    "VariationalFlowMatcher",
    # Latent encoders
    "LatentEncoder",
    "CategoricalLatentEncoder",
    "HybridLatentEncoder",
    # VFM Token generators
    "VFMTokenGenerator",
    "ConditionalVFMGenerator",
    # Training utilities
    "InvariantDataset",
    "train_vfm_tokenizer",
    "evaluate_model",
    "compute_reconstruction_error",
    "compute_generation_diversity",
    "linear_kl_schedule",
    "cyclical_kl_schedule",
    # ODE solvers
    "euler_solve",
    "rk4_solve",
    "ode_solve",
]
