"""Minimal VLA pipeline: dataset → DHB encode → tokenization → export."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

from dhb_xr.core.types import DHBMethod, EncodingMethod
from dhb_xr.encoder.dhb_dr import encode_dhb_dr
from dhb_xr.tokenization.vqvae import DHBTokenizer

try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_TORCH = False


@dataclass
class DHBVLAPipelineConfig:
    dhb_method: DHBMethod = DHBMethod.DOUBLE_REFLECTION
    method: EncodingMethod = EncodingMethod.POSITION
    use_default_initial_frames: bool = True
    codebook_size: int = 256
    latent_dim: int = 32


class DHBVLAPipeline:
    """Minimal end-to-end pipeline that emits DHB tokens."""

    def __init__(self, config: Optional[DHBVLAPipelineConfig] = None):
        self.config = config or DHBVLAPipelineConfig()
        self._tokenizer: Optional[DHBTokenizer] = None

    def _get_tokenizer(self, invariant_dim: int) -> DHBTokenizer:
        if not HAS_TORCH:
            raise ImportError("torch is required for tokenization (pip install dhb_xr[gpu]).")
        if self._tokenizer is None:
            self._tokenizer = DHBTokenizer(
                invariant_dim=invariant_dim,
                latent_dim=self.config.latent_dim,
                codebook_size=self.config.codebook_size,
            )
        return self._tokenizer

    def encode_trajectory(self, positions: np.ndarray, quaternions: np.ndarray) -> Dict:
        return encode_dhb_dr(
            positions,
            quaternions,
            method=self.config.method,
            use_default_initial_frames=self.config.use_default_initial_frames,
            dhb_method=self.config.dhb_method,
        )

    def tokenize_invariants(self, invariants: np.ndarray) -> np.ndarray:
        tokenizer = self._get_tokenizer(invariants.shape[1])
        inv_batch = torch.from_numpy(invariants.astype(np.float32)).unsqueeze(0)
        indices, _, _, _ = tokenizer(inv_batch)
        return indices.squeeze(0).cpu().numpy()

    def process_dataset(self, episodes: Iterable[Dict]) -> List[Dict]:
        outputs: List[Dict] = []
        for ep in episodes:
            positions = ep["positions"]
            quaternions = ep["quaternions"]
            meta = ep.get("metadata", {})

            enc = self.encode_trajectory(positions, quaternions)
            invariants = np.concatenate(
                [enc["linear_motion_invariants"], enc["angular_motion_invariants"]],
                axis=1,
            )
            tokens = self.tokenize_invariants(invariants)

            outputs.append(
                {
                    "tokens": tokens,
                    "invariants": invariants,
                    "initial_pose": enc["initial_pose"],
                    "metadata": meta,
                }
            )
        return outputs
