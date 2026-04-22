"""Minimal VLA pipeline: dataset → DHB encode → tokenization → export.

This module now exposes a small streaming-oriented token contract for VLA use:

- encode a waypoint chunk into DHB invariants
- tokenize the invariants with either VQ or OAT
- detokenize either the full token sequence or a streamed prefix

The streaming-prefix path is OAT-specific because OAT supports anytime prefix
decoding. That is the token path that matches chunked VLA execution best.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence

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
    tokenizer_kind: Literal["vq", "oat"] = "vq"
    codebook_size: int = 256
    latent_dim: int = 32
    oat_num_registers: int = 8
    oat_fsq_levels: Optional[List[int]] = None
    oat_emb_dim: int = 64
    oat_num_heads: int = 4
    oat_encoder_depth: int = 2
    oat_decoder_depth: int = 2


class DHBVLAPipeline:
    """Minimal end-to-end pipeline that emits DHB tokens."""

    def __init__(self, config: Optional[DHBVLAPipelineConfig] = None):
        self.config = config or DHBVLAPipelineConfig()
        self._tokenizer: Optional[object] = None

    def _get_tokenizer(self, invariant_dim: int, horizon: Optional[int] = None) -> object:
        if not HAS_TORCH:
            raise ImportError("torch is required for tokenization (pip install dhb_xr[gpu]).")
        if self._tokenizer is None:
            if self.config.tokenizer_kind == "vq":
                self._tokenizer = DHBTokenizer(
                    invariant_dim=invariant_dim,
                    latent_dim=self.config.latent_dim,
                    codebook_size=self.config.codebook_size,
                )
            elif self.config.tokenizer_kind == "oat":
                if horizon is None:
                    raise ValueError("OAT tokenization requires the invariant horizon.")
                from dhb_xr.tokenization.oat_tokenizer import OATTokenizer

                self._tokenizer = OATTokenizer(
                    input_dim=invariant_dim,
                    input_horizon=horizon,
                    emb_dim=self.config.oat_emb_dim,
                    latent_dim=4,
                    num_registers=self.config.oat_num_registers,
                    fsq_levels=self.config.oat_fsq_levels or [8, 5, 5, 5],
                    encoder_depth=self.config.oat_encoder_depth,
                    decoder_depth=self.config.oat_decoder_depth,
                    num_heads=self.config.oat_num_heads,
                )
            else:
                raise ValueError(f"Unsupported tokenizer_kind={self.config.tokenizer_kind!r}")
        return self._tokenizer

    def encode_trajectory(self, positions: np.ndarray, quaternions: np.ndarray) -> Dict:
        return encode_dhb_dr(
            positions,
            quaternions,
            method=self.config.method,
            use_default_initial_frames=self.config.use_default_initial_frames,
            dhb_method=self.config.dhb_method,
        )

    @staticmethod
    def _run_tokenizer_eval(tokenizer: object, fn):
        was_training = bool(getattr(tokenizer, "training", False))
        if hasattr(tokenizer, "eval"):
            tokenizer.eval()
        try:
            return fn()
        finally:
            if was_training and hasattr(tokenizer, "train"):
                tokenizer.train()

    def tokenize_invariants(self, invariants: np.ndarray) -> np.ndarray:
        tokenizer = self._get_tokenizer(invariants.shape[1], horizon=invariants.shape[0])
        inv_batch = torch.from_numpy(invariants.astype(np.float32)).unsqueeze(0)
        if self.config.tokenizer_kind == "vq":
            indices, _, _, _ = self._run_tokenizer_eval(tokenizer, lambda: tokenizer(inv_batch))
            return indices.squeeze(0).cpu().numpy()
        tokens = self._run_tokenizer_eval(tokenizer, lambda: tokenizer.tokenize(inv_batch))
        return tokens.squeeze(0).cpu().numpy()

    def detokenize_tokens(
        self,
        tokens: np.ndarray,
        *,
        invariant_dim: int,
        horizon: int,
        prefix_keep_k: Optional[int | Sequence[int]] = None,
    ) -> np.ndarray:
        """Decode DHB action tokens back to invariants.

        For OAT this supports prefix decoding, which is the exact streaming use
        case for chunked VLA execution. For VQ, only full-token decode is
        supported because there is no explicit anytime-prefix contract.
        """

        tokenizer = self._get_tokenizer(invariant_dim, horizon=horizon)
        token_arr = np.asarray(tokens)
        if token_arr.ndim == 1:
            token_arr = token_arr[None, :]
        token_batch = torch.from_numpy(token_arr.astype(np.int64))

        if self.config.tokenizer_kind == "vq":
            if prefix_keep_k is not None:
                raise ValueError("prefix_keep_k is only supported for tokenizer_kind='oat'")
            with torch.no_grad():
                recon = self._run_tokenizer_eval(tokenizer, lambda: tokenizer.decode_tokens(token_batch))
            return recon.detach().cpu().numpy()

        keep_k: Optional[List[int]]
        if prefix_keep_k is None:
            keep_k = None
        elif isinstance(prefix_keep_k, int):
            keep_k = [int(prefix_keep_k)] * token_batch.shape[0]
        else:
            keep_k = [int(v) for v in prefix_keep_k]
            if len(keep_k) != token_batch.shape[0]:
                raise ValueError(
                    f"Expected {token_batch.shape[0]} prefix lengths, got {len(keep_k)}"
                )

        with torch.no_grad():
            recon = self._run_tokenizer_eval(
                tokenizer,
                lambda: tokenizer.detokenize(token_batch, eval_keep_k=keep_k),
            )
        return recon.detach().cpu().numpy()

    def decode_token_stream(
        self,
        tokens: np.ndarray,
        *,
        invariant_dim: int,
        horizon: int,
        prefix_lengths: Sequence[int],
    ) -> List[Dict[str, np.ndarray | int]]:
        """Decode successive streamed token prefixes into full invariant chunks.

        This matches the VLA use case where a tokenized chunk arrives
        incrementally during policy execution and each prefix must decode to a
        valid coarse chunk.
        """
        if self.config.tokenizer_kind != "oat":
            raise ValueError("decode_token_stream requires tokenizer_kind='oat'")

        token_arr = np.asarray(tokens)
        if token_arr.ndim == 1:
            token_arr = token_arr[None, :]
        full_len = token_arr.shape[1]
        outputs: List[Dict[str, np.ndarray | int]] = []
        for keep_k in prefix_lengths:
            k = int(keep_k)
            if k <= 0 or k > full_len:
                raise ValueError(f"prefix length {k} is out of range for token length {full_len}")
            recon = self.detokenize_tokens(
                token_arr[:, :k],
                invariant_dim=invariant_dim,
                horizon=horizon,
                prefix_keep_k=k,
            )
            outputs.append({"keep_k": k, "reconstruction": recon})
        return outputs

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
                    "tokenizer_kind": self.config.tokenizer_kind,
                    "metadata": meta,
                }
            )
        return outputs
