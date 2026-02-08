"""
FAST-style action tokenizer using DCT + BPE.

Implements a non-learned (analytical) tokenizer inspired by Physical Intelligence's
FAST (Frequency-space Action Sequence Tokenization, arXiv:2501.09747).

Pipeline:
    invariants (B, T, D) --> DCT(axis=time) --> scale & round --> flatten --> BPE --> tokens
    tokens --> BPE decode --> unflatten --> inverse DCT --> reconstructed invariants

Key properties:
- No neural network training required (analytical DCT + learned BPE merges)
- Achieves ~10x compression on action sequences
- Works on DHB invariants or raw actions
- Pure numpy/scipy -- no PyTorch dependency

References:
- FAST (Physical Intelligence, 2025): arXiv:2501.09747
- DCT/IDCT: scipy.fft.dct/idct with ortho normalization
"""

from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from scipy.fft import dct, idct

    HAS_SCIPY_FFT = True
except ImportError:
    HAS_SCIPY_FFT = False


class FASTTokenizer:
    """
    FAST-style tokenizer: DCT frequency compression + BPE.

    Converts continuous invariant (or action) chunks into a compact sequence of
    discrete tokens using the Discrete Cosine Transform for energy compaction
    and Byte-Pair Encoding for further compression.

    Example:
        >>> tokenizer = FASTTokenizer(scale=10.0, vocab_size=1024, num_merges=200)
        >>> # Fit BPE on a corpus of invariant chunks
        >>> corpus = [np.random.randn(50, 8) for _ in range(100)]
        >>> tokenizer.fit(corpus)
        >>> # Tokenize a single chunk
        >>> tokens = tokenizer.encode(np.random.randn(50, 8))
        >>> recon = tokenizer.decode(tokens, time_horizon=50, dim=8)
    """

    def __init__(
        self,
        scale: float = 10.0,
        vocab_size: int = 1024,
        num_merges: int = 200,
    ):
        """
        Args:
            scale: Scaling factor for DCT coefficients before rounding.
                   Higher values preserve more detail but increase alphabet size.
            vocab_size: Maximum BPE vocabulary size (includes initial alphabet + merges).
            num_merges: Number of BPE merge operations to learn.
        """
        if not HAS_SCIPY_FFT:
            raise ImportError(
                "FASTTokenizer requires scipy. Install with: pip install scipy"
            )

        self.scale = scale
        self.vocab_size = vocab_size
        self.num_merges = num_merges

        # BPE state
        self.merges: Dict[Tuple[int, int], int] = {}
        self.reverse_merges: Dict[int, Tuple[int, int]] = {}
        self._fitted = False

        # Metadata for decoding
        self._min_token: int = 0
        self._time_horizon: Optional[int] = None
        self._dim: Optional[int] = None

    # ------------------------------------------------------------------
    # DCT helpers
    # ------------------------------------------------------------------

    def _dct_encode(self, chunk: np.ndarray) -> np.ndarray:
        """Apply DCT along time axis, scale, and round.

        Args:
            chunk: (T, D) continuous invariant/action chunk.

        Returns:
            (T, D) integer DCT coefficients.
        """
        coeffs = dct(chunk, axis=0, norm="ortho")
        return np.around(coeffs * self.scale).astype(int)

    def _dct_decode(self, int_coeffs: np.ndarray) -> np.ndarray:
        """Inverse: un-scale and apply IDCT.

        Args:
            int_coeffs: (T, D) integer DCT coefficients.

        Returns:
            (T, D) reconstructed continuous chunk.
        """
        coeffs = int_coeffs.astype(float) / self.scale
        return idct(coeffs, axis=0, norm="ortho")

    # ------------------------------------------------------------------
    # BPE helpers
    # ------------------------------------------------------------------

    def _flatten_to_tokens(self, int_coeffs: np.ndarray) -> List[int]:
        """Flatten integer DCT coefficients to a token sequence.

        Shifts values so that the minimum token is 0.

        Args:
            int_coeffs: (T, D) integer coefficients.

        Returns:
            List of non-negative integer tokens.
        """
        flat = int_coeffs.flatten()
        shifted = (flat - self._min_token).tolist()
        return [max(0, t) for t in shifted]

    def _unflatten_from_tokens(
        self,
        tokens: List[int],
        time_horizon: int,
        dim: int,
    ) -> np.ndarray:
        """Reshape flat token list back to (T, D) integer coefficients.

        Args:
            tokens: Flat token list (after BPE decode).
            time_horizon: Number of timesteps T.
            dim: Invariant/action dimension D.

        Returns:
            (T, D) integer coefficient array.
        """
        arr = np.array(tokens, dtype=int) + self._min_token
        expected = time_horizon * dim
        if len(arr) < expected:
            arr = np.pad(arr, (0, expected - len(arr)))
        elif len(arr) > expected:
            arr = arr[:expected]
        return arr.reshape(time_horizon, dim)

    def _bpe_encode(self, tokens: List[int]) -> List[int]:
        """Apply learned BPE merges to a flat token sequence."""
        for (a, b), merged in self.merges.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == a
                    and tokens[i + 1] == b
                ):
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def _bpe_decode(self, tokens: List[int]) -> List[int]:
        """Recursively expand BPE merges back to base tokens."""
        changed = True
        while changed:
            changed = False
            new_tokens = []
            for t in tokens:
                if t in self.reverse_merges:
                    a, b = self.reverse_merges[t]
                    new_tokens.extend([a, b])
                    changed = True
                else:
                    new_tokens.append(t)
            tokens = new_tokens
        return tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, chunks: List[np.ndarray]) -> "FASTTokenizer":
        """
        Learn BPE merges from a corpus of invariant/action chunks.

        Args:
            chunks: List of (T_i, D) arrays. Chunks may have different lengths
                    but must share the same dimension D.

        Returns:
            self (fitted tokenizer).
        """
        # Compute DCT + quantize for all chunks
        all_int_coeffs = [self._dct_encode(c) for c in chunks]

        # Determine global min/max for alphabet
        all_flat = np.concatenate([c.flatten() for c in all_int_coeffs])
        self._min_token = int(all_flat.min())
        max_token = int(all_flat.max())
        alphabet_size = max_token - self._min_token + 1

        if alphabet_size > self.vocab_size:
            import warnings

            warnings.warn(
                f"Initial alphabet size ({alphabet_size}) exceeds vocab_size "
                f"({self.vocab_size}). Consider increasing vocab_size or scale."
            )

        # Build flat token corpus
        corpus_tokens: List[int] = []
        for c in all_int_coeffs:
            corpus_tokens.extend(self._flatten_to_tokens(c))

        # Learn BPE merges
        current = list(corpus_tokens)
        next_id = alphabet_size  # Merge IDs start after initial alphabet

        for _ in range(self.num_merges):
            # Count pairs
            pair_counts: Dict[Tuple[int, int], int] = Counter()
            for i in range(len(current) - 1):
                pair_counts[(current[i], current[i + 1])] += 1

            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] < 2:
                break

            self.merges[best_pair] = next_id
            self.reverse_merges[next_id] = best_pair

            # Apply merge
            new_current = []
            i = 0
            while i < len(current):
                if (
                    i < len(current) - 1
                    and (current[i], current[i + 1]) == best_pair
                ):
                    new_current.append(next_id)
                    i += 2
                else:
                    new_current.append(current[i])
                    i += 1
            current = new_current
            next_id += 1

            if next_id >= self.vocab_size:
                break

        self._fitted = True

        # Cache typical dims from corpus
        if chunks:
            self._time_horizon = chunks[0].shape[0]
            self._dim = chunks[0].shape[1]

        return self

    def encode(self, chunk: np.ndarray) -> List[int]:
        """
        Tokenize a single invariant/action chunk.

        Args:
            chunk: (T, D) continuous chunk.

        Returns:
            List of BPE-compressed token IDs.
        """
        if not self._fitted:
            raise RuntimeError("FASTTokenizer must be fitted before encoding. Call .fit() first.")

        # Cache dimensions for decoding
        self._time_horizon = chunk.shape[0]
        self._dim = chunk.shape[1]

        int_coeffs = self._dct_encode(chunk)
        flat_tokens = self._flatten_to_tokens(int_coeffs)
        compressed = self._bpe_encode(flat_tokens)
        return compressed

    def encode_batch(self, chunks: np.ndarray) -> List[List[int]]:
        """
        Tokenize a batch of chunks.

        Args:
            chunks: (B, T, D) batch of chunks.

        Returns:
            List of B token sequences.
        """
        if chunks.ndim == 2:
            return [self.encode(chunks)]
        return [self.encode(chunks[i]) for i in range(chunks.shape[0])]

    def decode(
        self,
        tokens: List[int],
        time_horizon: Optional[int] = None,
        dim: Optional[int] = None,
    ) -> np.ndarray:
        """
        Decode tokens back to a continuous chunk.

        Args:
            tokens: BPE-compressed token sequence.
            time_horizon: Number of timesteps T (uses cached value if None).
            dim: Invariant/action dimension D (uses cached value if None).

        Returns:
            (T, D) reconstructed chunk.
        """
        T = time_horizon or self._time_horizon
        D = dim or self._dim
        if T is None or D is None:
            raise ValueError(
                "time_horizon and dim must be provided (or cached from encode)."
            )

        flat_tokens = self._bpe_decode(list(tokens))
        int_coeffs = self._unflatten_from_tokens(flat_tokens, T, D)
        return self._dct_decode(int_coeffs)

    def decode_batch(
        self,
        token_sequences: List[List[int]],
        time_horizon: Optional[int] = None,
        dim: Optional[int] = None,
    ) -> np.ndarray:
        """
        Decode a batch of token sequences.

        Args:
            token_sequences: List of B token sequences.
            time_horizon: Number of timesteps T.
            dim: Invariant/action dimension D.

        Returns:
            (B, T, D) batch of reconstructed chunks.
        """
        decoded = [self.decode(ts, time_horizon, dim) for ts in token_sequences]
        return np.stack(decoded)

    def compression_ratio(self, chunk: np.ndarray) -> float:
        """Compute compression ratio for a chunk.

        Returns:
            original_values / compressed_tokens.
        """
        tokens = self.encode(chunk)
        original = chunk.size  # T * D float values
        return original / len(tokens) if tokens else 1.0

    def reconstruction_error(self, chunk: np.ndarray) -> float:
        """Compute MSE reconstruction error for a chunk."""
        tokens = self.encode(chunk)
        recon = self.decode(tokens, chunk.shape[0], chunk.shape[1])
        return float(np.mean((chunk - recon) ** 2))

    def get_stats(self, chunk: Optional[np.ndarray] = None) -> Dict:
        """Get tokenizer statistics.

        Args:
            chunk: Optional chunk to compute per-chunk statistics.

        Returns:
            Dictionary of statistics.
        """
        stats = {
            "scale": self.scale,
            "vocab_size": self.vocab_size,
            "num_merges_learned": len(self.merges),
            "min_token": self._min_token,
            "fitted": self._fitted,
        }
        if chunk is not None:
            stats["compression_ratio"] = self.compression_ratio(chunk)
            stats["mse"] = self.reconstruction_error(chunk)
        return stats
