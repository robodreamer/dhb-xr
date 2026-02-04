"""Motion database: store trajectories and invariants, similarity search."""

import numpy as np
from typing import List, Dict, Any, Optional

from dhb_xr.encoder.dhb_dr import encode_dhb_dr
from dhb_xr.core.types import DHBMethod, EncodingMethod
from dhb_xr.database.similarity import invariant_distance, soft_dtw_distance

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class MotionDatabase:
    """
    Store trajectories as invariants; optional FAISS index for fast retrieval.
    add(trajectory, metadata) -> encode and store
    retrieve(query_trajectory, k) -> top-k similar (by L2 on invariants or DTW).
    """

    def __init__(self, dhb_method: str = "double_reflection", use_faiss: bool = False):
        self.dhb_method = DHBMethod.DOUBLE_REFLECTION if dhb_method == "double_reflection" else DHBMethod.ORIGINAL
        self.k_inv = 4 if self.dhb_method == DHBMethod.DOUBLE_REFLECTION else 3
        self.invariants_list: List[np.ndarray] = []
        self.metadata_list: List[Dict[str, Any]] = []
        self.use_faiss = use_faiss and HAS_FAISS
        self.index: Optional[Any] = None
        self.embedding_dim: Optional[int] = None

    def add(
        self,
        positions: np.ndarray,
        quaternions: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        out = encode_dhb_dr(
            positions, quaternions,
            method=EncodingMethod.POSITION, use_default_initial_frames=True, dhb_method=self.dhb_method,
        )
        U = np.concatenate([
            out["linear_motion_invariants"],
            out["angular_motion_invariants"],
        ], axis=1)
        self.invariants_list.append(U.astype(np.float32))
        self.metadata_list.append(metadata or {})
        if self.use_faiss and U.size > 0:
            flat = U.flatten()
            if self.embedding_dim is None:
                self.embedding_dim = flat.shape[0]
                self.index = faiss.IndexFlatL2(self.embedding_dim)
            emb = flat.reshape(1, -1).astype(np.float32)
            if emb.shape[1] == self.embedding_dim:
                self.index.add(emb)

    def retrieve(
        self,
        query_positions: np.ndarray,
        query_quaternions: np.ndarray,
        k: int = 10,
        use_dtw: bool = False,
    ) -> List[tuple]:
        out = encode_dhb_dr(
            query_positions, query_quaternions,
            method=EncodingMethod.POSITION, use_default_initial_frames=True, dhb_method=self.dhb_method,
        )
        U_q = np.concatenate([
            out["linear_motion_invariants"],
            out["angular_motion_invariants"],
        ], axis=1)
        if use_dtw:
            dists = [soft_dtw_distance(U_q, U) for U in self.invariants_list]
            idx = np.argsort(dists)[:k]
            return [(self.invariants_list[i], self.metadata_list[i], dists[i]) for i in idx]
        dists = [invariant_distance(U_q, U) for U in self.invariants_list]
        idx = np.argsort(dists)[:k]
        return [(self.invariants_list[i], self.metadata_list[i], dists[i]) for i in idx]
