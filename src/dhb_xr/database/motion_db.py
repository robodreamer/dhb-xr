"""Motion database: store trajectories and invariants, similarity search."""

import json
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

        # Use FAISS index when available and embeddings match
        if self.use_faiss and self.index is not None and self.embedding_dim is not None:
            flat = U_q.flatten().astype(np.float32)
            if flat.shape[0] == self.embedding_dim:
                q_emb = flat.reshape(1, -1)
                k_actual = min(k, len(self.invariants_list))
                distances, indices = self.index.search(q_emb, k_actual)
                return [
                    (self.invariants_list[indices[0, j]], self.metadata_list[indices[0, j]], float(distances[0, j]))
                    for j in range(k_actual)
                    if indices[0, j] >= 0
                ]

        # Brute-force fallback
        dists = [invariant_distance(U_q, U) for U in self.invariants_list]
        idx = np.argsort(dists)[:k]
        return [(self.invariants_list[i], self.metadata_list[i], dists[i]) for i in idx]

    def save(self, path: str) -> None:
        """Persist the database to disk.  Writes ``<path>.npz`` (invariants) and ``<path>.json`` (metadata)."""
        arrs = {f"inv_{i}": inv for i, inv in enumerate(self.invariants_list)}
        np.savez(path + ".npz", **arrs)
        meta = {
            "dhb_method": "double_reflection" if self.dhb_method == DHBMethod.DOUBLE_REFLECTION else "original",
            "metadata_list": self.metadata_list,
        }
        with open(path + ".json", "w") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, path: str) -> "MotionDatabase":
        """Load a database from ``<path>.npz`` and ``<path>.json``."""
        with open(path + ".json", "r") as f:
            meta = json.load(f)
        db = cls(dhb_method=meta["dhb_method"], use_faiss=False)
        db.metadata_list = meta["metadata_list"]
        data = np.load(path + ".npz")
        n = len(db.metadata_list)
        db.invariants_list = [data[f"inv_{i}"].astype(np.float32) for i in range(n)]
        return db
