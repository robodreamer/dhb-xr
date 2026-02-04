"""RoboCASA dataset adapter (HDF5/robomimic-style)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple

import numpy as np

try:
    import h5py
    HAS_H5PY = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_H5PY = False


DEFAULT_POS_KEYS = (
    "robot0_eef_pos",
    "eef_pos",
    "ee_pos",
)
DEFAULT_QUAT_KEYS = (
    "robot0_eef_quat",
    "eef_quat",
    "ee_quat",
)


@dataclass
class RoboCASAAdapter:
    """
    Minimal RoboCASA adapter that yields (positions, quaternions, metadata).

    Assumes a robomimic-style HDF5 with episodes under /data/<demo_id>/obs.
    """

    pos_keys: Tuple[str, ...] = DEFAULT_POS_KEYS
    quat_keys: Tuple[str, ...] = DEFAULT_QUAT_KEYS
    obs_group: str = "obs"

    def _find_key(self, obs_group: "h5py.Group", candidates: Tuple[str, ...]) -> Optional[str]:
        for key in candidates:
            if key in obs_group:
                return key
        return None

    def load_dataset(self, dataset_path: str) -> Iterator[Dict]:
        if not HAS_H5PY:
            raise ImportError("h5py is required for RoboCASA adapter (pip install h5py).")

        with h5py.File(dataset_path, "r") as h5:
            data_group = h5.get("data")
            if data_group is None:
                raise ValueError("RoboCASA HDF5 missing /data group.")

            for demo_id in data_group.keys():
                demo = data_group[demo_id]
                obs = demo.get(self.obs_group)
                if obs is None:
                    continue

                pos_key = self._find_key(obs, self.pos_keys)
                quat_key = self._find_key(obs, self.quat_keys)
                if pos_key is None or quat_key is None:
                    continue

                positions = np.asarray(obs[pos_key], dtype=np.float64)
                quaternions = np.asarray(obs[quat_key], dtype=np.float64)

                metadata = {
                    "demo_id": demo_id,
                    "pos_key": pos_key,
                    "quat_key": quat_key,
                    "source": "robocasa",
                }
                if "task" in demo.attrs:
                    metadata["task"] = demo.attrs["task"]
                if "language" in demo.attrs:
                    metadata["language_instruction"] = demo.attrs["language"]

                yield {
                    "positions": positions,
                    "quaternions": quaternions,
                    "metadata": metadata,
                }
