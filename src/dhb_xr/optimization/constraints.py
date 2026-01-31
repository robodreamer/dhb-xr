"""Constraint helpers for trajectory optimization (obstacles, bounds)."""

import numpy as np
from typing import Callable, Optional


def sphere_obstacle_constraint(
    center: np.ndarray,
    radius: float,
) -> Callable[[np.ndarray], float]:
    """Returns a constraint function c(positions) -> min distance squared - radius^2 (>= 0 feasible)."""
    center = np.asarray(center).reshape(3)

    def c(positions: np.ndarray) -> np.ndarray:
        pos = np.asarray(positions).reshape(-1, 3)
        d2 = np.sum((pos - center) ** 2, axis=1)
        return d2 - radius**2
    return c


def box_bounds_constraint(
    lower: np.ndarray,
    upper: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """Returns constraint c(positions) such that c >= 0 when inside box."""
    lower = np.asarray(lower).reshape(3)
    upper = np.asarray(upper).reshape(3)

    def c(positions: np.ndarray) -> np.ndarray:
        pos = np.asarray(positions).reshape(-1, 3)
        return np.concatenate([pos - lower, upper - pos], axis=1).ravel()
    return c
