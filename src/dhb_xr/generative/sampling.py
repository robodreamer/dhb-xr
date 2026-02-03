"""
ODE solvers for flow matching sampling.

Provides Euler and RK4 integrators for solving the flow ODE:
    dz/dt = v_theta(z_t, t)

Starting from z_0 (noise) and integrating to z_1 (data).
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Callable, Optional, Tuple, List


def euler_solve(
    velocity_fn: Callable[[Tensor, float], Tensor],
    z_init: Tensor,
    t_start: float = 0.0,
    t_end: float = 1.0,
    num_steps: int = 10,
    return_trajectory: bool = False,
) -> Tensor | Tuple[Tensor, List[Tensor]]:
    """
    Euler method for ODE integration.

    Solves dz/dt = v(z, t) from t_start to t_end.

    Args:
        velocity_fn: Function (z, t) -> velocity. Takes tensor z and scalar t.
        z_init: Initial state (B, T, D) or (B, D).
        t_start: Starting time (typically 0 for noise).
        t_end: Ending time (typically 1 for data).
        num_steps: Number of Euler steps.
        return_trajectory: If True, return list of intermediate states.

    Returns:
        Final state z_end, optionally with trajectory list.
    """
    dt = (t_end - t_start) / num_steps
    z = z_init.clone()
    trajectory = [z.clone()] if return_trajectory else None

    for i in range(num_steps):
        t = t_start + i * dt
        v = velocity_fn(z, t)
        z = z + dt * v
        if return_trajectory:
            trajectory.append(z.clone())

    if return_trajectory:
        return z, trajectory
    return z


def rk4_solve(
    velocity_fn: Callable[[Tensor, float], Tensor],
    z_init: Tensor,
    t_start: float = 0.0,
    t_end: float = 1.0,
    num_steps: int = 10,
    return_trajectory: bool = False,
) -> Tensor | Tuple[Tensor, List[Tensor]]:
    """
    4th-order Runge-Kutta method for ODE integration.

    More accurate than Euler for the same step count, but 4x more expensive.

    Args:
        velocity_fn: Function (z, t) -> velocity.
        z_init: Initial state (B, T, D) or (B, D).
        t_start: Starting time.
        t_end: Ending time.
        num_steps: Number of RK4 steps.
        return_trajectory: If True, return list of intermediate states.

    Returns:
        Final state z_end, optionally with trajectory list.
    """
    dt = (t_end - t_start) / num_steps
    z = z_init.clone()
    trajectory = [z.clone()] if return_trajectory else None

    for i in range(num_steps):
        t = t_start + i * dt

        k1 = velocity_fn(z, t)
        k2 = velocity_fn(z + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = velocity_fn(z + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = velocity_fn(z + dt * k3, t + dt)

        z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        if return_trajectory:
            trajectory.append(z.clone())

    if return_trajectory:
        return z, trajectory
    return z


def ode_solve(
    velocity_fn: Callable[[Tensor, float], Tensor],
    z_init: Tensor,
    t_start: float = 0.0,
    t_end: float = 1.0,
    num_steps: int = 10,
    method: str = "euler",
    return_trajectory: bool = False,
) -> Tensor | Tuple[Tensor, List[Tensor]]:
    """
    Unified ODE solver interface.

    Args:
        velocity_fn: Function (z, t) -> velocity.
        z_init: Initial state.
        t_start: Starting time.
        t_end: Ending time.
        num_steps: Number of integration steps.
        method: 'euler' or 'rk4'.
        return_trajectory: If True, return intermediate states.

    Returns:
        Final state, optionally with trajectory.
    """
    if method == "euler":
        return euler_solve(
            velocity_fn, z_init, t_start, t_end, num_steps, return_trajectory
        )
    elif method == "rk4":
        return rk4_solve(
            velocity_fn, z_init, t_start, t_end, num_steps, return_trajectory
        )
    else:
        raise ValueError(f"Unknown ODE method: {method}. Use 'euler' or 'rk4'.")


def adaptive_euler_solve(
    velocity_fn: Callable[[Tensor, float], Tensor],
    z_init: Tensor,
    t_start: float = 0.0,
    t_end: float = 1.0,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    max_steps: int = 1000,
    min_dt: float = 1e-6,
) -> Tuple[Tensor, int]:
    """
    Adaptive step-size Euler method with error estimation.

    Uses step doubling to estimate error and adjust step size.
    Useful when velocity field has varying stiffness.

    Args:
        velocity_fn: Function (z, t) -> velocity.
        z_init: Initial state.
        t_start: Starting time.
        t_end: Ending time.
        atol: Absolute tolerance.
        rtol: Relative tolerance.
        max_steps: Maximum number of steps.
        min_dt: Minimum step size.

    Returns:
        Final state and number of steps taken.
    """
    z = z_init.clone()
    t = t_start
    dt = (t_end - t_start) / 10  # Initial step size
    steps = 0

    while t < t_end and steps < max_steps:
        # Ensure we don't overshoot
        dt = min(dt, t_end - t)

        # Full step
        v = velocity_fn(z, t)
        z_full = z + dt * v

        # Two half steps
        z_half = z + 0.5 * dt * v
        v_half = velocity_fn(z_half, t + 0.5 * dt)
        z_double = z_half + 0.5 * dt * v_half

        # Error estimate
        error = (z_double - z_full).abs().max()
        tol = atol + rtol * z.abs().max()

        if error < tol:
            # Accept step
            z = z_double  # Use the more accurate estimate
            t = t + dt
            steps += 1

            # Increase step size
            if error < 0.5 * tol:
                dt = min(dt * 1.5, t_end - t)
        else:
            # Reject step, reduce step size
            dt = max(dt * 0.5, min_dt)

    return z, steps
