"""Trajectory sampling of the Glauber jump process via uniformisation.

The CTMC with generator L is simulated exactly as its uniformised skeleton
chain P = I + L/Λ (Λ = max exit rate) with i.i.d. Exponential(Λ) holding
times — the same law as Gillespie on the uniformised generator. Two exact
facts the estimator layer leans on:

- the skeleton chain shares the CTMC's stationary distribution;
- per-step entropy production of the skeleton equals EPR/Λ (the log-ratios
  π_a P_ab / π_b P_ba coincide with the CTMC edge ratios; self-loops are
  symmetric and contribute nothing).

References
----------
Uniformisation: standard CTMC theory (Jensen 1953). PROGRAMME v3 §3.5.
Tier: exact.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from strataq.core.dynamics.markov import stationary_distribution


class TrajectoryBatch(eqx.Module):
    """M skeleton trajectories with their exponential holding times."""

    states: Array  # (M, N+1) int32 profile indices
    dt: Array  # (M, N) holding times, Exponential(rate)
    rate: Array  # scalar uniformisation rate Λ
    n_states: int = eqx.field(static=True)


def uniformized_chain(generator: Array) -> tuple[Array, Array]:
    """Skeleton kernel P = I + L/Λ and the rate Λ = max_a |L_aa|."""
    rate = jnp.max(-jnp.diag(generator))
    kernel = jnp.eye(generator.shape[0]) + generator / rate
    return jnp.maximum(kernel, 0.0), rate


def sample_trajectories(
    generator: Array,
    key: Array,
    *,
    n_steps: int,
    n_trajectories: int,
    init: Array | None = None,
) -> TrajectoryBatch:
    """Sample M independent stationary trajectories of N steps each.

    Initial states are drawn from ``init`` (default: the exact stationary
    distribution, so every step is a stationary sample — no burn-in).
    """
    kernel, rate = uniformized_chain(generator)
    start = stationary_distribution(generator) if init is None else init
    log_kernel = jnp.log(jnp.maximum(kernel, 1e-300))

    init_key, step_key, time_key = jax.random.split(key, 3)
    s0 = jax.random.categorical(init_key, jnp.log(start), shape=(n_trajectories,))

    def one_step(state: Array, k: Array) -> tuple[Array, Array]:
        nxt = jax.random.categorical(k, log_kernel[state])
        return nxt, nxt

    def one_trajectory(s: Array, k: Array) -> Array:
        _, path = jax.lax.scan(one_step, s, jax.random.split(k, n_steps))
        return jnp.concatenate([s[None], path])

    states = jax.vmap(one_trajectory)(s0, jax.random.split(step_key, n_trajectories))
    dt = jax.random.exponential(time_key, shape=(n_trajectories, n_steps)) / rate
    return TrajectoryBatch(
        states=states.astype(jnp.int32),
        dt=dt,
        rate=rate,
        n_states=generator.shape[0],
    )


def trajectory_from_series(states: Array, n_states: int, *, dt: float = 1.0) -> TrajectoryBatch:
    """Wrap an OBSERVED discrete state sequence for the estimator layer.

    Real data arrives at fixed sampling intervals, not exponential holding
    times; the KLD block identity holds for any stationary discrete-time
    chain with per-step entropy production converted to per-time by
    rate = 1/dt. TUR windows use the fixed dt exactly.
    """
    seq = jnp.asarray(states, dtype=jnp.int32).reshape(1, -1)
    n = seq.shape[1] - 1
    return TrajectoryBatch(
        states=seq,
        dt=jnp.full((1, n), float(dt)),
        rate=jnp.asarray(1.0 / float(dt)),
        n_states=int(n_states),
    )
