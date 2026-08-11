"""Irreversibility estimators from observed trajectories (PROGRAMME v3 §3.5).

Two of the three planned estimators; NEEP-style neural estimation is deferred
(would need its own ADR). Both are validated against the exact Schnakenberg
EPR of the same generator before ever touching data:

1. **KLD / k-th order Markov** — plug-in estimate of
   σ̂⁽ᵏ⁾ = (1/kτ) Σ P(Y₀:ₖ) log P(Y₀:ₖ)/P(Yₖ:₀) with τ = 1/Λ per skeleton
   step. For a stationary Markov chain the (k+1)-block KLD equals
   k · (per-step EP) exactly, so every k targets the same rate — as a
   population identity. The plug-in needs n_samples ≫ n_states^(k+1)
   (see :func:`kld_epr` for the practical k ceiling); bias is
   O(n_cells / n_samples) and data-starved k underestimates.

2. **TUR lower bound** — σ ≥ 2⟨J⟩²/(Var(J)·T) from the first two cumulants
   of a time-integrated empirical current. A certified bound, not a point
   estimate; the headline number for partial-observation settings because
   it degrades gracefully (a worse current choice loosens, never breaks it).

References
----------
Roldán–Parrondo PRL 2010 (KLD); Barato–Seifert PRL 2015, Gingrich et al.
PRL 2016 (TUR); Otsubo et al. Comms Phys 2022 (estimator comparison).
Tier: exact identities; the estimators themselves are statistical.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from strataq.core.dynamics.currents import probability_currents
from strataq.core.dynamics.markov import stationary_distribution
from strataq.core.dynamics.sample import TrajectoryBatch


def _reversed_codes(n_states: int, block: int) -> Array:
    """Permutation sending each base-n block code to its digit-reversed code."""
    codes = jnp.arange(n_states**block)
    rev = jnp.zeros_like(codes)
    c = codes
    for _ in range(block):
        rev = rev * n_states + c % n_states
        c = c // n_states
    return rev


def kld_epr(batch: TrajectoryBatch, *, k: int = 1) -> Array:
    """KLD estimate of the entropy production rate (per unit time).

    Plug-in over (k+1)-blocks of the skeleton chain, converted to CTMC time
    by the stored uniformisation rate. Blocks whose reversal was never
    observed are dropped (standard plug-in truncation; vanishes as data
    grows since Glauber rates are strictly positive). On OBSERVED data with
    effectively one-way transitions (near-deterministic dynamics) the
    truncation clips infinite contributions — the estimate is then a lower
    bound, and a pure deterministic cycle reads 0, not ∞.

    **Choosing k**: the block identity holds for every k, but the plug-in
    needs n_samples ≫ n_states^(k+1) — data-starved k *underestimates*
    (unobserved blocks are dropped). On a 9-state game with ~5×10⁵ steps,
    k ≤ 2 is accurate to ~1%, k = 4 already reads ~17% low. Raise k only
    for non-Markov (partially observed) data, and raise n with it.
    """
    n = batch.n_states
    n_cells = n ** (k + 1)
    length = batch.states.shape[1]
    code = jnp.zeros_like(batch.states[:, : length - k])
    for j in range(k + 1):
        code = code + batch.states[:, j : length - k + j] * n**j
    counts = jnp.bincount(code.reshape(-1), length=n_cells)
    p_fwd = counts / jnp.sum(counts)
    p_rev = p_fwd[_reversed_codes(n, k + 1)]
    both = (p_fwd > 0) & (p_rev > 0)
    ratio = jnp.where(both, p_fwd / jnp.where(both, p_rev, 1.0), 1.0)
    per_step = jnp.sum(jnp.where(both, p_fwd * jnp.log(ratio), 0.0)) / k
    return batch.rate * per_step


def stationary_current_weights(generator: Array) -> Array:
    """Antisymmetric ±1 weights aligned with the exact stationary current.

    Oracle-informed: uses the generator, so this is for validation and
    synthetic tightness studies, not for blind data application.
    """
    pi = stationary_distribution(generator)
    return jnp.sign(probability_currents(generator, pi))


def empirical_flux_weights(batch: TrajectoryBatch, *, n_states: int) -> Array:
    """Antisymmetric ±1 weights from the data's own net transition flux.

    Plug-in choice for data settings. Deriving weights and evaluating the
    bound on the same sample is mildly optimistic; for strict certification
    derive weights on a held-out split and pass a fresh batch to
    :func:`tur_epr_bound`.
    """
    pair = batch.states[:, :-1] * n_states + batch.states[:, 1:]
    counts = jnp.bincount(pair.reshape(-1), length=n_states * n_states)
    flux = counts.reshape(n_states, n_states)
    return jnp.sign(flux - flux.T)


def window_currents(batch: TrajectoryBatch, weights: Array) -> tuple[Array, Array]:
    """Per-trajectory time-integrated currents over a common fixed horizon.

    The TUR is a statement about currents over a **fixed time horizon**:
    counting a fixed number of jumps instead (time random) suppresses the
    Poisson event-count fluctuation, underestimates Var(J), and can push the
    "bound" above the true EPR. So every trajectory is truncated at the
    common horizon T = min_m Σ dt_m (jumps after T contribute nothing).
    Returns (J array of shape (M,), horizon T).
    """
    horizon = jnp.min(jnp.sum(batch.dt, axis=1))
    within = jnp.cumsum(batch.dt, axis=1) <= horizon
    steps = weights[batch.states[:, :-1], batch.states[:, 1:]]
    return jnp.sum(jnp.where(within, steps, 0.0), axis=1), horizon


def tur_epr_bound(batch: TrajectoryBatch, weights: Array) -> Array:
    """TUR **point estimate** of the lower bound 2⟨J_T⟩² / (Var(J_T)·T).

    The TUR inequality certifies the *population* cumulants; this sample
    version is NOT itself guaranteed to sit below the true EPR. Near
    equilibrium the TUR saturates (population ratio → 1), so the point
    estimate legitimately straddles the exact value (observed up to ~7%
    above at 128 windows). For a statement you can certify, use the lower
    bootstrap quantile from :func:`tur_epr_bound_ci`.

    Two small-sample corrections keep the estimator from sitting above the
    bound it estimates: E[J̄²] = ⟨J⟩² + Var/M (subtract the plug-in excess)
    and E[1/V̂ar] = (M−1)/((M−3)·Var) under approximate normality of J_T
    (multiply by (M−3)/(M−1); J_T is a sum of many weakly dependent
    increments, so the CLT regime is the operative one).
    """
    j, horizon = window_currents(batch, weights)
    m = j.shape[0]
    var = jnp.var(j, ddof=1)
    mean_sq = jnp.maximum(jnp.mean(j) ** 2 - var / m, 0.0)
    return 2.0 * mean_sq * (m - 3) / ((m - 1) * var * horizon)


def tur_epr_bound_ci(
    batch: TrajectoryBatch,
    weights: Array,
    key: Array,
    *,
    n_resamples: int = 2000,
    quantile: float = 0.025,
) -> Array:
    """Bootstrap quantile of the TUR bound over window resamples.

    The default 2.5% quantile is the certifiable statement: "EPR ≥ this
    value" at ~97.5% coverage. This is what gate artifacts check against
    the exact EPR; the point estimate from :func:`tur_epr_bound` is
    reported alongside but never certified.
    """
    j, horizon = window_currents(batch, weights)
    m = j.shape[0]
    r = j[jax.random.randint(key, (n_resamples, m), 0, m)]
    var = jnp.var(r, axis=1, ddof=1)
    mean_sq = jnp.maximum(jnp.mean(r, axis=1) ** 2 - var / m, 0.0)
    boots = 2.0 * mean_sq * (m - 3) / ((m - 1) * var * horizon)
    return jnp.quantile(boots, quantile)
