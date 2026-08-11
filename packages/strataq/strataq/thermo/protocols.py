"""Hatano–Sasa decomposition and λ-quench protocols (PROGRAMME v3 §3.5 remainder).

Two exact layers over the Glauber chain:

1. **Housekeeping/excess split** of the instantaneous entropy production for
   an arbitrary distribution p relative to the generator's stationary π:
   σ_tot = σ_hk + σ_ex, with σ_hk the adiabatic (NESS-maintaining) part —
   identically zero under detailed balance, i.e. on potential games — and
   σ_ex = −d/dt D(p‖π) the relaxation part. Both are individually
   nonnegative (Esposito–Van den Broeck).

2. **Stepwise λ-quench protocols** with their integral fluctuation theorems:
   the Hatano–Sasa IFT ⟨e^{−Y}⟩ = 1 holds for ANY game — including
   harmonic ones whose stationary states are NESSes — while the Jarzynski
   form ⟨e^{ΣΔλΦ}⟩ = Z_K/Z_0 needs a potential Φ. Both are computed exactly
   by weighted-vector transfer (machine-precision identities) and, because
   real data only ever sees trajectories, re-estimated from sampled paths
   with a bootstrap-free CLT interval.

References
----------
Hatano–Sasa 2001; Esposito–Van den Broeck 2010 (three faces of the second
law); Jarzynski 1997. Tier: exact (identities), derived (sampled reads).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import expm

from strataq.core.dynamics.markov import glauber_generator, stationary_distribution
from strataq.finite.games.tensor import DenseTensorGame

__all__ = [
    "QuenchProtocol",
    "epr_split",
    "hatano_sasa_exact",
    "hatano_sasa_sampled",
    "jarzynski_exact",
    "relax",
]

_FLOOR = 1e-300


class QuenchProtocol(eqx.Module):
    """Stepwise protocol: hold λ_k for τ_k after each instantaneous switch.

    ``lambdas`` has K+1 entries (the system starts stationary at λ_0);
    ``taus`` has K entries — the relaxation time spent at each λ_{k≥1}.
    """

    lambdas: Array
    taus: Array

    def __check_init__(self) -> None:
        if self.lambdas.shape[0] != self.taus.shape[0] + 1:
            raise ValueError("need len(lambdas) == len(taus) + 1")


def epr_split(generator: Array, pi: Array, p: Array) -> tuple[Array, Array, Array]:
    """(total, housekeeping, excess) entropy production of distribution p.

    σ_tot(p) = Σ_{a≠b} p_a w_ab ln[(p_a w_ab)/(p_b w_ba)]
    σ_hk(p)  = Σ_{a≠b} p_a w_ab ln[(w_ab π_a)/(w_ba π_b)]
    σ_ex(p)  = σ_tot − σ_hk = Σ_{a≠b} p_a w_ab ln[(π_b p_a)/(π_a p_b)]

    Glauber-logit rates are strictly positive, so every ratio is finite and
    the split is a finite sum — no clamping beyond a numeric floor.
    """
    rates = generator - jnp.diag(jnp.diag(generator))
    off = ~jnp.eye(rates.shape[0], dtype=bool)
    flux = p[:, None] * rates
    log_flux_ratio = jnp.where(
        off, jnp.log(jnp.maximum(flux, _FLOOR)) - jnp.log(jnp.maximum(flux.T, _FLOOR)), 0.0
    )
    ness_flux = pi[:, None] * rates
    log_hk_ratio = jnp.where(
        off,
        jnp.log(jnp.maximum(ness_flux, _FLOOR)) - jnp.log(jnp.maximum(ness_flux.T, _FLOOR)),
        0.0,
    )
    total = jnp.sum(jnp.where(off, flux * log_flux_ratio, 0.0))
    hk = jnp.sum(jnp.where(off, flux * log_hk_ratio, 0.0))
    return total, hk, total - hk


def relax(p: Array, generator: Array, t: float | Array) -> Array:
    """Propagate a distribution for time t under a fixed generator (dense expm)."""
    return jnp.asarray(p @ expm(jnp.asarray(t) * generator))


def _protocol_tables(
    game: DenseTensorGame, protocol: QuenchProtocol
) -> tuple[list[Array], list[Array]]:
    """Per-λ generators and stationary distributions along the protocol."""
    gens = [glauber_generator(game, float(lam)) for lam in protocol.lambdas]
    pis = [stationary_distribution(g) for g in gens]
    return gens, pis


def hatano_sasa_exact(game: DenseTensorGame, protocol: QuenchProtocol) -> tuple[Array, Array]:
    """(⟨e^{−Y}⟩, ⟨Y⟩) for the Hatano–Sasa functional, by exact transfer.

    Y = Σ_k ln[π_{λ_{k−1}}(s_k)/π_{λ_k}(s_k)] with s_k the state at the k-th
    switch. ⟨e^{−Y}⟩ = 1 is an identity for any game (the NESS-level second
    law); ⟨Y⟩ ≥ 0 is its Jensen consequence, vanishing quasi-statically —
    the protocol's excess dissipation, computed from the actual lagging p(t).

    HONESTY NOTE (red-team O-1): for stepwise protocols the weighted-vector
    transfer TELESCOPES — after each switch the weighted vector equals the
    new π exactly, which relax() leaves invariant — so the ⟨e^{−Y}⟩ = 1
    output is an algebraic identity verification that cannot catch a bug in
    relax() or the generators. The correctness test of the kernels is the
    independent sampled path in :func:`hatano_sasa_sampled`. ⟨Y⟩, by
    contrast, is a genuine computation (it depends on the lagging p(t)).
    """
    gens, pis = _protocol_tables(game, protocol)
    weighted = pis[0]  # E[e^{-Y} · 1{path}] mass vector
    p = pis[0]  # the physical (unweighted) distribution
    mean_y = jnp.zeros(())
    for k in range(1, len(gens)):
        jump = jnp.log(pis[k - 1]) - jnp.log(pis[k])
        mean_y = mean_y + jnp.sum(p * jump)
        weighted = weighted * jnp.exp(-jump)
        tau = protocol.taus[k - 1]
        weighted = relax(weighted, gens[k], tau)
        p = relax(p, gens[k], tau)
    return jnp.sum(weighted), mean_y


def jarzynski_exact(
    game: DenseTensorGame, phi: Array, protocol: QuenchProtocol
) -> tuple[Array, Array]:
    """(⟨e^{ΣΔλΦ(s_k)}⟩, Z_K/Z_0) — the potential-game Jarzynski equality.

    Only meaningful when ``phi`` is an exact potential for the game (the
    Gibbs form π_λ ∝ e^{λΦ} is what turns −ΔλΦ into work). The caller owns
    that premise; on a non-potential game the equality simply fails.
    """
    gens, _ = _protocol_tables(game, protocol)
    phi = phi.reshape(-1)
    log_z = jax.vmap(lambda lam: jax.scipy.special.logsumexp(lam * phi))(protocol.lambdas)
    weighted = jnp.exp(protocol.lambdas[0] * phi - log_z[0])  # π_{λ0}
    for k in range(1, len(gens)):
        dlam = protocol.lambdas[k] - protocol.lambdas[k - 1]
        weighted = weighted * jnp.exp(dlam * phi)
        weighted = relax(weighted, gens[k], protocol.taus[k - 1])
    return jnp.sum(weighted), jnp.exp(log_z[-1] - log_z[0])


def hatano_sasa_sampled(
    game: DenseTensorGame,
    protocol: QuenchProtocol,
    *,
    n_trajectories: int,
    steps_per_unit_time: int = 20,
    seed: int = 0,
) -> tuple[float, float, float, float]:
    """(⟨e^{−Y}⟩ estimate, CI low, CI high, ⟨Y⟩) from sampled quench paths.

    The data-side face of the IFT: initial states drawn from π_{λ_0}, each
    relaxation segment stepped with the EXACT finite-time kernel e^{Lτ/n}
    (sub-stepping changes nothing in distribution — it mirrors a sampled
    time series). The 95% interval is the CLT band of the e^{−Y} mean; the
    heavy right tail of e^{−Y} is exactly why finite-sample Jarzynski
    estimates are biased low — visible here, quantified by the CI.
    """
    gens, pis = _protocol_tables(game, protocol)
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    states = jax.random.categorical(init_key, jnp.log(pis[0]), shape=(n_trajectories,))
    y = jnp.zeros((n_trajectories,))
    for k in range(1, len(gens)):
        y = y + (jnp.log(pis[k - 1]) - jnp.log(pis[k]))[states]
        n_sub = max(1, round(float(protocol.taus[k - 1]) * steps_per_unit_time))
        kernel = expm((protocol.taus[k - 1] / n_sub) * gens[k])
        log_kernel = jnp.log(jnp.maximum(kernel, _FLOOR))
        for _ in range(n_sub):
            key, step_key = jax.random.split(key)
            states = jax.random.categorical(step_key, log_kernel[states])
    vals = jnp.exp(-y)
    est = float(jnp.mean(vals))
    half = 1.96 * float(jnp.std(vals, ddof=1)) / float(jnp.sqrt(n_trajectories))
    return est, est - half, est + half, float(jnp.mean(y))
