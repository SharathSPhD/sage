"""Glauber (single-site logit revision) dynamics on the joint profile space.

Continuous-time Markov jump process: from profile a, a uniformly chosen player
i revises to action b_i with logit probability over the *pure* payoffs
u_i(·, a_{-i}). In an exact potential game these are the heat-bath conditionals
of π ∝ e^{λΦ}, so the chain is reversible with that Gibbs stationary measure
(K3, tier: exact). Off potentiality, detailed balance breaks and the chain
settles into a NESS carrying probability current.

Dense generator — exact path for small games (state space Πm_i); the
trajectory/estimator path covers what dense cannot.

References
----------
Blume 1993; Monderer–Shapley 1996 (K3, exact). PROGRAMME v3 §3.5.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
from jax import Array

from strataq.finite.games.tensor import DenseTensorGame


def profile_space(num_actions: tuple[int, ...]) -> list[tuple[int, ...]]:
    """All joint pure profiles, in deterministic lexicographic order."""
    return list(itertools.product(*(range(m) for m in num_actions)))


def glauber_generator(game: DenseTensorGame, lam: float | Array) -> Array:
    """The generator L (n_states × n_states): L[a, a'] = w(a → a'), rows sum to 0.

    Revision opportunities arrive at unit rate per player; the revising player
    draws from softmax(λ_i u_i(·, a_{-i})). Diagonal set to −Σ off-diagonal.
    """
    lam_vec = jnp.asarray(lam, dtype=jnp.float64)
    if lam_vec.ndim == 0:
        lam_vec = jnp.full((game.n_players,), lam_vec)

    states = profile_space(game.num_actions)
    index = {s: k for k, s in enumerate(states)}
    n = len(states)
    gen = jnp.zeros((n, n))

    for a in states:
        row = index[a]
        for i in range(game.n_players):
            payoffs = game.payoff_tensor(i)
            slicer = list(a)
            u_i = jnp.stack(
                [
                    payoffs[tuple([*slicer[:i], b, *slicer[i + 1 :]])]
                    for b in range(game.num_actions[i])
                ]
            )
            probs = jnp.exp(jax.nn.log_softmax(lam_vec[i] * u_i))
            for b in range(game.num_actions[i]):
                if b == a[i]:
                    continue
                target = tuple((*a[:i], b, *a[i + 1 :]))
                gen = gen.at[row, index[target]].add(probs[b])
    gen = gen - jnp.diag(jnp.sum(gen, axis=1))
    return gen


def stationary_distribution(generator: Array, *, tol: float = 1e-12) -> Array:
    """π with πL = 0, Σπ = 1, via the nullspace of Lᵀ (smallest singular vector)."""
    _, _, vt = jnp.linalg.svd(generator.T)
    pi = jnp.abs(vt[-1])
    pi = pi / jnp.sum(pi)
    residual = jnp.max(jnp.abs(pi @ generator))
    if float(residual) > tol * max(1.0, float(jnp.max(jnp.abs(generator)))):
        raise RuntimeError(f"stationary solve residual {float(residual):.2e} above tolerance")
    return pi
