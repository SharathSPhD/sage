"""The reversibilized-Markov surrogate null — promoted from experiment code.

This is the null that survived the F-0008 retraction and certified F-0009:
symmetrize the observed pair flux (C + Cᵀ)/2, renormalise rows, and sample
surrogate chains from the resulting detailed-balance-exact, persistence-
matched kernel. A detection means the observed sequence's KLD irreversibility
exceeds the (1 − α) quantile of that null — irreversibility beyond anything
a reversible chain with the same pair statistics can produce.

Promoted into the library (2026-08-12, product directive): the decisive
instrument for real time series must be importable, not buried in an
experiment script.

References
----------
The F-0009 machinery (experiments/electricity_reading.py, unit
domains.electricity); Schnakenberg network theory. Tier: derived.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import jax.numpy as jnp
import numpy as np

from strataq.core.dynamics.sample import trajectory_from_series
from strataq.thermo.estimators import kld_epr

__all__ = ["ReversibilizedNullResult", "reversibilized_null_test"]


@dataclass(frozen=True)
class ReversibilizedNullResult:
    """Verdict of the reversibilized-Markov surrogate test."""

    statistic: float  # KLD irreversibility of the observed sequence (nats/step basis)
    null_quantile: float  # the (1 − alpha) null quantile
    null_median: float
    p_value: float  # fraction of surrogates ≥ statistic (add-one corrected)
    detected: bool
    n_surrogates: int
    null_mismatch_low: bool  # statistic below the null's 5% quantile: model mismatch flag


def reversibilized_null_test(
    states: np.ndarray | list[int],
    n_states: int,
    *,
    n_surrogates: int = 200,
    alpha: float = 0.01,
    seed: int = 0,
    k: int = 1,
) -> ReversibilizedNullResult:
    """Test a discrete state sequence for irreversibility beyond a reversible chain.

    The null preserves the observed symmetrized pair flux (persistence
    included) and is detailed-balance-exact by construction. ``k`` is the
    KLD block order (k = 1 is the pair-level test used on the market data).
    """
    seq = np.asarray(states, dtype=np.int64)
    if seq.ndim != 1 or len(seq) < 10:
        raise ValueError("need a 1-D state sequence with at least 10 observations")
    if seq.min() < 0 or seq.max() >= n_states:
        raise ValueError("states must lie in [0, n_states)")

    stat = float(kld_epr(trajectory_from_series(jnp.asarray(seq), n_states), k=k))

    counts = np.zeros((n_states, n_states))
    for a, b in pairwise(seq):
        counts[a, b] += 1.0
    sym = (counts + counts.T) / 2.0
    rows = sym.sum(axis=1, keepdims=True)
    kernel = np.divide(sym, rows, out=np.full_like(sym, 1.0 / n_states), where=rows > 0)

    rng = np.random.default_rng(seed)
    null_stats = np.empty(n_surrogates)
    for i in range(n_surrogates):
        out = np.empty(len(seq), dtype=np.int64)
        out[0] = seq[0]
        for t in range(1, len(seq)):
            out[t] = rng.choice(n_states, p=kernel[out[t - 1]])
        null_stats[i] = float(kld_epr(trajectory_from_series(jnp.asarray(out), n_states), k=k))

    q_hi = float(np.quantile(null_stats, 1.0 - alpha))
    p_val = (1.0 + float(np.sum(null_stats >= stat))) / (n_surrogates + 1.0)
    return ReversibilizedNullResult(
        statistic=stat,
        null_quantile=q_hi,
        null_median=float(np.median(null_stats)),
        p_value=p_val,
        detected=stat > q_hi,
        n_surrogates=n_surrogates,
        null_mismatch_low=stat < float(np.quantile(null_stats, 0.05)),
    )
