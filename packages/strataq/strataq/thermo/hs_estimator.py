"""Hatano–Sasa Y from observed trajectories — the data-facing quench meter.

Real systems never hand over π_λ. The plug-in estimator pools occupation
frequencies inside each hold window into π̂_k (Laplace-smoothed) and
accumulates, per trajectory,

    Ŷ = Σ_k ln π̂_{k−1}(s at switch k) − ln π̂_k(s at switch k),

with s the trajectory's last state of window k−1. **The instrument carries
its own validity meter**: the true Y satisfies ⟨e^{−Y}⟩ = 1 exactly, so the
sampled ⟨e^{−Ŷ}⟩ CI bracketing 1 diagnoses the plug-in — windows too short
for π̂ to be stationary push it off 1, and the estimator then reports
``usable = False`` instead of a number to quote.

References
----------
Hatano–Sasa 2001; the exact layer in strataq.thermo.protocols (unit
thermo.protocols). Tier: derived.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import expm

from strataq.core.dynamics.markov import glauber_generator
from strataq.core.dynamics.markov import stationary_distribution as _stationary
from strataq.finite.games.tensor import DenseTensorGame
from strataq.thermo.protocols import QuenchProtocol

__all__ = ["HSEstimate", "hs_y_estimate", "sample_quench_states"]

_FLOOR = 1e-300


def sample_quench_states(
    game: DenseTensorGame,
    protocol: QuenchProtocol,
    *,
    n_trajectories: int,
    steps_per_unit_time: int = 20,
    seed: int = 0,
) -> list[np.ndarray]:
    """Sampled per-hold state windows for a stepwise quench.

    Returns one ``(n_trajectories, n_sub_k)`` int array per hold k ≥ 1 —
    exactly the shape of data a discretised observed quench would give.
    Initial states are stationary at λ₀; each segment steps the EXACT
    finite-time kernel e^{Lτ/n}.
    """
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    pi0 = _stationary(glauber_generator(game, float(protocol.lambdas[0])))
    states = jax.random.categorical(init_key, jnp.log(pi0), shape=(n_trajectories,))
    windows: list[np.ndarray] = []
    for k in range(1, protocol.lambdas.shape[0]):
        gen = glauber_generator(game, float(protocol.lambdas[k]))
        tau = float(protocol.taus[k - 1])
        n_sub = max(1, round(tau * steps_per_unit_time))
        kernel = expm((tau / n_sub) * gen)
        log_kernel = jnp.log(jnp.maximum(kernel, _FLOOR))
        seq = np.empty((n_trajectories, n_sub), dtype=np.int64)
        for t in range(n_sub):
            key, step_key = jax.random.split(key)
            states = jax.random.categorical(step_key, log_kernel[states])
            seq[:, t] = np.asarray(states)
        windows.append(seq)
    return windows


@dataclass(frozen=True)
class HSEstimate:
    """Plug-in Hatano–Sasa read with its self-calibration verdict."""

    mean_y: float
    mean_y_ci_low: float
    mean_y_ci_high: float
    ift_estimate: float
    ift_ci_low: float
    ift_ci_high: float
    usable: bool  # the IFT diagnostic passed; only then quote mean_y
    n_trajectories: int
    warnings: list[str] = field(default_factory=list)


def hs_y_estimate(
    windows: list[np.ndarray],
    *,
    n_states: int,
    pseudocount: float = 0.5,
    burn_in_fraction: float = 0.25,
    ift_tolerance: float = 0.05,
    ci_level: float = 0.95,
) -> HSEstimate:
    """Estimate ⟨Y⟩ from per-hold observed state windows.

    ``windows[k]`` is an ``(n_trajectories, len_k)`` int array of states
    during hold k (after the k-th λ switch); the pre-quench stationary
    window may be prepended as ``windows[0]`` or omitted — the estimator
    uses window k−1's occupation as π̂ entering switch k, so at least two
    windows are required.
    """
    if len(windows) < 2:
        raise ValueError("need at least two hold windows")
    n_traj = windows[0].shape[0]
    for w in windows:
        if w.ndim != 2 or w.shape[0] != n_traj:
            raise ValueError("every window needs the same number of trajectories (rows)")
        if w.min() < 0 or w.max() >= n_states:
            raise ValueError(f"states must lie in [0, n_states={n_states})")

    def pi_hat(w: np.ndarray) -> np.ndarray:
        # drop the relaxation transient: occupation is only stationary-like
        # after the hold has settled, so the leading burn_in_fraction of each
        # window is excluded from the pi estimate (the switch state itself is
        # still read from the window's END, which is the most-relaxed sample)
        start = int(burn_in_fraction * w.shape[1])
        counts = (
            np.bincount(w[:, start:].reshape(-1), minlength=n_states).astype(float) + pseudocount
        )
        return counts / counts.sum()

    pis = [pi_hat(w) for w in windows]
    y = np.zeros(n_traj)
    for k in range(1, len(windows)):
        s = windows[k - 1][:, -1]  # state at the switch
        y += np.log(pis[k - 1][s]) - np.log(pis[k][s])

    z = (
        1.959963984540054
        if ci_level == 0.95
        else float(
            abs(np.quantile(np.random.default_rng(0).standard_normal(200000), (1 - ci_level) / 2))
        )
    )
    mean_y = float(np.mean(y))
    half_y = z * float(np.std(y, ddof=1)) / float(np.sqrt(n_traj))
    vals = np.exp(-y)
    ift = float(np.mean(vals))
    half_ift = z * float(np.std(vals, ddof=1)) / float(np.sqrt(n_traj))
    # EQUIVALENCE-style diagnostic (upgraded after the registered P3 run
    # failed with a covers-1 rule — non-monotone at the boundary, because
    # failure-to-reject flips by seed): usable demands the whole IFT CI
    # inside [1 - tol, 1 + tol] — positive evidence of closeness to the
    # identity, not absence of evidence against it.
    usable = (1.0 - ift_tolerance) <= ift - half_ift and ift + half_ift <= 1.0 + ift_tolerance
    warnings = []
    if not usable:
        warnings.append(
            "IFT diagnostic failed: <e^{-Y_hat}> CI is not contained in "
            f"[{1 - ift_tolerance}, {1 + ift_tolerance}] — either the hold windows "
            "are too short for the plug-in stationary estimates (or the system is "
            "not stepwise-stationary), or the sample is too small to certify the "
            "identity; do NOT quote mean_y from this read"
        )
    min_len = min(w.shape[1] for w in windows)
    if min_len * n_traj < 20 * n_states:
        warnings.append(
            f"thin occupation statistics (shortest window {min_len} samples x "
            f"{n_traj} trajectories for {n_states} states): pi_hat is noisy"
        )
    return HSEstimate(
        mean_y=mean_y,
        mean_y_ci_low=mean_y - half_y,
        mean_y_ci_high=mean_y + half_y,
        ift_estimate=ift,
        ift_ci_low=ift - half_ift,
        ift_ci_high=ift + half_ift,
        usable=usable,
        n_trajectories=n_traj,
        warnings=warnings,
    )
