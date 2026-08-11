"""Hatano-Sasa Y from observed trajectories - EXPERIMENTAL, NOT CERTIFIED.

.. warning::
   Red-team verdict WITHHELD (2026-08-12, F-0016): multi-seed coverage at
   the one admitted hold is ~2/5 (residual plug-in bias beyond the CI); the
   relaxation-time underestimate is game-dependent (up to ~19x at alpha=0,
   far beyond any fixed safety factor); the autocorrelation gate breaks at
   small n_trajectories (24x underestimate at n=1). Do NOT use this module
   for scientific claims - it is retained as the measured failure map and
   the starting point for a redesign (F-0016 continuation notes).

Real systems never hand over π_λ. The plug-in estimator pools occupation
frequencies inside each hold window into π̂_k (Laplace-smoothed) and
accumulates, per trajectory,

    Ŷ = Σ_k ln π̂_{k−1}(s at switch k) − ln π̂_k(s at switch k),

with s the trajectory's last state of window k−1. **The usability gate is
the per-window relaxation check** (F-0016): each hold must exceed
``relax_safety`` × its own data-estimated relaxation time, else π̂ never
settled and the read is refused. The IFT ⟨e^{−Ŷ}⟩ ≈ 1 equivalence check is
only a COMPANION — it is measurably insufficient alone (a 45% bias can
hide behind an IFT of 1.01, because state-correlated plug-in errors cancel
in the exponential average but not the mean).

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


def _relaxation_time(window: np.ndarray, n_states: int, dt: float) -> float:
    """Data-side relaxation-time estimate from within-window autocorrelation.

    Categorical autocorrelation at lag N/4 (long lags see the slow modes a
    lag-1 estimate under-weights), fitted as a single exponential. This is
    what makes the usability gate estimable from observations alone.
    """
    # lag N/4 sees slow modes far better than lag 1, but still UNDERESTIMATES
    # them (~25% measured on a gap-collapsed window: 5.2 vs true 6.6) — the
    # default relax_safety of 4 exists to absorb exactly that bias
    lag = max(1, window.shape[1] // 4)
    a, b = window[:, :-lag].reshape(-1), window[:, lag:].reshape(-1)
    pi = np.bincount(window.reshape(-1), minlength=n_states) / window.size
    base = float(np.sum(pi**2))
    rho = (float(np.mean(a == b)) - base) / max(1.0 - base, 1e-12)
    rho = min(max(rho, 1e-12), 1.0 - 1e-12)
    return float(-lag * dt / np.log(rho))


def hs_y_estimate(
    windows: list[np.ndarray],
    *,
    n_states: int,
    hold_durations: list[float] | None = None,
    pseudocount: float = 0.5,
    burn_in_fraction: float = 0.25,
    relax_safety: float = 4.0,
    ift_tolerance: float = 0.05,
    ci_level: float = 0.95,
) -> HSEstimate:
    """Estimate ⟨Y⟩ from per-hold observed state windows.

    ``windows[k]`` is an ``(n_trajectories, len_k)`` int array of states
    during hold k (after the k-th λ switch); the pre-quench stationary
    window may be prepended as ``windows[0]`` or omitted — the estimator
    uses window k−1's occupation as π̂ entering switch k, so at least two
    windows are required.

    **The primary usability gate** (F-0016): with ``hold_durations`` (the τ
    of each hold, in the same time units the data was sampled at), every
    window must satisfy τ ≥ ``relax_safety`` × its own estimated relaxation
    time — otherwise the hold never settled and π̂ is biased regardless of
    what the IFT says. The IFT equivalence check is the necessary COMPANION
    diagnostic, not the primary one: ⟨e^{−Ŷ}⟩ can sit at 1 while ⟨Ŷ⟩ is
    badly biased (measured: 45% bias behind an IFT of 1.01). Without
    ``hold_durations`` the relaxation gate cannot run and the read is
    marked unusable with an explicit warning.
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
    # companion diagnostic (equivalence-style; known-insufficient alone —
    # F-0016): the whole IFT CI must sit inside [1 - tol, 1 + tol]
    ift_ok = (1.0 - ift_tolerance) <= ift - half_ift and ift + half_ift <= 1.0 + ift_tolerance
    warnings = []
    # PRIMARY gate (F-0016): every hold must be long relative to its own
    # data-estimated relaxation time, else pi_hat never settled
    relax_ok = False
    if hold_durations is None:
        warnings.append(
            "hold_durations not supplied: the primary relaxation gate cannot run, "
            "so this read is marked unusable — pass the hold lengths (same time "
            "units as your sampling) to enable it"
        )
    else:
        if len(hold_durations) != len(windows):
            raise ValueError("hold_durations must have one entry per window")
        relax_times = [
            _relaxation_time(w, n_states, float(tau) / w.shape[1])
            for w, tau in zip(windows, hold_durations, strict=True)
        ]
        offenders = [
            k
            for k, (tau, tr) in enumerate(zip(hold_durations, relax_times, strict=True))
            if float(tau) < relax_safety * tr
        ]
        relax_ok = not offenders
        if offenders:
            worst = max(offenders, key=lambda k: relax_times[k] / float(hold_durations[k]))
            warnings.append(
                f"relaxation gate failed for {len(offenders)} window(s) (worst: window "
                f"{worst}, hold {float(hold_durations[worst]):g} < {relax_safety} x "
                f"estimated relaxation time {relax_times[worst]:.2f}) — the hold never "
                "settled; pi_hat is biased there and mean_y must not be quoted"
            )
    usable = relax_ok and ift_ok
    if relax_ok and not ift_ok:
        warnings.append(
            "IFT companion diagnostic failed: <e^{-Y_hat}> CI is not contained in "
            f"[{1 - ift_tolerance}, {1 + ift_tolerance}] despite settled holds — "
            "sample too small to certify the identity, or the system is not "
            "stepwise-stationary; do NOT quote mean_y from this read"
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
