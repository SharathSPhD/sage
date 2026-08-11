"""Hatano-Sasa Y from observed trajectories — the data-facing quench meter.

CERTIFIED (fresh red-team GRANTED 2026-08-12 after a first WITHHELD; the
full arc — two refuted hypotheses, one fixed missing-window bug, five
recorded escalations — is F-0016) within its validated scope: Markov-chain
quenches on 2x2 mixed families (alpha in {0, 0.25}, 4 joint states,
n_trajectories >= 200). Outside that scope the gate margins are unverified
— read the hs_y_estimate docstring's contract before use.

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

    Returns one ``(n_trajectories, n_sub)`` int array per hold — INCLUDING a
    leading pre-quench window observed at λ₀ (stationary). That first window
    is not book-keeping: without it the estimator can only form K−1 of the
    K jump terms and silently drops the λ₀ → λ₁ contribution — the exact
    bug behind F-0016's 2/5 coverage (root-caused 2026-08-12). Each segment
    steps the EXACT finite-time kernel e^{Lτ/n}.
    """
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    gen0 = glauber_generator(game, float(protocol.lambdas[0]))
    pi0 = _stationary(gen0)
    states = jax.random.categorical(init_key, jnp.log(pi0), shape=(n_trajectories,))
    windows: list[np.ndarray] = []
    # pre-quench window: stationary observation at lambda_0 for one mean hold
    tau0 = float(jnp.mean(protocol.taus))
    n_sub0 = max(1, round(tau0 * steps_per_unit_time))
    kernel0 = expm((tau0 / n_sub0) * gen0)
    log_kernel0 = jnp.log(jnp.maximum(kernel0, _FLOOR))
    seq0 = np.empty((n_trajectories, n_sub0), dtype=np.int64)
    for t in range(n_sub0):
        key, step_key = jax.random.split(key)
        states = jax.random.categorical(step_key, log_kernel0[states])
        seq0[:, t] = np.asarray(states)
    windows.append(seq0)
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


def _relaxation_time(window: np.ndarray, n_states: int, dt: float) -> tuple[float, float]:
    """Data-side relaxation-time estimate (value, SE) from autocorrelation.

    Categorical autocorrelation at lag N/4 (long lags see the slow modes a
    lag-1 estimate under-weights), fitted as a single exponential; the SE
    comes from a 4-way trajectory split. The gate uses tau_hat + 2 SE so a
    noisy estimate cannot FLICKER a marginal hold into the usable set
    (measured flicker: usable 4/20 at a boundary hold before this).
    """
    # lag N/4 sees slow modes far better than lag 1, but still UNDERESTIMATES
    # them (~25% measured on a gap-collapsed window: 5.2 vs true 6.6) — the
    # default relax_safety of 4 exists to absorb exactly that bias
    lag = max(1, window.shape[1] // 4)

    def one(win: np.ndarray) -> float:
        a, b = win[:, :-lag].reshape(-1), win[:, lag:].reshape(-1)
        pi = np.bincount(win.reshape(-1), minlength=n_states) / win.size
        base = float(np.sum(pi**2))
        rho = (float(np.mean(a == b)) - base) / max(1.0 - base, 1e-12)
        rho = min(max(rho, 1e-12), 1.0 - 1e-12)
        return float(-lag * dt / np.log(rho))

    est = one(window)
    n_traj = window.shape[0]
    if n_traj >= 8:
        groups = [one(window[i::4]) for i in range(4)]
        se = float(np.std(groups, ddof=1)) / 2.0
    else:
        se = est  # too few trajectories to estimate SE: maximally cautious
    return est, se


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
    during hold k. ``windows[0]`` MUST be the pre-quench stationary window
    (observed at λ₀ before the first switch) — omitting it silently drops
    the first jump term, a bug this estimator's own history documents
    (F-0016). ``hold_durations`` and your sampling interval must share ONE
    time unit: the k-th window's implied sample spacing is
    ``hold_durations[k] / windows[k].shape[1]``, and the relaxation gate is
    computed in that unit — passing milliseconds where the data was described
    in seconds rescales the gate silently (nothing data-side can detect
    that; the contract is yours to honour). Example: 6 holds of 24 time
    units sampled 25×/unit → each window has 600 columns and
    ``hold_durations=[24.0]*6``.

    VALIDATED SCOPE: 2×2 mixed families (α ∈ {0, 0.25}) with 4 joint
    states, n_trajectories ≥ 200; the relaxation-time underestimate is
    game-dependent and only measured there — treat other games' gate
    margins as unverified.

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
        # explicit asarray: numpy stub versions differ on whether the division
        # is Any (CI strict caught this where the local stubs did not)
        return np.asarray(counts / counts.sum(), dtype=float)

    pis = [pi_hat(w) for w in windows]
    y = np.zeros(n_traj)
    for k in range(1, len(windows)):
        s = windows[k - 1][:, -1]  # state at the switch
        y += np.log(pis[k - 1][s]) - np.log(pis[k][s])
    mean_y = float(np.mean(y))
    vals = np.exp(-y)
    ift = float(np.mean(vals))

    # CI by trajectory bootstrap WITH pi_hat re-estimated per resample: the
    # occupation estimates are shared across trajectories, so their noise is
    # a common error invisible to a per-trajectory CLT (measured: seed-to-
    # seed spread ~1.4x the CLT width — the residual coverage failure after
    # the missing-window bug fix)
    rng = np.random.default_rng(0)
    n_boot = 200
    boot_y = np.empty(n_boot)
    boot_ift = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_traj, size=n_traj)
        rp = [pi_hat(w[idx]) for w in windows]
        yb = np.zeros(n_traj)
        for k in range(1, len(windows)):
            s = windows[k - 1][idx, -1]
            yb += np.log(rp[k - 1][s]) - np.log(rp[k][s])
        boot_y[b] = float(np.mean(yb))
        boot_ift[b] = float(np.mean(np.exp(-yb)))
    tail = (1.0 - ci_level) / 2.0
    y_lo, y_hi = (float(q) for q in np.quantile(boot_y, [tail, 1.0 - tail]))
    ift_lo, ift_hi = (float(q) for q in np.quantile(boot_ift, [tail, 1.0 - tail]))
    # companion diagnostic — ROLE CHANGED with the relaxation gate primary
    # (F-0016 history): as an ANOMALY DETECTOR it flags only when the IFT CI
    # EXCLUDES 1 (gross violation: non-stepwise-stationary system or broken
    # sampling). The old equivalence form was right when this check was
    # primary, but as companion it just flickered at CI-width ~ tolerance
    # (measured 3/10 usable at a fully-settled hold); certification power now
    # lives in the relaxation gate, detection power here.
    ift_ok = ift_lo <= 1.0 <= ift_hi
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
        relax_reads = [
            _relaxation_time(w, n_states, float(tau) / w.shape[1])
            for w, tau in zip(windows, hold_durations, strict=True)
        ]
        # noise-aware threshold: tau_hat + 2 SE, so estimate noise cannot
        # flicker a marginal hold into the usable set
        relax_times = [tr + 2.0 * se for tr, se in relax_reads]
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
            "IFT companion ANOMALY: <e^{-Y_hat}> CI excludes 1 despite settled "
            "holds — the system is likely not stepwise-stationary (or the "
            "sampling is broken); do NOT quote mean_y from this read"
        )
    if n_traj < 100:
        warnings.append(
            f"n_trajectories={n_traj} is thin: the bootstrap CI and the "
            "relaxation-SE machinery are validated at n >= 200; below ~100 "
            "treat every number here as indicative only"
        )
    min_len = min(w.shape[1] for w in windows)
    if min_len * n_traj < 20 * n_states:
        warnings.append(
            f"thin occupation statistics (shortest window {min_len} samples x "
            f"{n_traj} trajectories for {n_states} states): pi_hat is noisy"
        )
    return HSEstimate(
        mean_y=mean_y,
        mean_y_ci_low=y_lo,
        mean_y_ci_high=y_hi,
        ift_estimate=ift,
        ift_ci_low=ift_lo,
        ift_ci_high=ift_hi,
        usable=usable,
        n_trajectories=n_traj,
        warnings=warnings,
    )
