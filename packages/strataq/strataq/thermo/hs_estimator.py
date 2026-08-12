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

__all__ = [
    "SE_METHODS",
    "HSEstimate",
    "RelaxationGate",
    "hs_y_estimate",
    "relaxation_gate",
    "sample_quench_states",
]

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
    boot_se: float  # bootstrap SE of mean_y — the baseline an interval's
    # half-width should be compared against; a much wider interval buys
    # coverage with uselessness (R8 red-team objection 2)
    usable: bool  # the IFT diagnostic passed; only then quote mean_y
    n_trajectories: int
    warnings: list[str] = field(default_factory=list)


SE_METHODS = ("split", "jackknife", "delta", "bootstrap")


def _tau_from_stats(
    match_sum: float, n_pairs: float, counts: np.ndarray, lag: int, dt: float
) -> float:
    """tau_hat from the three sufficient statistics of the lag autocorrelation.

    Every SE candidate reduces to calls of this function on subsetted or
    resampled statistics, which is why the point estimate is identical across
    methods by construction rather than by test (the test exists anyway).
    """
    pi = counts / max(float(counts.sum()), 1e-300)
    base = float(np.sum(pi**2))
    rho = (match_sum / max(n_pairs, 1e-300) - base) / max(1.0 - base, 1e-12)
    rho = min(max(rho, 1e-12), 1.0 - 1e-12)
    return float(-lag * dt / np.log(rho))


def _window_stats(
    window: np.ndarray, n_states: int, lag: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """PER-TRAJECTORY sufficient statistics: (matches, state counts, pairs/traj).

    This decomposition is what makes an exactly order-invariant SE affordable.
    tau_hat depends on the window only through pooled matched pairs and pooled
    state occupancy, so leave-one-out becomes SUBTRACTION (O(nT) total) rather
    than n recomputations (O(n^2 T)), and resampling becomes a gather over n
    rows rather than over n x T states.
    """
    n_traj, n_obs = window.shape
    matches = (window[:, :-lag] == window[:, lag:]).sum(axis=1).astype(float)
    flat = (window + n_states * np.arange(n_traj, dtype=window.dtype)[:, None]).reshape(-1)
    counts = np.bincount(flat, minlength=n_traj * n_states).reshape(n_traj, n_states).astype(float)
    return matches, counts, float(n_obs - lag)


def _relaxation_time(
    window: np.ndarray,
    n_states: int,
    dt: float,
    se_method: str = "split",
    bootstrap_resamples: int = 200,
) -> tuple[float, float]:
    """Data-side relaxation-time estimate (value, SE) from autocorrelation.

    Categorical autocorrelation at lag N/4 (long lags see the slow modes a
    lag-1 estimate under-weights), fitted as a single exponential. The gate
    uses tau_hat + 2 SE so a noisy estimate cannot FLICKER a marginal hold
    into the usable set (measured flicker: usable 4/20 at a boundary hold
    before this).

    ``se_method`` selects how that SE is obtained (R9, criteria G1-G5 in
    config/experiments/gate_se.yaml). The point estimate is IDENTICAL for
    every method; only the error bar changes:

    ``split``
        The incumbent 4-way ``i::4`` trajectory split. Kept for comparison
        and reproducibility of pre-R9 reads, but it is ORDER-DEPENDENT: a
        permutation of the trajectories — which cannot change any physical
        property — reshuffles the split groups and moves the SE. R8/F-0019
        measured the consequence (anomaly-flag flips 6/20 at n=30, collapsing
        to 0/20 under a permutation that preserves the split's composition).
    ``jackknife``
        Leave-one-out over trajectories, in closed form via the sufficient
        statistics. EXACTLY order-invariant (the set of n replicates is
        permutation-invariant) and it recomputes pi_hat per replicate, so it
        carries the stationary-distribution noise ``delta`` drops.
    ``delta``
        Per-trajectory match-rate variance propagated through dtau/drho.
        Cheapest and also exactly order-invariant, but it treats pi_hat as
        fixed; agreement with ``jackknife`` is the evidence that the dropped
        term is immaterial.
    ``bootstrap``
        Trajectory resampling. Invariant only IN DISTRIBUTION — with a fixed
        resampling seed, permuting trajectories changes which ones land in
        each resample, leaving a residual of order SE/sqrt(2B); measured at
        0/20 flips anyway. **RECOMMENDED** (see below).

    **Recommendation: use ``bootstrap``; the default is ``split`` only for
    reproducibility.** On fast-mixing windows the lag-N/4 autocorrelation has
    already decayed into noise, so rho sits at or below zero and tau_hat
    returns a clip-floor artifact rather than a relaxation time (F-0021).
    Every SE that depends on local sensitivity to rho fails there, in ways
    that look method-specific but share one cause: ``delta`` explodes (its
    gradient divides by rho), ``jackknife`` collapses to EXACTLY zero SE on
    6/20 seeds at n=30 (all leave-one-out replicates pin to the same floor,
    so the gate is told tau_hat is known perfectly), ``split`` collapses on
    2/20. ``bootstrap`` collapses on 0/20 because with-replacement resampling
    perturbs far enough to escape the flat region, and it is also the most
    ACCURATE against an independently measured oracle SE (0.18-0.29 relative
    deviation vs split's 0.44-0.49). ``split`` remains the default so that
    every previously recorded verdict reproduces; changing it is R10's job,
    together with re-running the affected artifacts.
    """
    if se_method not in SE_METHODS:
        raise ValueError(f"se_method must be one of {SE_METHODS}, got {se_method!r}")
    # lag N/4 sees slow modes far better than lag 1, but still UNDERESTIMATES
    # them (~25% measured on a gap-collapsed window: 5.2 vs true 6.6) — the
    # default relax_safety of 4 exists to absorb exactly that bias
    lag = max(1, window.shape[1] // 4)
    n_traj = window.shape[0]
    matches, counts, pairs = _window_stats(window, n_states, lag)
    total_m, total_c = float(matches.sum()), counts.sum(axis=0)
    est = _tau_from_stats(total_m, n_traj * pairs, total_c, lag, dt)

    # too few trajectories to estimate a spread at all: maximally cautious,
    # so the gate demands ~3x the hold and refuses rather than guessing
    min_n = 8 if se_method == "split" else 4
    if n_traj < min_n:
        return est, est

    if se_method == "split":
        groups = [
            _tau_from_stats(
                float(matches[i::4].sum()),
                matches[i::4].size * pairs,
                counts[i::4].sum(axis=0),
                lag,
                dt,
            )
            for i in range(4)
        ]
        return est, float(np.std(groups, ddof=1)) / 2.0

    if se_method == "jackknife":
        loo = np.asarray(
            [
                _tau_from_stats(
                    total_m - float(matches[j]),
                    (n_traj - 1) * pairs,
                    total_c - counts[j],
                    lag,
                    dt,
                )
                for j in range(n_traj)
            ]
        )
        var = (n_traj - 1) / n_traj * float(np.sum((loo - loo.mean()) ** 2))
        return est, float(np.sqrt(max(var, 0.0)))

    if se_method == "delta":
        # tau = -lag*dt/ln(rho), rho = (m - base)/(1 - base):
        #   dtau/dm = lag*dt / (rho ln^2 rho) * 1/(1 - base)
        pi = total_c / max(float(total_c.sum()), 1e-300)
        base = float(np.sum(pi**2))
        rho = (total_m / (n_traj * pairs) - base) / max(1.0 - base, 1e-12)
        rho = min(max(rho, 1e-12), 1.0 - 1e-12)
        se_m = float(np.std(matches / pairs, ddof=1)) / np.sqrt(n_traj)
        grad = lag * dt / (rho * np.log(rho) ** 2) / max(1.0 - base, 1e-12)
        return est, float(abs(grad) * se_m)

    # bootstrap: resample trajectory indices, recompute from the statistics
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n_traj, size=(bootstrap_resamples, n_traj))
    reps = [
        _tau_from_stats(float(matches[row].sum()), n_traj * pairs, counts[row].sum(axis=0), lag, dt)
        for row in idx
    ]
    return est, float(np.std(reps, ddof=1))


@dataclass(frozen=True)
class RelaxationGate:
    """Did every hold window actually settle? (R9 — the gate, standalone.)

    Extracted from ``hs_y_estimate`` so the settling question can be asked
    on its own — it is the precondition for any plug-in stationary quantity,
    not just Hatano-Sasa Y — and so its SE machinery is testable in isolation.
    """

    ok: bool
    tau_hats: tuple[float, ...]
    ses: tuple[float, ...]
    thresholds: tuple[float, ...]  # tau_hat + se_sigma x SE, what the gate compares
    offenders: tuple[int, ...]
    se_method: str
    warnings: list[str] = field(default_factory=list)


def relaxation_gate(
    windows: list[np.ndarray],
    *,
    n_states: int,
    hold_durations: list[float],
    relax_safety: float = 4.0,
    se_method: str = "split",
    bootstrap_resamples: int = 200,
    se_sigma: float = 2.0,
) -> RelaxationGate:
    """Per-window settling check: every hold must exceed ``relax_safety`` x
    its own noise-inflated relaxation-time estimate.

    ``se_sigma`` (default 2.0, the historical hard-coded value) is the number
    of SEs added before comparison. It is exposed rather than buried because
    once the SE is accurate (R9) the right multiplier becomes a separate,
    answerable question — but changing it is NOT part of R9 and the default
    preserves every previously recorded verdict.
    """
    reads = [
        _relaxation_time(w, n_states, float(tau) / w.shape[1], se_method, bootstrap_resamples)
        for w, tau in zip(windows, hold_durations, strict=True)
    ]
    tau_hats = tuple(t for t, _ in reads)
    ses = tuple(s for _, s in reads)
    thresholds = tuple(t + se_sigma * s for t, s in reads)
    offenders = tuple(
        k
        for k, (tau, thr) in enumerate(zip(hold_durations, thresholds, strict=True))
        if float(tau) < relax_safety * thr
    )
    warns: list[str] = []
    if offenders:
        worst = max(offenders, key=lambda k: thresholds[k] / float(hold_durations[k]))
        warns.append(
            f"relaxation gate failed for {len(offenders)} window(s) (worst: window "
            f"{worst}, hold {float(hold_durations[worst]):g} < {relax_safety} x "
            f"estimated relaxation time {thresholds[worst]:.2f}) — the hold never "
            "settled; pi_hat is biased there and mean_y must not be quoted"
        )
    return RelaxationGate(
        ok=not offenders,
        tau_hats=tau_hats,
        ses=ses,
        thresholds=thresholds,
        offenders=offenders,
        se_method=se_method,
        warnings=warns,
    )


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
    interval_method: str = "percentile",
    relax_se_method: str = "split",
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

    ``interval_method`` selects the CI construction: ``percentile`` (the
    certified incumbent), ``bootstrap_t`` (studentised — the principled
    small-n route, since the bootstrap itself estimates the pivot's tail
    shape), or ``t_widened``. **``t_widened`` is an admitted HEURISTIC**: it
    scales a nonparametric percentile half-width by the parametric ratio
    t_{n-1}/z, mixing frameworks without theoretical justification, and it
    is retained only as an empirical comparator (R8 red-team objection 4).
    Its validity is whatever the registered coverage evidence says. Compare
    any interval's half-width against ``boot_se``: coverage bought by
    unbounded width is not usable precision.

    ``relax_se_method`` selects how the relaxation gate's SE is estimated
    (see ``relaxation_gate``). The default ``split`` is the incumbent 4-way
    trajectory split, kept as the default so no previously recorded verdict
    moves silently; ``jackknife`` and ``delta`` are exactly order-invariant.

    S3 note (R8 red-team objection 3): the flag-stability criterion permutes
    trajectory ORDER, which is null for the physics but NOT for the default
    SE machinery — the ``split`` SE is computed from ``window[i::4]``, so a
    permutation reassigns split groups. A flip therefore indicts the JOINT
    system (physics + SE estimation), not physical nullity alone; R8 measured
    that the split dominates, which is why the alternatives exist (R9).

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
    if interval_method not in ("percentile", "bootstrap_t", "t_widened"):
        raise ValueError(
            f"interval_method must be percentile | bootstrap_t | t_widened, got {interval_method!r}"
        )
    y_lo, y_hi = (float(q) for q in np.quantile(boot_y, [tail, 1.0 - tail]))
    ift_lo, ift_hi = (float(q) for q in np.quantile(boot_ift, [tail, 1.0 - tail]))
    if interval_method != "percentile":
        # R8 small-n corrections. bootstrap_t: studentise the resample means
        # by the resample SD, so the interval inherits the t-shape the CLT
        # loses at small n. t_widened: keep the percentile shape but scale its
        # half-width by t_{n-1}/z — the cheap correction, kept as a comparator
        # so the choice is made on registered coverage, not on preference.
        se_boot = float(np.std(boot_y, ddof=1))
        if interval_method == "bootstrap_t" and se_boot > 0:
            piv = (boot_y - mean_y) / se_boot
            q_lo, q_hi = (float(q) for q in np.quantile(piv, [tail, 1.0 - tail]))
            y_lo, y_hi = mean_y - q_hi * se_boot, mean_y - q_lo * se_boot
        elif interval_method == "t_widened":
            # Student-t / normal quantile ratio at n-1 df, computed from the
            # t density by quadrature (no scipy dependency in the library)
            dfree = max(n_traj - 1, 1)
            grid = np.linspace(-12.0, 12.0, 240001)
            dens = (1.0 + grid**2 / dfree) ** (-(dfree + 1) / 2.0)
            cdf = np.cumsum(dens)
            cdf /= cdf[-1]
            t_q = float(np.interp(1.0 - tail, cdf, grid))
            z_q = 1.959963984540054 if ci_level == 0.95 else t_q
            scale = max(t_q / z_q, 1.0)
            centre = 0.5 * (y_lo + y_hi)
            half = 0.5 * (y_hi - y_lo) * scale
            y_lo, y_hi = centre - half, centre + half
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
        gate = relaxation_gate(
            windows,
            n_states=n_states,
            hold_durations=hold_durations,
            relax_safety=relax_safety,
            se_method=relax_se_method,
        )
        relax_ok = gate.ok
        warnings.extend(gate.warnings)
    usable = relax_ok and ift_ok
    if interval_method == "t_widened":
        warnings.append(
            "interval_method='t_widened' is a heuristic t/z widening of a "
            "nonparametric percentile interval, not a justified method — it is "
            "kept as an empirical comparator only; prefer 'bootstrap_t' unless "
            "the registered coverage evidence says otherwise"
        )
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
        boot_se=float(np.std(boot_y, ddof=1)),
        usable=usable,
        n_trajectories=n_traj,
        warnings=warnings,
    )
