"""The λ-estimator family: four routes to the rationality parameter.

λ is the most abused dial in the QRE literature — "λ absorbs unmodelled
heterogeneity". The defence is redundancy: estimate it four structurally
different ways and let disagreement be a diagnostic, not an embarrassment.

1. **Frequency MLE** (`lambda_mle`) — maximise the multinomial log-likelihood
   of observed choice counts under σ*(λ). Profile-likelihood CI. Flags a flat
   likelihood (λ unidentified — e.g. any game whose QRE is uniform at every
   λ, like symmetric RPS) instead of returning a meaningless number.
2. **Autodiff MLE** (`lambda_mle_implicit`) — the same objective, but scored
   by JAX autodiff through an unrolled damped solve; agreement with the grid
   is itself a check on both.
3. **χ moment-matching** (`lambda_moment_chi`) — fit λ to an observed
   cross-response matrix (pass-through asymmetries), the route that works
   when choices are equilibrium-degenerate but responses are measurable.
4. **Dispersion inversion** (`lambda_dispersion`) — invert mean choice
   entropy H(σ*(λ)) = H_obs; assumption-light, moment-only.

All tolerances/windows come from ``config/base.yaml`` (estimate section).
Tier: statistical machinery over exact solves; nothing here is claimed
beyond consistency on well-specified synthetic data (gate-checked).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from strataq.core.defaults import base_config
from strataq.core.solve.fixedpoint import logit_qre
from strataq.finite.games.tensor import DenseTensorGame
from strataq.finite.response.susceptibility import chi_equilibrium


@dataclass(frozen=True)
class LambdaEstimate:
    """One estimator's answer, with its uncertainty and its caveats."""

    lam: float
    ci_low: float
    ci_high: float
    method: str
    objective: float
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class LambdaAgreement:
    """The four-estimator protocol: estimates, spread, and the verdict."""

    estimates: dict[str, LambdaEstimate] = field(default_factory=dict)
    agreement_gap: float = 0.0
    disagreement_flag: bool = False
    warnings: tuple[str, ...] = ()


def sample_choices(game: DenseTensorGame, lam: float, n: int, key: Array) -> tuple[Array, ...]:
    """Synthetic choice counts: n i.i.d. draws per player from σ*(λ)."""
    sigma = logit_qre(game, lam).sigma
    keys = jax.random.split(key, game.n_players)
    counts = []
    for k, s in zip(keys, sigma, strict=True):
        draws = jax.random.categorical(k, jnp.log(s), shape=(n,))
        counts.append(jnp.bincount(draws, length=s.shape[0]))
    return tuple(counts)


def _log_likelihood(game: DenseTensorGame, counts: tuple[Array, ...], lam: float) -> float:
    sigma = logit_qre(game, lam).sigma
    ll = 0.0
    for c, s in zip(counts, sigma, strict=True):
        ll += float(jnp.sum(c * jnp.log(jnp.maximum(s, 1e-300))))
    return ll


def _lam_grid() -> np.ndarray:
    cfg = base_config().estimate
    return np.geomspace(cfg.lam_min, cfg.lam_max, cfg.grid_points)


def _golden_refine(f: Callable[[float], float], lo: float, hi: float, iters: int) -> float:
    """Golden-section MAXIMISATION of f on [lo, hi] (log-λ space)."""
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = np.log(lo), np.log(hi)
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = f(np.exp(c)), f(np.exp(d))
    for _ in range(iters):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = f(np.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = f(np.exp(d))
    return float(np.exp((a + b) / 2.0))


def lambda_mle(game: DenseTensorGame, counts: tuple[Array, ...]) -> LambdaEstimate:
    """Frequency MLE with profile-likelihood CI and a flat-likelihood guard."""
    cfg = base_config().estimate
    grid = _lam_grid()
    lls = np.array([_log_likelihood(game, counts, la) for la in grid])
    warnings: list[str] = []
    n_obs = float(sum(int(jnp.sum(c)) for c in counts))
    if (lls.max() - lls.min()) < cfg.flat_ll_per_obs * max(n_obs, 1.0):
        warnings.append(
            "lambda unidentified: likelihood is flat across the search window "
            "(the QRE of this game barely moves with lambda)"
        )
    k = int(np.argmax(lls))
    lo = grid[max(k - 1, 0)]
    hi = grid[min(k + 1, len(grid) - 1)]
    lam_hat = _golden_refine(lambda la: _log_likelihood(game, counts, la), lo, hi, cfg.refine_iters)
    ll_max = _log_likelihood(game, counts, lam_hat)

    # profile-likelihood CI on the grid (conservative: nearest grid crossing)
    inside = grid[lls >= ll_max - cfg.profile_ci_drop]
    ci_low = float(inside.min()) if inside.size else float(grid[0])
    ci_high = float(inside.max()) if inside.size else float(grid[-1])
    return LambdaEstimate(
        lam=lam_hat,
        ci_low=min(ci_low, lam_hat),
        ci_high=max(ci_high, lam_hat),
        method="mle",
        objective=ll_max,
        warnings=tuple(warnings),
    )


def _unrolled_sigma(
    game: DenseTensorGame, lam: Array, n_iter: int, damping: float
) -> tuple[Array, ...]:
    """σ*(λ) by a fixed unrolled damped iteration — differentiable in λ."""
    sigma = tuple(jnp.full((m,), 1.0 / m) for m in game.num_actions)
    for _ in range(n_iter):
        new = []
        for i in range(game.n_players):
            u = game.payoffs[i]
            axes = list(range(game.n_players))
            eu = u
            # contract every opponent axis with their current mix
            for j in reversed([a for a in axes if a != i]):
                eu = jnp.tensordot(eu, sigma[j], axes=([j if j < i else 1], [0]))
            eu = eu.reshape(-1)
            new.append(sigma[i] + damping * (jax.nn.softmax(lam * eu) - sigma[i]))
        sigma = tuple(new)
    return sigma


def lambda_mle_implicit(game: DenseTensorGame, counts: tuple[Array, ...]) -> LambdaEstimate:
    """MLE scored by autodiff through an unrolled solve (Newton on the score).

    The point estimate is refined independently; the confidence interval is
    INHERITED from the grid MLE's profile likelihood (the two share an
    objective, and re-profiling through the unrolled solve buys nothing) —
    disclosed here and in the returned warnings.
    """
    cfg = base_config().estimate
    solver = base_config().solver

    def nll(log_lam: Array) -> Array:
        sigma = _unrolled_sigma(game, jnp.exp(log_lam), 400, solver.damping)
        out = jnp.asarray(0.0)
        for c, s in zip(counts, sigma, strict=True):
            out = out - jnp.sum(c * jnp.log(jnp.maximum(s, 1e-300)))
        return out

    grad = jax.grad(nll)
    start = lambda_mle(game, counts)
    x = jnp.log(jnp.asarray(start.lam))
    for _ in range(cfg.refine_iters):
        g = grad(x)
        # secant-style Newton on the scalar score
        g2 = grad(x + 1e-4)
        hess = (g2 - g) / 1e-4
        step = jnp.where(jnp.abs(hess) > 1e-12, g / hess, 0.1 * jnp.sign(g))
        x = x - jnp.clip(step, -0.5, 0.5)
    lam_hat = float(jnp.exp(x))
    return LambdaEstimate(
        lam=lam_hat,
        ci_low=start.ci_low,
        ci_high=start.ci_high,
        method="mle_implicit",
        objective=-float(nll(x)),
        warnings=(*start.warnings, "ci inherited from grid MLE profile likelihood"),
    )


def lambda_moment_chi(game: DenseTensorGame, chi_obs: Array) -> LambdaEstimate:
    """Match the observed cross-response matrix to χ^eq(λ) in Frobenius norm.

    ``chi_obs`` must be a MEASURED full susceptibility (all cross pass-through
    entries), e.g. from poke experiments or pass-through regressions; the gate
    validates recovery from the oracle χ, which bounds only the inversion
    error, not measurement noise.
    """
    cfg = base_config().estimate

    def neg_gap(la: float) -> float:
        point = logit_qre(game, la)
        chi = chi_equilibrium(game, point).chi_full
        return -float(jnp.linalg.norm(chi - chi_obs))

    grid = _lam_grid()
    gaps = np.array([neg_gap(la) for la in grid])
    k = int(np.argmax(gaps))
    lam_hat = _golden_refine(
        neg_gap, grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)], cfg.refine_iters
    )
    return LambdaEstimate(
        lam=lam_hat,
        ci_low=lam_hat,
        ci_high=lam_hat,
        method="moment_chi",
        objective=neg_gap(lam_hat),
        warnings=(),
    )


def _mean_entropy(sigma: tuple[Array, ...]) -> float:
    hs = [float(-jnp.sum(jnp.where(s > 0, s * jnp.log(s), 0.0))) for s in sigma]
    return float(np.mean(hs))


def lambda_dispersion(
    game: DenseTensorGame, counts: tuple[Array, ...], *, bootstrap: bool = True
) -> LambdaEstimate:
    """Invert mean choice entropy; bootstrap CI over multinomial resamples.

    ``bootstrap=False`` skips the CI (point estimate only, ci collapsed to
    the point) — the sync API uses this to stay inside its latency budget.
    """
    cfg = base_config().estimate
    freqs = tuple(c / jnp.sum(c) for c in counts)
    h_obs = _mean_entropy(freqs)
    warnings: list[str] = []

    def neg_gap(la: float) -> float:
        return -abs(_mean_entropy(logit_qre(game, la).sigma) - h_obs)

    grid = _lam_grid()
    model_h = np.array([_mean_entropy(logit_qre(game, la).sigma) for la in grid])
    if model_h.max() - model_h.min() < cfg.flat_entropy_threshold:
        warnings.append(
            "lambda unidentified from dispersion: model entropy is flat across the "
            "search window (e.g. a symmetric principal branch below its bifurcation)"
        )
    gaps = -np.abs(model_h - h_obs)
    k = int(np.argmax(gaps))
    lam_hat = _golden_refine(
        neg_gap, grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)], cfg.refine_iters
    )

    if not bootstrap:
        return LambdaEstimate(
            lam=lam_hat,
            ci_low=lam_hat,
            ci_high=lam_hat,
            method="dispersion",
            objective=neg_gap(lam_hat),
            warnings=tuple(warnings),
        )
    rng = np.random.default_rng(base_config().seeds.root)
    boots = []
    for _ in range(cfg.bootstrap_resamples):
        rcounts = tuple(
            jnp.asarray(rng.multinomial(int(jnp.sum(c)), np.asarray(c / jnp.sum(c))))
            for c in counts
        )
        h_b = _mean_entropy(tuple(c / jnp.sum(c) for c in rcounts))

        def neg_gap_b(la: float, h_b: float = h_b) -> float:
            return -abs(_mean_entropy(logit_qre(game, la).sigma) - h_b)

        gb = np.array([neg_gap_b(la) for la in grid])
        boots.append(float(grid[int(np.argmax(gb))]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return LambdaEstimate(
        lam=lam_hat,
        ci_low=float(min(lo, lam_hat)),
        ci_high=float(max(hi, lam_hat)),
        method="dispersion",
        objective=neg_gap(lam_hat),
        warnings=tuple(warnings),
    )


def agreement_protocol(
    game: DenseTensorGame,
    counts: tuple[Array, ...],
    chi_obs: Array | None = None,
) -> LambdaAgreement:
    """Run every applicable estimator; the spread is the diagnostic.

    Disagreement beyond the configured gap flags likely misspecification
    (heterogeneous λ, non-QRE data, wrong payoff model) — the protocol
    reports it rather than averaging it away.

    Precedence: unidentifiability warnings SUPPRESS the disagreement flag —
    if λ is not identified, estimator spread is noise, not evidence of
    misspecification. Both are always visible in ``warnings``.
    """
    cfg = base_config().estimate
    estimates: dict[str, LambdaEstimate] = {
        "mle": lambda_mle(game, counts),
        "mle_implicit": lambda_mle_implicit(game, counts),
        "dispersion": lambda_dispersion(game, counts),
    }
    if chi_obs is not None:
        estimates["moment_chi"] = lambda_moment_chi(game, chi_obs)

    warnings = [w for e in estimates.values() for w in e.warnings]
    ident_warnings = [w for w in warnings if "unidentified" in w]
    lams = np.array([e.lam for e in estimates.values()])
    gap = float((lams.max() - lams.min()) / max(lams.mean(), 1e-12))
    flagged = gap > cfg.agreement_flag_gap and not ident_warnings
    if flagged:
        warnings.append(
            "estimators disagree beyond the configured gap: data are unlikely to be "
            "a single-lambda QRE of this payoff model"
        )
    return LambdaAgreement(
        estimates=estimates,
        agreement_gap=gap,
        disagreement_flag=flagged,
        warnings=tuple(dict.fromkeys(warnings)),
    )
