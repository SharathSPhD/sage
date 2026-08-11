"""Bayesian inference + EFE experiment selection (ADR-0012, plan-v2 R6).

Three layers, smallest that carries the programme's needs:

1. **Grid posterior over λ** from multinomial choice counts — exact on a
   dense log-grid (trapezoid-normalised), with credible intervals. The
   scale-fold identity σ(λ, s·u) = σ(sλ, u) (F-0006) means (λ, scale) is
   never identified from choices — only their product. The posterior API
   is λ-only by design, and the fold shows up as an exact reparameterised
   equality (tested), not a hidden ridge.

2. **Marginal likelihood / Bayes factors** — model comparison over game
   models with λ integrated out; the R1 misspecification diagnostic
   (estimator disagreement) re-expressed as an evidence drop.

3. **EFE probe selection** — the active-inference experiment chooser
   (the ActiveCircuitDiscovery pattern, TRIZ session 2026-08-12):
   competing *quantitative* hypotheses each predict every candidate
   probe's outcome; the next probe maximises expected information gain
   about which hypothesis is true (pure epistemic value — BALD mutual
   information between hypothesis and predicted outcome under a Gaussian
   observation model); beliefs update by Bayes; the campaign stops on
   posterior concentration or budget. Used by
   ``experiments/efe_mechanism_campaign.py`` to resolve F-0012's open
   mechanism with the library's own exact solver as the evidence source.

References
----------
Friston et al. 2017 (expected free energy); Houlsby et al. 2011 (BALD);
Kass–Raftery 1995 (Bayes factors). Tier: derived.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np
from jax import Array

from strataq.core.solve.fixedpoint import logit_qre
from strataq.finite.games.tensor import DenseTensorGame

__all__ = [
    "CampaignResult",
    "CampaignStep",
    "Hypothesis",
    "Posterior",
    "bayes_factor",
    "efe_scores",
    "grid_posterior",
    "log_evidence",
    "log_evidence_mixture",
    "precompute_sigmas",
    "refined_posterior",
    "run_campaign",
    "update_beliefs",
]


def _logsumexp(a: np.ndarray, axis: int | None = None) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis) if axis is not None else np.asarray(out).reshape(())


# --------------------------------------------------------------------------
# 1. Grid posterior over λ
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Posterior:
    """A normalised discrete posterior on a λ grid."""

    grid: np.ndarray
    weights: np.ndarray  # sums to 1

    @property
    def mean(self) -> float:
        return float(np.sum(self.grid * self.weights))

    @property
    def map(self) -> float:
        return float(self.grid[int(np.argmax(self.weights))])

    @property
    def grid_resolved(self) -> bool:
        """False when the grid is too coarse to QUOTE an interval: fewer than
        ~6 effective grid points (participation ratio 1/Σw²) quantises the
        central interval to a couple of grid steps, which measurably
        undercovers (calibration run: 78% at PR ≈ 3 vs ~95% resolved).
        Refine before quoting — :func:`refined_posterior` does it for you."""
        return 1.0 / float(np.sum(self.weights**2)) >= 6.0

    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Central interval from the cumulative weights."""
        cdf = np.cumsum(self.weights)
        tail = (1.0 - level) / 2.0
        lo = float(self.grid[int(np.searchsorted(cdf, tail))])
        hi = float(self.grid[min(int(np.searchsorted(cdf, 1.0 - tail)), len(self.grid) - 1)])
        return lo, hi


def precompute_sigmas(
    game: DenseTensorGame, grid: Sequence[float] | np.ndarray
) -> list[tuple[Array, ...]]:
    """One QRE solve per grid point, reusable across many datasets (the
    expensive part of every posterior; principle 10, preliminary action)."""
    return [logit_qre(game, float(lam)).sigma for lam in np.asarray(grid, dtype=float)]


def _log_likelihoods(
    game: DenseTensorGame,
    counts: tuple[Array, ...],
    grid: np.ndarray,
    sigmas: list[tuple[Array, ...]] | None = None,
) -> np.ndarray:
    if sigmas is None:
        sigmas = precompute_sigmas(game, grid)
    lls = np.empty(len(grid))
    for i, sigma in enumerate(sigmas):
        ll = 0.0
        for c, s in zip(counts, sigma, strict=True):
            ll += float(jnp.sum(c * jnp.log(jnp.maximum(s, 1e-300))))
        lls[i] = ll
    return lls


def grid_posterior(
    game: DenseTensorGame,
    counts: tuple[Array, ...],
    grid: Sequence[float] | np.ndarray,
    *,
    log_prior: np.ndarray | None = None,
    sigmas: list[tuple[Array, ...]] | None = None,
) -> Posterior:
    """Posterior over λ on a fixed grid (default: uniform-on-grid prior).

    The grid prior is deliberate: it makes the scale-fold reparameterisation
    exact (posterior under payoffs s·u on grid g equals posterior under u on
    grid s·g, weight for weight) and keeps the object seed-reproducible.
    Pass ``sigmas`` from :func:`precompute_sigmas` to amortise the solves
    across many datasets on the same (game, grid).
    """
    g = np.asarray(grid, dtype=float)
    logs = _log_likelihoods(game, counts, g, sigmas)
    if log_prior is not None:
        logs = logs + log_prior
    w = np.exp(logs - _logsumexp(logs))
    return Posterior(grid=g, weights=w / np.sum(w))


def refined_posterior(
    game: DenseTensorGame,
    counts: tuple[Array, ...],
    grid: Sequence[float] | np.ndarray,
    *,
    max_rounds: int = 3,
    zoom: float = 0.15,
    points: int | None = None,
) -> Posterior:
    """grid_posterior that FOLLOWS the resolution guard's prescription:
    while ``grid_resolved`` is False, rebuild the grid zoomed around the
    MAP (±``zoom`` in log-λ) and recompute — the interval is only quoted
    from a resolved grid. Coverage is calibrated for THIS entry point;
    ``grid_posterior`` alone can be resolution-limited at sharp likelihoods
    (measured: 78% raw coverage at an informative λ* vs ~95% refined).
    """
    g = np.asarray(grid, dtype=float)
    n_pts = points or max(len(g), 400)
    post = grid_posterior(game, counts, g)
    z = zoom
    for _ in range(max_rounds):
        if post.grid_resolved:
            break
        center = post.map
        g = np.geomspace(center * np.exp(-z), center * np.exp(z), n_pts)
        post = grid_posterior(game, counts, g)
        z /= 4.0  # each round tightens the window so resolution actually grows
    return post


def log_evidence(
    game: DenseTensorGame,
    counts: tuple[Array, ...],
    grid: Sequence[float] | np.ndarray,
) -> float:
    """log p(counts | game) with λ marginalised over the uniform grid prior."""
    g = np.asarray(grid, dtype=float)
    logs = _log_likelihoods(game, counts, g)
    return float(_logsumexp(logs) - np.log(len(g)))


def log_evidence_mixture(
    game: DenseTensorGame,
    counts: tuple[Array, ...],
    grid: Sequence[float] | np.ndarray,
) -> float:
    """log p(counts | equal two-λ mixture), marginalised over grid pairs.

    The R1 misspecification diagnostic as a model: choices come from
    ½σ*(λ₁) + ½σ*(λ₂), (λ₁, λ₂) uniform on the grid square. One solve per
    grid point; the pair sweep is then pure vector arithmetic.
    """
    g = np.asarray(grid, dtype=float)
    sigmas = [logit_qre(game, float(lam)).sigma for lam in g]
    per_player = []
    for p in range(game.n_players):
        s = np.stack([np.asarray(sig[p]) for sig in sigmas])  # (G, m)
        mix = 0.5 * s[:, None, :] + 0.5 * s[None, :, :]  # (G, G, m)
        c = np.asarray(counts[p], dtype=float)
        per_player.append(np.tensordot(np.log(np.maximum(mix, 1e-300)), c, axes=([2], [0])))
    total = np.sum(per_player, axis=0)  # (G, G) log-likelihood per pair
    return float(_logsumexp(total) - 2.0 * np.log(len(g)))


def bayes_factor(log_evidence_a: float, log_evidence_b: float) -> float:
    """Evidence ratio p(D|A)/p(D|B); > 100 is decisive (Kass–Raftery)."""
    return float(np.exp(np.clip(log_evidence_a - log_evidence_b, -700.0, 700.0)))


# --------------------------------------------------------------------------
# 3. EFE probe selection (generic; probes and predictions live in log or
#    linear space as the caller chooses — sigma is in the same units)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Hypothesis:
    """A named quantitative mechanism: predicts every probe's outcome."""

    name: str
    predict: Callable[[object], float]


@dataclass(frozen=True)
class CampaignStep:
    probe: object
    efe: float
    predictions: np.ndarray
    observed: float
    beliefs: np.ndarray


@dataclass(frozen=True)
class CampaignResult:
    winner: str
    beliefs: np.ndarray
    hypothesis_names: list[str] = field(default_factory=list)
    history: list[CampaignStep] = field(default_factory=list)
    stopped_early: bool = False


def _gauss_mixture_entropy(mus: np.ndarray, weights: np.ndarray, sigma: float) -> float:
    """Entropy of Σ_h w_h N(μ_h, σ²) by dense quadrature (1-D, exact enough)."""
    span = 6.0 * sigma + (np.max(mus) - np.min(mus))
    xs = np.linspace(np.min(mus) - 0.5 * span, np.max(mus) + 0.5 * span, 4001)
    log_comps = (
        -0.5 * ((xs[None, :] - mus[:, None]) / sigma) ** 2
        - 0.5 * np.log(2.0 * np.pi * sigma**2)
        + np.log(np.maximum(weights, 1e-300))[:, None]
    )
    log_mix = _logsumexp(log_comps, axis=0)
    p = np.exp(log_mix)
    dx = xs[1] - xs[0]
    return float(-np.sum(p * log_mix) * dx)


def efe_scores(
    hypotheses: Sequence[Hypothesis],
    beliefs: np.ndarray,
    probes: Sequence[object],
    sigma: float,
) -> np.ndarray:
    """Expected information gain of each probe (BALD mutual information).

    I(H; y_x) = H[Σ_h b_h N(μ_h(x), σ²)] − Σ_h b_h H[N(μ_h(x), σ²)]:
    zero where all live hypotheses predict alike, maximal where belief-heavy
    hypotheses disagree by ≫ σ. Pure epistemic value — the campaign has no
    pragmatic preferences, resolving the hypothesis IS the goal.
    """
    h_component = 0.5 * np.log(2.0 * np.pi * np.e * sigma**2)
    scores = np.empty(len(probes))
    for j, x in enumerate(probes):
        mus = np.array([h.predict(x) for h in hypotheses])
        scores[j] = _gauss_mixture_entropy(mus, beliefs, sigma) - h_component
    return np.maximum(scores, 0.0)


def update_beliefs(
    hypotheses: Sequence[Hypothesis],
    beliefs: np.ndarray,
    *,
    probe: object,
    observed: float,
    sigma: float,
) -> np.ndarray:
    """Bayes update under the Gaussian observation model."""
    mus = np.array([h.predict(probe) for h in hypotheses])
    log_post = np.log(np.maximum(beliefs, 1e-300)) - 0.5 * ((observed - mus) / sigma) ** 2
    w = np.exp(log_post - _logsumexp(log_post))
    return np.asarray(w / np.sum(w))


def run_campaign(
    hypotheses: Sequence[Hypothesis],
    probes: Sequence[object],
    *,
    run_probe: Callable[[object], float],
    sigma: float,
    budget: int,
    stop_confidence: float = 0.95,
    prior: np.ndarray | None = None,
) -> CampaignResult:
    """Greedy EFE loop: score → run best un-run probe → update → repeat.

    Stops when max belief ≥ ``stop_confidence`` or the budget is spent.
    The full audit trail (probe chosen, its EFE, every hypothesis's
    prediction, the observation, the posterior) is returned — campaigns
    are artifacts, not black boxes.
    """
    beliefs = np.full(len(hypotheses), 1.0 / len(hypotheses)) if prior is None else prior.copy()
    remaining = list(probes)
    history: list[CampaignStep] = []
    stopped = False
    for _ in range(budget):
        if not remaining:
            break
        scores = efe_scores(hypotheses, beliefs, remaining, sigma)
        j = int(np.argmax(scores))
        x = remaining.pop(j)
        y = run_probe(x)
        beliefs = update_beliefs(hypotheses, beliefs, probe=x, observed=y, sigma=sigma)
        history.append(
            CampaignStep(
                probe=x,
                efe=float(scores[j]),
                predictions=np.array([h.predict(x) for h in hypotheses]),
                observed=y,
                beliefs=beliefs.copy(),
            )
        )
        if float(np.max(beliefs)) >= stop_confidence:
            stopped = True
            break
    return CampaignResult(
        winner=hypotheses[int(np.argmax(beliefs))].name,
        beliefs=beliefs,
        hypothesis_names=[h.name for h in hypotheses],
        history=history,
        stopped_early=stopped,
    )
