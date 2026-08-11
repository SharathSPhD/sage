"""strataq.toolkit — the instruments on YOUR data, plain arrays in, verdicts out.

The rest of the library speaks JAX and DenseTensorGame; this module speaks
numpy and lists, and every result carries its own honesty warnings. Three
questions it answers in one call each:

>>> import strataq.toolkit as tk

**"How payoff-sensitive are my agents?"** — from a payoff matrix per player
and observed choice counts::

    est = tk.estimate_rationality([u1, u2], counts=[c1, c2])
    est.mean, est.ci_low, est.ci_high, est.warnings

**"Is my system a landscape or a whirlpool?"** — from a cross-response
(pass-through) matrix measured any way you like::

    read = tk.reciprocity_read(chi)
    read.r, read.verdict, read.calibration

**"Is my time series irreversibly driven?"** — from a scalar series::

    verdict = tk.irreversibility_test(prices_per_week)
    verdict.detected, verdict.p_value

And the whole dashboard for a game you can write down::

    tk.game_thermo([u1, u2], lam=1.5)

Every number here is produced by the same gated, red-teamed machinery the
research findings use — nothing is reimplemented for the facade.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from strataq.core.dynamics.entropy import entropy_production_rate
from strataq.core.dynamics.markov import glauber_generator, stationary_distribution
from strataq.core.solve.fixedpoint import logit_qre
from strataq.domains.electricity import phase_embed
from strataq.estimate.bayes import refined_posterior
from strataq.finite.decompose.hodge import alpha as harmonic_fraction
from strataq.finite.games.tensor import DenseTensorGame
from strataq.finite.response.reciprocity import reciprocity_defect
from strataq.thermo.nulls import ReversibilizedNullResult, reversibilized_null_test

__all__ = [
    "GameThermoRead",
    "RationalityEstimate",
    "ReciprocityRead",
    "estimate_rationality",
    "game_thermo",
    "irreversibility_test",
    "reciprocity_read",
]

# the committed calibration bracket (units web.app / findings): real network
# reads 0, Blotto ~0.12, RPS ~0.69 at the reference λ
_CALIBRATION = {
    "road network (potential)": 0.0,
    "Colonel Blotto": 0.12,
    "rock-paper-scissors": 0.69,
}


def _require_finite(name: str, arr: np.ndarray) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"{name} contains NaN/inf — clean your data first; the instruments "
            "will not silently compute through missing values"
        )


def _as_game(payoff_matrices: list[object]) -> DenseTensorGame:
    arrays = [np.asarray(u, dtype=float) for u in payoff_matrices]
    for i, u in enumerate(arrays):
        _require_finite(f"payoff matrix for player {i}", u)
    return DenseTensorGame([jnp.asarray(u) for u in arrays])


@dataclass(frozen=True)
class RationalityEstimate:
    """Posterior over the logit rationality λ (per payoff unit)."""

    mean: float
    map: float
    ci_low: float
    ci_high: float
    grid_resolved: bool
    warnings: list[str] = field(default_factory=list)


def estimate_rationality(
    payoff_matrices: list[object],
    counts: list[object],
    *,
    lam_range: tuple[float, float] = (0.05, 20.0),
    grid_points: int = 400,
) -> RationalityEstimate:
    """Bayesian λ estimate from per-player choice counts under a known game.

    ``payoff_matrices``: one array per player, one axis per player (2-player:
    row player's matrix then column player's). ``counts``: observed choice
    tallies per player (same order as each player's own axis).
    """
    game = _as_game(payoff_matrices)
    raw_counts = [np.asarray(c, dtype=float) for c in counts]
    for i, c in enumerate(raw_counts):
        _require_finite(f"counts for player {i}", c)
        if c.ndim != 1 or len(c) != game.num_actions[i]:
            raise ValueError(
                f"counts for player {i} must be a length-{game.num_actions[i]} tally "
                f"(one entry per action), got shape {c.shape}"
            )
        if np.any(c < 0):
            raise ValueError(f"counts for player {i} contains negatives")
    count_arrays = tuple(jnp.asarray(c) for c in raw_counts)
    grid = np.geomspace(lam_range[0], lam_range[1], grid_points)
    post = refined_posterior(game, count_arrays, grid)
    lo, hi = post.credible_interval(0.95)
    warnings = [
        "lambda is per payoff unit: rescaling payoffs by s rescales lambda by 1/s "
        "(the scale fold) — compare lambdas only across identically-scaled payoffs"
    ]
    span = np.log(lam_range[1] / lam_range[0])
    if (hi - lo) / max(post.mean, 1e-12) > 0.5 * span:
        warnings.append(
            "flat likelihood: these choices barely constrain lambda on this game "
            "(symmetric games can be uniform at every lambda) — do not quote a point estimate"
        )
    if not post.grid_resolved:
        warnings.append("grid-resolution limited even after refinement; treat the CI as indicative")
    return RationalityEstimate(
        mean=post.mean,
        map=post.map,
        ci_low=lo,
        ci_high=hi,
        grid_resolved=post.grid_resolved,
        warnings=warnings,
    )


@dataclass(frozen=True)
class ReciprocityRead:
    """The reciprocity defect ℛ of a measured cross-response matrix."""

    r: float
    verdict: str
    calibration: dict[str, float]
    ci_low: float | None = None  # only when chi_se is supplied
    ci_high: float | None = None
    warnings: list[str] = field(default_factory=list)


def reciprocity_read(
    chi: np.ndarray | list[list[float]],
    *,
    chi_se: np.ndarray | list[list[float]] | None = None,
    n_draws: int = 2000,
    seed: int = 0,
) -> ReciprocityRead:
    """ℛ = ‖χ − χᵀ‖/‖χ + χᵀ‖ from any square cross-response matrix.

    χ[i, j] = response of agent i's action to a shift in agent j's
    incentives (e.g. pass-through of j's cost into i's price). ℛ = 0 means
    reciprocal (landscape-like); order-one means circulating response.

    Real χ matrices are ESTIMATED and noisy: pass ``chi_se`` (elementwise
    standard errors, e.g. from your regression) to get a Monte-Carlo 95%
    interval on ℛ and an uncertainty-aware verdict. Without it the read is
    a point value, the verdict says so, and near-threshold values are
    labelled borderline rather than classified.
    """
    m = np.asarray(chi, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("chi must be a square matrix")
    _require_finite("chi", m)

    def _r(x: np.ndarray) -> float:
        return float(np.linalg.norm(x - x.T)) / max(float(np.linalg.norm(x + x.T)), 1e-300)

    r = _r(m)
    ci_low = ci_high = None
    warnings = [
        "only the ZERO test is lambda-free; the magnitude of R scales with the "
        "agents' payoff sensitivity, so compare magnitudes only at matched conditions",
        "measure chi from SMALL incentive shifts (it is a local derivative)",
    ]
    if chi_se is not None:
        se = np.asarray(chi_se, dtype=float)
        if se.shape != m.shape or np.any(se < 0):
            raise ValueError("chi_se must be nonnegative and match chi's shape")
        _require_finite("chi_se", se)
        rng = np.random.default_rng(seed)
        draws = np.array([_r(m + se * rng.standard_normal(m.shape)) for _ in range(n_draws)])
        ci_low, ci_high = (float(q) for q in np.quantile(draws, [0.025, 0.975]))

    def _band(value: float) -> str:
        if value < 0.02:
            return "reciprocal (landscape-like)"
        if value < 0.3:
            return "mildly non-reciprocal (mixed structure)"
        return "strongly non-reciprocal (whirlpool-like)"

    if ci_low is not None and ci_high is not None:
        lo_band, hi_band = _band(ci_low), _band(ci_high)
        if lo_band == hi_band:
            verdict = f"{lo_band} — 95% CI [{ci_low:.4g}, {ci_high:.4g}]"
        else:
            verdict = (
                f"uncertain between '{lo_band}' and '{hi_band}' at this noise level "
                f"(95% CI [{ci_low:.4g}, {ci_high:.4g}]) — more or cleaner response data needed"
            )
    else:
        near = min(abs(r - 0.02), abs(r - 0.3))
        if near < 0.5 * max(r, 0.02):
            verdict = (
                f"borderline point read (R = {r:.4g} near a verdict threshold) — supply "
                "chi_se for an uncertainty-aware verdict"
            )
        else:
            verdict = f"{_band(r)} — point read; supply chi_se for a defensible interval"
        warnings.append(
            "no chi_se supplied: this is a point read of a noisy estimate — do not "
            "publish a classification from it without an uncertainty interval"
        )
    return ReciprocityRead(
        r=r,
        verdict=verdict,
        calibration=dict(_CALIBRATION),
        ci_low=ci_low,
        ci_high=ci_high,
        warnings=warnings,
    )


def irreversibility_test(
    series: list[float] | np.ndarray,
    *,
    n_bins: int = 3,
    n_surrogates: int = 200,
    alpha_level: float = 0.01,
    seed: int = 0,
) -> ReversibilizedNullResult:
    """Is a scalar time series irreversibly driven? (the F-0009 instrument).

    Phase-embeds the series as (value bin, direction of last change) — plain
    value binning is provably blind to loop irreversibility — then tests the
    KLD irreversibility against the reversibilized-Markov null (detailed-
    balance-exact, persistence-matched). ``detected=True`` at level
    ``alpha_level`` means no reversible chain with the same pair statistics
    produces this reading. Power: measured ≥ 80% detection on a known driven
    series at n ≥ 300; n ≈ 100 is underpowered — a non-detection there is
    weak evidence.
    """
    arr = np.asarray(series, dtype=float)
    _require_finite("series", arr)
    if arr.ndim != 1 or len(arr) < 50:
        raise ValueError("need a 1-D series with at least 50 observations (power needs n >= 300)")
    if float(np.std(arr)) == 0.0:
        raise ValueError(
            "series is constant — there is no dynamics to test; a not-detected "
            "verdict on it would be vacuous, so this raises instead"
        )
    values = [float(x) for x in arr]
    states, n_states = phase_embed(values, n_bins)
    return reversibilized_null_test(
        np.asarray(states),
        n_states,
        n_surrogates=n_surrogates,
        alpha=alpha_level,
        seed=seed,
    )


@dataclass(frozen=True)
class GameThermoRead:
    """One-call dashboard for a written-down game at rationality λ."""

    alpha: float  # harmonic fraction: 0 = pure landscape, 1 = pure whirlpool
    r: float  # reciprocity defect of the equilibrium response
    epr: float  # entropy production rate of the joint revision dynamics
    verdict: str


def game_thermo(payoff_matrices: list[object], lam: float = 1.5) -> GameThermoRead:
    """α, ℛ and dissipation for a small game given as plain payoff arrays."""
    game = _as_game(payoff_matrices)
    a = float(harmonic_fraction(game))
    r = float(reciprocity_defect(game, logit_qre(game, lam)))
    gen = glauber_generator(game, lam)
    epr = float(entropy_production_rate(gen, stationary_distribution(gen)))
    if a < 0.05:
        verdict = "landscape: relaxes to equilibrium, zero dissipation"
    elif a > 0.6:
        verdict = "whirlpool: circulates forever, dissipating"
    else:
        verdict = "mixed: partial gradient structure with a circulating component"
    return GameThermoRead(alpha=a, r=r, epr=epr, verdict=verdict)
