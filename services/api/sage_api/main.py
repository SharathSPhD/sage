"""The strataq instruments over HTTP.

Sync endpoints for small games (size-guarded); every response carries
``provenance`` (library version, payoff normalisation, λ handling) and
``warnings`` (``near_criticality`` etc.) per services/api/CLAUDE.md. Async
queue, persistence and auth arrive with the deployment unit; this module is
the instrument surface itself, and everything here is a thin shim over
gate-closed library calls — no science in the API layer.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Annotated, Any, Literal

import jax.numpy as jnp
import strataq
import strataq.toolkit as tk
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from strataq.core.dynamics.markov import glauber_generator, profile_space
from strataq.core.dynamics.sample import sample_trajectories
from strataq.core.solve.fixedpoint import logit_qre
from strataq.core.solve.homotopy import logit_branch
from strataq.diagnose import diagnose
from strataq.estimate.lam import lambda_dispersion, lambda_mle
from strataq.finite.decompose.hodge import hodge_decompose
from strataq.finite.games.tensor import DenseTensorGame
from strataq.finite.response.reciprocity import reciprocity_defect
from strataq.finite.response.spectral import strategic_spectrum
from strataq.finite.response.susceptibility import chi_equilibrium
from strataq.fit import fit
from strataq.problems import (
    AllocationProblem,
    AuctionProblem,
    DemandModel,
    ElectricityProblem,
    LinearDemand,
    LogitDemand,
    PricingProblem,
    RoutingProblem,
)
from strataq.thermo.estimators import (
    kld_epr,
    stationary_current_weights,
    tur_epr_bound,
    tur_epr_bound_ci,
)
from strataq.thermo.exact import thermo_read


class Settings(BaseSettings):
    """Env-driven service limits (pydantic-settings; no secrets in the repo)."""

    max_actions_per_player: int = 12
    max_players: int = 3
    max_profile_states: int = 400  # dense dynamics guard
    max_sample_steps: int = 20000
    max_sample_trajectories: int = 16
    max_sample_budget: int = 120_000  # n_steps * n_trajectories cap
    max_tidy_rows: int = 200_000  # /v1/fit long-form ingestion cap
    model_config = {"env_prefix": "SAGE_API_"}


settings = Settings()
app = FastAPI(
    title="strataq API",
    version=strataq.__version__,
    description="Susceptibility, reciprocity, dissipation and phase readings for finite games.",
)

# Public, read-compute API with no credentials or user state: permissive CORS
# is safe and lets any origin (the app, notebooks, third-party pages) call it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class GamePayload(BaseModel):
    """A finite game as one payoff tensor per player (row-major nested lists)."""

    payoffs: list[Any] = Field(description="One nested-list tensor per player.")
    lam: float = Field(gt=0, le=100, description="Logit precision λ (uniform across players).")


class Provenance(BaseModel):
    library_version: str
    payoff_range: float
    lambda_normalised: float
    solver: Literal["damped"] = "damped"


def _game_from(payload: GamePayload) -> DenseTensorGame:
    try:
        game = DenseTensorGame(tuple(jnp.asarray(u, dtype=jnp.float64) for u in payload.payoffs))
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"invalid game: {exc}") from exc
    if not all(bool(jnp.all(jnp.isfinite(u))) for u in game.payoffs):
        raise HTTPException(status_code=422, detail="payoffs must be finite (no NaN/Inf)")
    if game.n_players > settings.max_players:
        raise HTTPException(
            status_code=413, detail=f"max {settings.max_players} players (sync API)"
        )
    if max(game.num_actions) > settings.max_actions_per_player:
        raise HTTPException(
            status_code=413, detail=f"max {settings.max_actions_per_player} actions per player"
        )
    return game


def _provenance(game: DenseTensorGame, lam: float) -> Provenance:
    payoff_range = float(game.payoff_range)
    return Provenance(
        library_version=strataq.__version__,
        payoff_range=payoff_range,
        lambda_normalised=lam * payoff_range,
    )


@app.get("/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok", "library": strataq.__version__}


@app.post("/v1/solve/qre")
def solve_qre(payload: GamePayload) -> dict[str, Any]:
    game = _game_from(payload)
    point = logit_qre(game, payload.lam)
    if not bool(point.converged):
        raise HTTPException(status_code=422, detail="solver did not converge at requested lambda")
    return {
        "sigma": [s.tolist() for s in point.sigma],
        "residual": float(point.residual),
        "n_iter": int(point.n_iter),
        "provenance": _provenance(game, payload.lam).model_dump(),
        "warnings": [],
    }


@app.post("/v1/decompose")
def decompose(payload: GamePayload) -> dict[str, Any]:
    game = _game_from(payload)
    dec = hodge_decompose(game)
    return {
        "alpha": float(dec.alpha),
        "potential_norm": float(dec.potential_norm),
        "harmonic_norm": float(dec.harmonic_norm),
        "provenance": _provenance(game, payload.lam).model_dump(),
        "warnings": [],
    }


@app.post("/v1/response")
def response(payload: GamePayload) -> dict[str, Any]:
    """χ^eq, ℛ, and the spectral phase read — with honesty flags attached.

    Near criticality the χ magnitudes are unreliable; the API flags rather
    than hides (and refuses nothing here: matrix-input games carry their own
    conjugate field implicitly — the payoff perturbation h itself).
    """
    game = _game_from(payload)
    point = logit_qre(game, payload.lam)
    resp = chi_equilibrium(game, point)
    spec = strategic_spectrum(game, point)
    warnings = []
    if bool(resp.near_critical):
        warnings.append("near_criticality: chi magnitudes unreliable (distance below threshold)")
    if float(spec.rho) >= 1.0:
        warnings.append("supercritical: resolvent near-singular; R is direction-only here")
    return {
        "reciprocity_defect": float(reciprocity_defect(game, point, response=resp)),
        "distance_to_criticality": float(resp.distance_to_criticality),
        "rho_sb": float(resp.rho_sb),
        "bifurcation_type": int(spec.bifurcation_type),
        "chi_eq": resp.chi_full.tolist(),
        "provenance": _provenance(game, payload.lam).model_dump(),
        "warnings": warnings,
    }


@app.post("/v1/dynamics/stationary")
def dynamics_stationary(payload: GamePayload) -> dict[str, Any]:
    game = _game_from(payload)
    n_states = 1
    for m in game.num_actions:
        n_states *= m
    if n_states > settings.max_profile_states:
        raise HTTPException(
            status_code=413,
            detail=f"profile space {n_states} exceeds dense limit "
            f"{settings.max_profile_states}; trajectory estimators land in a later unit",
        )
    reading = thermo_read(game, payload.lam)
    return {
        "epr": float(reading.epr),
        "max_current": float(reading.max_current),
        "detailed_balance": bool(reading.detailed_balance),
        "pi": reading.pi.tolist(),
        "currents": reading.currents.tolist(),
        "states": [list(s) for s in profile_space(game.num_actions)],
        "provenance": _provenance(game, payload.lam).model_dump(),
        "warnings": [],
    }


class BranchPayload(BaseModel):
    payoffs: list[Any]
    lam_max: Annotated[float, Field(gt=0, le=20)]
    n_points: Annotated[int, Field(gt=10, le=500)] = 200


@app.post("/v1/solve/branch")
def solve_branch(payload: BranchPayload) -> dict[str, Any]:
    game = _game_from(GamePayload(payoffs=payload.payoffs, lam=1.0))
    branch = logit_branch(game, payload.lam_max, n_points=payload.n_points)
    return {
        "lambdas": branch.lambdas.tolist(),
        "sigmas": branch.sigmas.tolist(),
        "rhos": branch.rhos.tolist(),
        "turning_points": branch.turning_points.tolist(),
        "provenance": _provenance(game, 1.0).model_dump(),
        "warnings": [],
    }


@app.get("/v1/examples")
def examples() -> dict[str, Any]:
    """Named example games for the app's Lab (both α anchors + the middle)."""
    from strataq.finite.games.library import (
        congestion,
        coordination,
        matching_pennies,
        rock_paper_scissors,
    )

    costs = jnp.array([[1.0, 2.5, 4.0], [1.5, 2.0, 3.5], [0.5, 3.0, 5.0]])
    named = {
        "congestion (alpha=0)": congestion(2, costs),
        "coordination (alpha=0)": coordination(2, 3, bonus=2.0),
        "matching_pennies (alpha=1)": matching_pennies(),
        "rock_paper_scissors (alpha=1)": rock_paper_scissors(),
    }
    return {name: {"payoffs": [u.tolist() for u in game.payoffs]} for name, game in named.items()}


class PokePayload(BaseModel):
    """A finite payoff nudge to one player's one action — the ℛ measurement procedure."""

    payoffs: list[Any]
    lam: Annotated[float, Field(gt=0, le=100)]
    player: Annotated[int, Field(ge=0)]
    action: Annotated[int, Field(ge=0)]
    size: Annotated[float, Field(gt=-100, lt=100)]


@app.post("/v1/response/poke")
def response_poke(payload: PokePayload) -> dict[str, Any]:
    """Nudge one player's incentives, re-equilibrate, report who moved.

    This is the operational content of the reciprocity meter (theory doc 07):
    poke player 1, read player 2; poke player 2, read player 1. The poke is a
    finite h added to the chosen action's payoff for the chosen player; both
    equilibria are solved fresh (no linearisation), so the readings are honest
    at any poke size.
    """
    game = _game_from(GamePayload(payoffs=payload.payoffs, lam=payload.lam))
    if payload.player >= game.n_players:
        raise HTTPException(status_code=422, detail="player index out of range")
    if payload.action >= game.num_actions[payload.player]:
        raise HTTPException(status_code=422, detail="action index out of range")

    base = logit_qre(game, payload.lam)
    idx: list[Any] = [slice(None)] * game.n_players
    idx[payload.player] = payload.action
    poked_payoffs = list(game.payoffs)
    poked_payoffs[payload.player] = poked_payoffs[payload.player].at[tuple(idx)].add(payload.size)
    poked_game = DenseTensorGame(tuple(poked_payoffs))
    poked = logit_qre(poked_game, payload.lam)
    if not (bool(base.converged) and bool(poked.converged)):
        raise HTTPException(status_code=422, detail="solver did not converge")
    return {
        "sigma_base": [s.tolist() for s in base.sigma],
        "sigma_poked": [s.tolist() for s in poked.sigma],
        "delta": [(sp - sb).tolist() for sp, sb in zip(poked.sigma, base.sigma, strict=True)],
        "provenance": _provenance(game, payload.lam).model_dump(),
        "warnings": [],
    }


class SamplePayload(BaseModel):
    """Trajectory sampling request for the estimator panels."""

    payoffs: list[Any]
    lam: Annotated[float, Field(gt=0, le=100)]
    n_steps: Annotated[int, Field(gt=100)]
    n_trajectories: Annotated[int, Field(gt=1)]
    seed: Annotated[int, Field(ge=0)] = 0


@app.post("/v1/dynamics/sample")
def dynamics_sample(payload: SamplePayload) -> dict[str, Any]:
    """Sample Glauber trajectories and read the irreversibility estimators.

    Returns the KLD(k=1) point estimate, the TUR point estimate AND its
    certified bootstrap-lower quantile, alongside the exact EPR of the same
    generator — the app shows convergence of data-side meters to the exact
    one. Seeded and deterministic.
    """
    import jax

    game = _game_from(GamePayload(payoffs=payload.payoffs, lam=payload.lam))
    n_states = 1
    for m in game.num_actions:
        n_states *= m
    if n_states > settings.max_profile_states:
        raise HTTPException(status_code=413, detail="profile space exceeds dense limit")
    if (
        payload.n_steps > settings.max_sample_steps
        or payload.n_trajectories > settings.max_sample_trajectories
        or payload.n_steps * payload.n_trajectories > settings.max_sample_budget
    ):
        raise HTTPException(status_code=413, detail="sampling budget exceeds service limit")

    gen = glauber_generator(game, payload.lam)
    key, boot_key = jax.random.split(jax.random.key(payload.seed))
    batch = sample_trajectories(
        gen, key, n_steps=payload.n_steps, n_trajectories=payload.n_trajectories
    )
    weights = stationary_current_weights(gen)
    reading = thermo_read(game, payload.lam)
    return {
        "exact_epr": float(reading.epr),
        "kld_epr": float(kld_epr(batch, k=1)),
        "tur_point": float(tur_epr_bound(batch, weights)),
        "tur_ci_low": float(tur_epr_bound_ci(batch, weights, boot_key)),
        "n_steps": payload.n_steps,
        "n_trajectories": payload.n_trajectories,
        "seed": payload.seed,
        "provenance": _provenance(game, payload.lam).model_dump(),
        "warnings": [
            "tur_point is a point estimate and may exceed exact EPR near equilibrium; "
            "tur_ci_low is the certified statement"
        ],
    }


class EstimatePayload(BaseModel):
    """Observed choice counts for the λ-estimator family."""

    payoffs: list[Any]
    counts: list[list[int]] = Field(description="Per-player observed action counts.")


@app.post("/v1/estimate/lambda")
def estimate_lambda(payload: EstimatePayload) -> dict[str, Any]:
    """Run the fast members of the λ-estimator family on observed counts.

    Sync-budget subset: frequency MLE (profile-likelihood CI) and dispersion
    inversion (point estimate, no bootstrap). Unidentifiability warnings pass
    through — the API never returns a bare number where λ is not identified.
    """
    game = _game_from(GamePayload(payoffs=payload.payoffs, lam=1.0))
    if len(payload.counts) != game.n_players:
        raise HTTPException(status_code=422, detail="one count vector per player required")
    counts = []
    for c, m in zip(payload.counts, game.num_actions, strict=True):
        if len(c) != m or any(x < 0 for x in c) or sum(c) == 0:
            raise HTTPException(
                status_code=422, detail="counts must be nonnegative, per-action, nonempty"
            )
        counts.append(jnp.asarray(c))
    total = sum(int(jnp.sum(c)) for c in counts)
    if total > 10_000_000:
        raise HTTPException(status_code=413, detail="count total exceeds service limit")

    mle = lambda_mle(game, tuple(counts))
    disp = lambda_dispersion(game, tuple(counts), bootstrap=False)
    lams = [mle.lam, disp.lam]
    gap = (max(lams) - min(lams)) / max(sum(lams) / len(lams), 1e-12)
    return {
        "estimates": {
            e.method: {
                "lam": e.lam,
                "ci_low": e.ci_low,
                "ci_high": e.ci_high,
                "warnings": list(e.warnings),
            }
            for e in (mle, disp)
        },
        "agreement_gap": gap,
        "provenance": _provenance(game, mle.lam if mle.lam > 0 else 1.0).model_dump(),
        "warnings": list(dict.fromkeys([*mle.warnings, *disp.warnings])),
    }


# ---------------------------------------------------------------------------
# Sioux Falls domain lab — real TNTP data through the population engine.
# The RoutingNetwork is built lazily on first request (TNTP fetch is cached on
# disk) and memoised for the process lifetime; route sets are the k shortest
# per OD, a documented restriction of the congestion plugin.
_SIOUX: dict[str, Any] = {}


def _sioux() -> dict[str, Any]:
    if not _SIOUX:
        from strataq.domains.congestion import load_sioux_falls, routing_network_from_tntp

        tntp = load_sioux_falls()
        # explicit lambda, not dict.get: mypy cannot narrow the overloaded
        # bound method to the Callable sorted() wants
        od_pairs = sorted(tntp.demand, key=lambda od: tntp.demand[od], reverse=True)[:12]
        network = routing_network_from_tntp(tntp, od_pairs, k_routes=3)
        _SIOUX.update(
            tntp=tntp,
            network=network,
            od_pairs=od_pairs,
            links=[
                {
                    "from": int(a),
                    "to": int(b),
                    "free_flow": float(f),
                    "capacity": float(c),
                }
                for a, b, f, c in zip(
                    tntp.init_node, tntp.term_node, tntp.free_flow_time, tntp.capacity, strict=True
                )
            ],
        )
    return _SIOUX


@app.get("/v1/domains/sioux_falls/network")
def sioux_network() -> dict[str, Any]:
    """The real Sioux Falls network: 76 links, top-12 OD pairs, k=3 routes."""
    s = _sioux()
    return {
        "n_nodes": int(s["tntp"].n_nodes),
        "links": s["links"],
        "od_pairs": [[int(o), int(d)] for o, d in s["od_pairs"]],
        "n_routes": int(s["network"].n_routes),
        "warnings": [
            "route sets restricted to k=3 shortest per OD (documented plugin restriction)"
        ],
    }


class SUEPayload(BaseModel):
    """Stochastic-user-equilibrium request with optional per-link tolls."""

    theta: Annotated[float, Field(gt=0, le=10)] = 0.5
    tolls: dict[int, float] | None = Field(
        default=None, description="Sparse per-link tolls (link index -> toll)."
    )


@app.post("/v1/domains/sioux_falls/sue")
def sioux_sue(payload: SUEPayload) -> dict[str, Any]:
    """Solve the Fisk SUE on real data; returns link flows, costs and totals.

    Tolls enter route costs exactly linearly — the congestion plugin's
    conjugate field. This is the measurement procedure behind the network's
    R = 0 calibration reading: toll a link, watch flows re-equilibrate.
    """
    from strataq.population.games.routing import solve_sue

    s = _sioux()
    net = s["network"]
    tolls = None
    if payload.tolls:
        import numpy as _np

        arr = _np.zeros(net.n_links)
        for k, v in payload.tolls.items():
            if not 0 <= int(k) < net.n_links:
                raise HTTPException(status_code=422, detail="toll link index out of range")
            if not -100 <= float(v) <= 100:
                raise HTTPException(status_code=422, detail="toll magnitude out of range")
            arr[int(k)] = float(v)
        tolls = jnp.asarray(arr)
    x, residual, steps = solve_sue(net, payload.theta, tolls=tolls)
    v = net.link_flows(x)
    costs = net.link_costs(v)
    total_time = float(v @ costs)
    return {
        "link_flows": [float(f) for f in v],
        "link_costs": [float(c) for c in costs],
        "total_travel_time": total_time,
        "beckmann": float(net.beckmann(x)),
        "residual": float(residual),
        "newton_steps": int(steps),
        "warnings": [],
    }


# ---------------------------------------------------------------------------
# /v1/toolkit — the plain-data product surface (unit product.toolkit): the
# same three questions strataq.toolkit answers in Python, over HTTP. Verdicts
# carry their honesty warnings; guards raise 422s with instructive messages.
# ---------------------------------------------------------------------------


class ToolkitReciprocityPayload(BaseModel):
    chi: list[list[float]]
    chi_se: list[list[float]] | None = None


@app.post("/v1/toolkit/reciprocity")
def toolkit_reciprocity(payload: ToolkitReciprocityPayload) -> dict[str, Any]:
    """R + verdict from a measured cross-response matrix (curl-able F-0011).

    Example: curl -X POST .../v1/toolkit/reciprocity -H 'Content-Type: application/json'
    -d '{"chi": [[1.07, 0.003], [0.0005, 0.97]]}'
    """
    try:
        read = tk.reciprocity_read(payload.chi, chi_se=payload.chi_se)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return {
        "r": read.r,
        "verdict": read.verdict,
        "ci_low": read.ci_low,
        "ci_high": read.ci_high,
        "calibration": read.calibration,
        "warnings": read.warnings,
    }


class ToolkitSeriesPayload(BaseModel):
    series: list[float]
    n_bins: int = 3
    n_surrogates: int = 150
    alpha_level: float = 0.01
    seed: int = 0


@app.post("/v1/toolkit/irreversibility")
def toolkit_irreversibility(payload: ToolkitSeriesPayload) -> dict[str, Any]:
    """Irreversibility verdict for a scalar time series (the F-0009 instrument)."""
    if len(payload.series) > 20_000:
        raise HTTPException(status_code=422, detail="series capped at 20000 points on this host")
    if not 1 <= payload.n_bins <= 6 or not 20 <= payload.n_surrogates <= 500:
        raise HTTPException(status_code=422, detail="n_bins in [1,6], n_surrogates in [20,500]")
    try:
        v = tk.irreversibility_test(
            payload.series,
            n_bins=payload.n_bins,
            n_surrogates=payload.n_surrogates,
            alpha_level=payload.alpha_level,
            seed=payload.seed,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return {
        "detected": v.detected,
        "p_value": v.p_value,
        "statistic": v.statistic,
        "null_quantile": v.null_quantile,
        "null_median": v.null_median,
        "null_mismatch_low": v.null_mismatch_low,
        "n_surrogates": v.n_surrogates,
        "warnings": (
            ["n < 300: underpowered — a non-detection here is weak evidence"]
            if len(payload.series) < 300
            else []
        ),
    }


class ToolkitRationalityPayload(BaseModel):
    payoff_matrices: list[list[list[float]]]
    counts: list[list[float]]
    lam_min: float = 0.05
    lam_max: float = 20.0


@app.post("/v1/toolkit/rationality")
def toolkit_rationality(payload: ToolkitRationalityPayload) -> dict[str, Any]:
    """Bayesian lambda posterior from choice counts under a known game."""
    if len(payload.payoff_matrices) > settings.max_players:
        raise HTTPException(status_code=422, detail=f"max {settings.max_players} players")
    for u in payload.payoff_matrices:
        if len(u) > settings.max_actions_per_player:
            raise HTTPException(
                status_code=422, detail=f"max {settings.max_actions_per_player} actions/player"
            )
    try:
        est = tk.estimate_rationality(
            list(payload.payoff_matrices),
            list(payload.counts),
            lam_range=(payload.lam_min, payload.lam_max),
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return {
        "mean": est.mean,
        "map": est.map,
        "ci_low": est.ci_low,
        "ci_high": est.ci_high,
        "grid_resolved": est.grid_resolved,
        "warnings": est.warnings,
    }


# ---------------------------------------------------------------------------
# /v1/domains/blotto — the allocation lab's backend (plan-v2 A3 second half).
# Budgets are the conjugate field: slide them and read alpha / R / EPR move.
# ---------------------------------------------------------------------------


class BlottoPayload(BaseModel):
    budget_a: int = Field(ge=1, le=8)
    budget_b: int = Field(ge=1, le=8)
    n_fields: int = Field(default=3, ge=2, le=3)
    field_values: list[float] | None = None
    lam: float = Field(default=1.5, gt=0.0, le=20.0)


@app.post("/v1/domains/blotto/read")
def blotto_read(payload: BlottoPayload) -> dict[str, Any]:
    """Full instrument read of a Colonel Blotto game at the given budgets.

    Returns each player's QRE allocation mix (for the heatmap), alpha, R,
    and — when the joint space fits the dense guard — the dissipation read.
    """
    from strataq.domains.blotto.oracle import BlottoOracle, blotto_game_tensors
    from strataq.finite.decompose.hodge import alpha as harmonic_fraction
    from strataq.finite.response.reciprocity import reciprocity_defect

    values = payload.field_values or [1.0] * payload.n_fields
    if len(values) != payload.n_fields:
        raise HTTPException(status_code=422, detail="field_values length must equal n_fields")
    oracle = BlottoOracle(jnp.asarray(values, dtype=jnp.float64))
    u_a, u_b, grid_a, grid_b = blotto_game_tensors(oracle, (payload.budget_a, payload.budget_b))
    game = DenseTensorGame((u_a, u_b))
    n_states = len(grid_a) * len(grid_b)
    point = logit_qre(game, payload.lam)
    a_val = float(harmonic_fraction(game))
    r_val = float(reciprocity_defect(game, point))
    out: dict[str, Any] = {
        "allocations_a": [list(g) for g in grid_a],
        "allocations_b": [list(g) for g in grid_b],
        "sigma_a": [float(x) for x in point.sigma[0]],
        "sigma_b": [float(x) for x in point.sigma[1]],
        "alpha": a_val,
        "r": r_val,
        "n_joint_states": n_states,
        "warnings": [],
    }
    if n_states <= settings.max_profile_states:
        reading = thermo_read(game, payload.lam)
        out["epr"] = float(reading.epr)
        out["max_current"] = float(reading.max_current)
    else:
        out["epr"] = None
        out["warnings"].append(
            f"joint space {n_states} exceeds the dense dynamics guard "
            f"({settings.max_profile_states}); alpha and R are exact, EPR omitted"
        )
    return out


# ---------------------------------------------------------------------------
# /v1/diagnose and /v1/fit — the two product entry points over HTTP.
#
# `diagnose` is the whole-system verdict (COMPETITIVE_POSITION §5: "hand us
# your data in your format and get a number with an uncertainty band and a
# null-model comparison"); `fit` is the estimation workflow of Bland & Turocy
# (GEB 2025). Both return the FULL library object — every coordinate with its
# kind and bounds, every warning, every refusal — because the honesty
# affordances are the product, and an API that drops them ships a bare number.
# ---------------------------------------------------------------------------


class DiagnosePayload(BaseModel):
    """A game (``payoffs`` + ``lam``) or readings you already have (``chi`` / ``series``)."""

    payoffs: list[Any] | None = None
    lam: Annotated[float, Field(gt=0, le=100)] | None = None
    chi: list[list[float]] | None = None
    chi_se: list[list[float]] | None = None
    series: list[float] | None = None
    n_bins: Annotated[int, Field(ge=1, le=6)] = 3
    n_surrogates: Annotated[int, Field(ge=20, le=500)] = 200
    alpha_level: Annotated[float, Field(gt=0.0, lt=0.5)] = 0.01
    seed: Annotated[int, Field(ge=0)] = 0


@app.post("/v1/diagnose")
def diagnose_reading(payload: DiagnosePayload) -> dict[str, Any]:
    """Locate a system in the irreversibility plane: one quadrant, two bounded coordinates.

    Returns the whole :class:`~strataq.diagnose.Diagnosis`: the verdict, the live
    quadrant set when it cannot be narrowed to one, both coordinates with their
    ``kind`` (point / interval / one-sided bound / absent) and their method text,
    ``alpha``, ``lam``, every warning, every refusal, and provenance. Refusals are
    bounds, not errors — a reading that cannot separate two quadrants says which
    two rather than returning nothing.
    """
    if payload.payoffs is None and payload.chi is None and payload.series is None:
        raise HTTPException(
            status_code=422,
            detail="supply payoffs= (with lam=), chi= (a measured response matrix), "
            "or series= (an observed trajectory)",
        )
    game: DenseTensorGame | None = None
    if payload.payoffs is not None:
        if payload.lam is None:
            raise HTTPException(status_code=422, detail="payoffs= requires lam=")
        game = _game_from(GamePayload(payoffs=payload.payoffs, lam=payload.lam))
        n_states = 1
        for m in game.num_actions:
            n_states *= m
        if n_states > settings.max_profile_states:
            raise HTTPException(
                status_code=413,
                detail=f"profile space {n_states} exceeds the dense generator limit "
                f"{settings.max_profile_states}; supply series= and read dissipation "
                "from the trajectory instead",
            )
    if payload.chi is not None:
        n = len(payload.chi)
        limit = settings.max_actions_per_player * settings.max_players
        if n == 0 or any(len(row) != n for row in payload.chi):
            raise HTTPException(status_code=422, detail="chi must be a non-empty square matrix")
        if n > limit:
            raise HTTPException(status_code=413, detail=f"chi capped at {limit}x{limit}")
        if payload.chi_se is not None and (
            len(payload.chi_se) != n or any(len(row) != n for row in payload.chi_se)
        ):
            raise HTTPException(status_code=422, detail="chi_se must match chi's shape")
    if payload.series is not None and len(payload.series) > 20_000:
        raise HTTPException(status_code=422, detail="series capped at 20000 points on this host")

    try:
        reading = diagnose(
            game,
            lam=payload.lam,
            chi=payload.chi,
            chi_se=payload.chi_se,
            series=payload.series,
            n_bins=payload.n_bins,
            n_surrogates=payload.n_surrogates,
            alpha_level=payload.alpha_level,
            seed=payload.seed,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    out = reading.as_dict()
    out["headline"] = repr(reading)
    out["snippet"] = reading.snippet()
    if game is not None and payload.lam is not None:
        out["provenance"] = {
            **out["provenance"],
            **_provenance(game, payload.lam).model_dump(),
        }
    return out


class FitPayload(BaseModel):
    """Observed choices for :func:`strataq.fit` — aggregated counts OR a tidy table.

    ``data`` is the point of this endpoint: one row per observed choice, with the
    subject / round / treatment columns your experiment software emitted, so the
    panel structure survives into the likelihood and into the interval.
    """

    payoffs: list[Any]
    counts: list[list[float]] | None = None
    data: list[dict[str, Any]] | dict[str, list[Any]] | None = None
    by: str | None = None
    method: Literal["mle", "agreement", "bayes"] = "mle"
    ci: Literal["bootstrap", "profile", "posterior", "none"] = "bootstrap"
    n_boot: Annotated[int, Field(ge=1, le=2000)] = 200
    n_grid: Annotated[int, Field(ge=8, le=400)] = 120
    level: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.95
    seed: Annotated[int, Field(ge=0)] = 0
    player: str | None = None
    action: str | None = None
    subject: str | None = None


@app.post("/v1/fit")
def fit_lambda(payload: FitPayload) -> dict[str, Any]:
    """Estimate lambda with an interval that names its method, and both LR tests.

    Returns lambda-hat (``null`` when the likelihood is flat — the refusal is a
    bound, not a number), the interval and the method that produced it, n, the
    log-likelihood, likelihood-ratio tests against Nash (lambda -> inf) and
    uniform (lambda = 0) with df, p and the boundary caveat, per-group lambda and
    a homogeneity test under ``by=``, every warning and refusal, provenance, and
    the rendered ``summary`` text.
    """
    game = _game_from(GamePayload(payoffs=payload.payoffs, lam=1.0))
    if (payload.counts is None) == (payload.data is None):
        raise HTTPException(
            status_code=422,
            detail="supply exactly one of counts= (aggregated, one vector per player) or "
            "data= (tidy, one row per observed choice)",
        )
    observed: Any
    if payload.counts is not None:
        total = sum(sum(c) for c in payload.counts)
        if total <= 0:
            raise HTTPException(status_code=422, detail="counts must contain observations")
        if total > 10_000_000:
            raise HTTPException(status_code=413, detail="count total exceeds service limit")
        observed = payload.counts
    else:
        rows = payload.data
        if isinstance(rows, list):
            n_rows = len(rows)
        else:
            n_rows = len(next(iter(rows.values()))) if rows else 0
        if n_rows == 0:
            raise HTTPException(status_code=422, detail="data= is empty")
        if n_rows > settings.max_tidy_rows:
            raise HTTPException(
                status_code=413,
                detail=f"tidy table capped at {settings.max_tidy_rows} rows on this host",
            )
        observed = rows

    try:
        result = fit(
            game,
            observed,
            by=payload.by,
            method=payload.method,
            ci=payload.ci,
            n_boot=payload.n_boot,
            seed=payload.seed,
            level=payload.level,
            player=payload.player,
            action=payload.action,
            subject=payload.subject,
            n_grid=payload.n_grid,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    out = result.as_dict()
    out["summary"] = str(result.summary())
    out["provenance"] = {
        **out["provenance"],
        **_provenance(game, result.lam_hat).model_dump(),
    }
    return out


# ---------------------------------------------------------------------------
# /v1/solve/* — the problem API over HTTP. One endpoint per problem type; the
# body mirrors the Python constructor and the response is the Solution as JSON
# (domain-named fields, plus ``diagnostics`` only when asked for). Bad input is
# a 422 carrying the library's own message; a solve that misses tolerance
# returns ``success: false`` and says so in ``warnings`` rather than raising.
#
# Grids arrive either as explicit levels (``grid=[1.0, 1.1, ...]``) or as a
# range (``grid_range=[start, stop, step]``) — exactly one, so the JSON is never
# ambiguous the way a bare three-element list would be.
# ---------------------------------------------------------------------------

MAX_SOLVE_PROFILES = 20_000
MAX_SOLVE_LEVELS = 60
MAX_SOLVE_EDGES = 200
MAX_SOLVE_OD = 20

GridInput = tuple[float, float, float] | list[float]


def _run_solve(build: Callable[[], Any], diagnostics: bool) -> dict[str, Any]:
    """Solve, collect ordinary Python warnings, and shape the JSON response."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            solution = build()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except OSError as exc:  # dataset fetch
            raise HTTPException(status_code=503, detail=f"dataset unavailable: {exc}") from exc
    body: dict[str, Any] = dict(solution.as_dict())
    body["warnings"] = [str(w.message) for w in caught]
    if diagnostics:
        body["diagnostics"] = solution.diagnostics.as_dict()
    return body


def _grid(levels: list[float] | None, span: list[float] | None, name: str) -> GridInput:
    """Exactly one of an explicit level list or a ``[start, stop, step]`` range."""
    if (levels is None) == (span is None):
        raise HTTPException(
            status_code=422,
            detail=f"supply exactly one of {name}= (explicit levels) or "
            f"{name}_range= ([start, stop, step])",
        )
    if span is not None:
        if len(span) != 3:
            raise HTTPException(status_code=422, detail=f"{name}_range must be [start, stop, step]")
        start, stop, step = span
        if step <= 0 or stop < start:
            raise HTTPException(
                status_code=422, detail=f"{name}_range needs step > 0 and stop >= start"
            )
        if (stop - start) / step + 1 > MAX_SOLVE_LEVELS:
            raise HTTPException(
                status_code=413, detail=f"{name} exceeds {MAX_SOLVE_LEVELS} levels (sync API)"
            )
        return (start, stop, step)
    assert levels is not None
    if len(levels) > MAX_SOLVE_LEVELS:
        raise HTTPException(
            status_code=413, detail=f"{name} exceeds {MAX_SOLVE_LEVELS} levels (sync API)"
        )
    return list(levels)


def _n_levels(spec: GridInput) -> int:
    if isinstance(spec, tuple):
        start, stop, step = spec
        return int((stop - start) / step) + 1
    return len(spec)


class DemandPayload(BaseModel):
    """A demand system. ``CustomDemand`` is Python-only — it takes a callable."""

    kind: Literal["logit", "linear"] = "logit"
    price_sensitivity: float | None = None
    quality: list[float] | None = None
    market_size: float = 1.0
    outside_option: bool = True
    intercept: list[float] | None = None
    own_slope: float | None = None
    cross_slope: float = 0.0

    def build(self) -> DemandModel:
        if self.kind == "logit":
            if self.price_sensitivity is None or self.quality is None:
                raise HTTPException(
                    status_code=422, detail="logit demand needs price_sensitivity= and quality="
                )
            return LogitDemand(
                self.price_sensitivity,
                self.quality,
                market_size=self.market_size,
                outside_option=self.outside_option,
            )
        if self.intercept is None or self.own_slope is None:
            raise HTTPException(
                status_code=422, detail="linear demand needs intercept= and own_slope="
            )
        return LinearDemand(self.intercept, self.own_slope, self.cross_slope)


class PricingPayload(BaseModel):
    costs: list[float] = Field(min_length=1, max_length=3)
    demand: DemandPayload
    grid: list[float] | None = None
    grid_range: list[float] | None = None
    precision: Annotated[float, Field(gt=0, le=1000)] = 1.0
    firm: Annotated[int, Field(ge=0)] = 0
    diagnostics: bool = False


@app.post("/v1/solve/pricing")
def solve_pricing(payload: PricingPayload) -> dict[str, Any]:
    """Recommended price, expected profit, rival price distribution, elasticities."""
    grid = _grid(payload.grid, payload.grid_range, "grid")
    if _n_levels(grid) ** len(payload.costs) > MAX_SOLVE_PROFILES:
        raise HTTPException(status_code=413, detail="price grid too large for the sync API")
    demand = payload.demand.build()
    return _run_solve(
        lambda: PricingProblem(
            costs=payload.costs,
            grid=grid,
            demand=demand,
            precision=payload.precision,
            firm=payload.firm,
        ).solve(),
        payload.diagnostics,
    )


class AuctionPayload(BaseModel):
    values: list[float] | None = None
    costs: list[float] | None = None
    grid: list[float] | None = None
    grid_range: list[float] | None = None
    n_bidders: Annotated[int, Field(ge=1, le=4)] | None = None
    reserve: float | None = None
    precision: Annotated[float, Field(gt=0, le=1000)] = 1.0
    bidder: Annotated[int, Field(ge=0)] = 0
    diagnostics: bool = False


@app.post("/v1/solve/auction")
def solve_auction(payload: AuctionPayload) -> dict[str, Any]:
    """Recommended bid, expected surplus, win probability, rival-bid distribution."""
    grid = _grid(payload.grid, payload.grid_range, "grid")
    listed = payload.values if payload.values is not None else payload.costs
    count = payload.n_bidders or (len(listed) if listed is not None else 1)
    if _n_levels(grid) ** count > MAX_SOLVE_PROFILES:
        raise HTTPException(status_code=413, detail="bid grid too large for the sync API")
    return _run_solve(
        lambda: AuctionProblem(
            grid=grid,
            values=payload.values,
            costs=payload.costs,
            n_bidders=payload.n_bidders,
            reserve=payload.reserve,
            precision=payload.precision,
            bidder=payload.bidder,
        ).solve(),
        payload.diagnostics,
    )


class EdgePayload(BaseModel):
    origin: int
    destination: int
    free_flow: float
    capacity: float
    b: float = 0.15
    power: float = 4.0


class ODPayload(BaseModel):
    origin: int
    destination: int
    trips: float


class RoutingPayload(BaseModel):
    network: Literal["sioux_falls"] | list[EdgePayload] = "sioux_falls"
    demand: list[ODPayload] | None = None
    tolls: dict[int, float] | None = None
    precision: Annotated[float, Field(gt=0, le=100)] = 0.5
    k_routes: Annotated[int, Field(ge=1, le=6)] = 3
    max_od: Annotated[int, Field(ge=1, le=MAX_SOLVE_OD)] = 12
    diagnostics: bool = False


@app.post("/v1/solve/routing")
def solve_routing(payload: RoutingPayload) -> dict[str, Any]:
    """Link flows, travel times, total cost, and what a toll did to them."""
    if isinstance(payload.network, list) and len(payload.network) > MAX_SOLVE_EDGES:
        raise HTTPException(status_code=413, detail=f"max {MAX_SOLVE_EDGES} edges (sync API)")
    if payload.demand is not None and len(payload.demand) > MAX_SOLVE_OD:
        raise HTTPException(status_code=413, detail=f"max {MAX_SOLVE_OD} OD pairs (sync API)")
    network: Any = payload.network
    if isinstance(network, list):
        network = [
            {
                "from": edge.origin,
                "to": edge.destination,
                "free_flow": edge.free_flow,
                "capacity": edge.capacity,
                "b": edge.b,
                "power": edge.power,
            }
            for edge in network
        ]
    demand = (
        None
        if payload.demand is None
        else {(od.origin, od.destination): od.trips for od in payload.demand}
    )
    return _run_solve(
        lambda: RoutingProblem(
            network=network,
            demand=demand,
            tolls=payload.tolls,
            precision=payload.precision,
            k_routes=payload.k_routes,
            max_od=payload.max_od,
        ).solve(),
        payload.diagnostics,
    )


class AllocationPayload(BaseModel):
    budget: Annotated[int, Field(ge=1, le=12)]
    field_values: list[float] | None = None
    n_fields: Annotated[int, Field(ge=2, le=4)] | None = None
    rival_budget: Annotated[int, Field(ge=1, le=12)] | None = None
    precision: Annotated[float, Field(gt=0, le=1000)] = 1.0
    diagnostics: bool = False


@app.post("/v1/solve/allocation")
def solve_allocation(payload: AllocationPayload) -> dict[str, Any]:
    """Recommended Blotto allocation, win probability, and both sides' mixes."""
    return _run_solve(
        lambda: AllocationProblem(
            budget=payload.budget,
            field_values=payload.field_values,
            n_fields=payload.n_fields,
            rival_budget=payload.rival_budget,
            precision=payload.precision,
        ).solve(),
        payload.diagnostics,
    )


class ElectricityPayload(BaseModel):
    costs: list[float] = Field(min_length=2, max_length=2)
    offers: list[float] | None = None
    offers_range: list[float] | None = None
    capacities: list[float] | None = None
    demand: Annotated[float, Field(gt=0)] = 1.0
    precision: Annotated[float, Field(gt=0, le=1000)] = 1.0
    generator: Annotated[int, Field(ge=0, le=1)] = 0
    diagnostics: bool = False


@app.post("/v1/solve/electricity")
def solve_electricity(payload: ElectricityPayload) -> dict[str, Any]:
    """Offer curve, expected clearing price, expected revenue for one generator."""
    offers = _grid(payload.offers, payload.offers_range, "offers")
    return _run_solve(
        lambda: ElectricityProblem(
            costs=payload.costs,
            offers=offers,
            capacities=payload.capacities,
            demand=payload.demand,
            precision=payload.precision,
            generator=payload.generator,
        ).solve(),
        payload.diagnostics,
    )
