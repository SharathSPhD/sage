"""The strataq instruments over HTTP.

Sync endpoints for small games (size-guarded); every response carries
``provenance`` (library version, payoff normalisation, λ handling) and
``warnings`` (``near_criticality`` etc.) per services/api/CLAUDE.md. Async
queue, persistence and auth arrive with the deployment unit; this module is
the instrument surface itself, and everything here is a thin shim over
gate-closed library calls — no science in the API layer.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

import jax.numpy as jnp
import strataq
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from strataq.core.dynamics.markov import glauber_generator, profile_space
from strataq.core.dynamics.sample import sample_trajectories
from strataq.core.solve.fixedpoint import logit_qre
from strataq.core.solve.homotopy import logit_branch
from strataq.estimate.lam import lambda_dispersion, lambda_mle
from strataq.finite.decompose.hodge import hodge_decompose
from strataq.finite.games.tensor import DenseTensorGame
from strataq.finite.response.reciprocity import reciprocity_defect
from strataq.finite.response.spectral import strategic_spectrum
from strataq.finite.response.susceptibility import chi_equilibrium
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
