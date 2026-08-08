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
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from strataq.core.solve.fixedpoint import logit_qre
from strataq.core.solve.homotopy import logit_branch
from strataq.finite.decompose.hodge import hodge_decompose
from strataq.finite.games.tensor import DenseTensorGame
from strataq.finite.response.reciprocity import reciprocity_defect
from strataq.finite.response.spectral import strategic_spectrum
from strataq.finite.response.susceptibility import chi_equilibrium
from strataq.thermo.exact import thermo_read


class Settings(BaseSettings):
    """Env-driven service limits (pydantic-settings; no secrets in the repo)."""

    max_actions_per_player: int = 12
    max_players: int = 3
    max_profile_states: int = 400  # dense dynamics guard
    model_config = {"env_prefix": "SAGE_API_"}


settings = Settings()
app = FastAPI(
    title="strataq API",
    version=strataq.__version__,
    description="Susceptibility, reciprocity, dissipation and phase readings for finite games.",
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
