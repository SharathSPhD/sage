"""Typed configuration schemas — configs are objects, never raw dicts.

Hydra/OmegaConf composes YAML from ``config/``; the result is validated into
these Pydantic v2 models before any library code sees it. No literal
tolerance, seed, grid size or λ range appears in library code — it arrives
through these schemas.

References
----------
Master spec §7 (config-driven); config/base.yaml for the shipped defaults.
Engineering invariant, not a scientific claim.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class NumericsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    jax_enable_x64: bool = True
    dtype: Literal["float64", "float32"] = "float64"


class SeedConfig(BaseModel):
    """Seed policy: explicit PRNG-key threading from a recorded root."""

    model_config = ConfigDict(frozen=True)

    root: int
    policy: Literal["explicit-key-threading"] = "explicit-key-threading"


class ToleranceLadder(BaseModel):
    """The named tolerances tests and solvers draw from — never inline numbers."""

    model_config = ConfigDict(frozen=True)

    identity: float = Field(gt=0, description="Exact mathematical identities (K1–K7).")
    decompose: float = Field(gt=0, description="Hodge orthogonality/idempotence/reconstruction.")
    oracle: float = Field(gt=0, description="Cross-checks vs pygambit / analytic potentials.")
    fd: float = Field(gt=0, description="Finite-difference agreement for chi_eq / implicit diff.")
    solve: float = Field(gt=0, description="Fixed-point convergence residual.")


class LambdaConventions(BaseModel):
    """Payoff-scale folding: rescale internally, fold scale into λ, report both."""

    model_config = ConfigDict(frozen=True)

    normalise_payoffs: bool = True
    report_both: bool = True


class CriticalityConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    warn_below: float = Field(
        gt=0, description="Warn when distance_to_criticality = 1 - rho(SB) falls below this."
    )


class SolverDefaults(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_iter: int = Field(gt=0)
    damping: float = Field(gt=0, le=1)


class EstimateConfig(BaseModel):
    """λ-estimator family defaults (PROGRAMME v3 §4)."""

    model_config = ConfigDict(frozen=True)

    lam_min: float = Field(gt=0)
    lam_max: float = Field(gt=0)
    grid_points: int = Field(gt=4)
    refine_iters: int = Field(gt=0)
    profile_ci_drop: float = Field(gt=0)
    flat_ll_per_obs: float = Field(gt=0)
    flat_entropy_threshold: float = Field(gt=0)
    agreement_flag_gap: float = Field(gt=0)
    bootstrap_resamples: int = Field(gt=1)


class BaseConfig(BaseModel):
    """The validated form of ``config/base.yaml``."""

    model_config = ConfigDict(frozen=True)

    numerics: NumericsConfig
    seeds: SeedConfig
    tolerances: ToleranceLadder
    lambda_conventions: LambdaConventions
    criticality: CriticalityConfig
    solver: SolverDefaults
    estimate: EstimateConfig

    @classmethod
    def from_mapping(cls, cfg: dict[str, Any]) -> BaseConfig:
        """Validate a resolved OmegaConf container (``OmegaConf.to_container`` output)."""
        cfg = dict(cfg)
        cfg.pop("defaults", None)  # Hydra composition key, not config content
        return cls.model_validate(cfg)
