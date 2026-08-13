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


class RepeatedConfig(BaseModel):
    """Infinitely repeated games: discount bracket and Edgeworth cycle search."""

    model_config = ConfigDict(frozen=True)

    delta_max: float = Field(gt=0, lt=1, description="Upper bracket for critical-delta bisection.")
    bisect_iters: int = Field(gt=0, description="Bisection steps for the critical discount factor.")
    cycle_max_steps: int = Field(gt=1, description="Revision steps in an Edgeworth path.")
    cycle_tol: float = Field(gt=0, description="Distribution match tolerance for cycle detection.")


class EvolutionaryConfig(BaseModel):
    """Replicator integration and the finite-population Moran chain."""

    model_config = ConfigDict(frozen=True)

    step: float = Field(gt=0, description="Replicator integration step.")
    steps: int = Field(gt=0, description="Replicator integration horizon.")
    rest_tol: float = Field(gt=0, description="|xdot| below which a point is a rest point.")
    mutation: float = Field(
        gt=0, lt=1, description="Moran mutation rate keeping the chain ergodic."
    )


class ExtensiveConfig(BaseModel):
    """Game-tree size guards."""

    model_config = ConfigDict(frozen=True)

    max_nodes: int = Field(gt=0, description="Largest tree the dense passes will build.")
    max_pure_strategies: int = Field(gt=0, description="Reduced normal form size guard.")
    damping_backoff: float = Field(
        gt=0, lt=1, description="Damping multiplier on an AQRE run that missed tolerance."
    )
    max_restarts: int = Field(gt=0, description="How many times to back the damping off.")
    continuation_points: int = Field(gt=1, description="Lambda steps in the continuation fallback.")


class SituationConfig(BaseModel):
    """The precision ladder solve_situation() sweeps for sensitivity."""

    model_config = ConfigDict(frozen=True)

    ladder_points: int = Field(gt=2, description="Number of precisions on the ladder.")
    ladder_low: float = Field(gt=0, description="Lowest multiple of the stated precision.")
    ladder_high: float = Field(gt=0, description="Highest multiple of the stated precision.")


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
    repeated: RepeatedConfig
    evolutionary: EvolutionaryConfig
    extensive: ExtensiveConfig
    situation: SituationConfig

    @classmethod
    def from_mapping(cls, cfg: dict[str, Any]) -> BaseConfig:
        """Validate a resolved OmegaConf container (``OmegaConf.to_container`` output)."""
        cfg = dict(cfg)
        cfg.pop("defaults", None)  # Hydra composition key, not config content
        return cls.model_validate(cfg)
