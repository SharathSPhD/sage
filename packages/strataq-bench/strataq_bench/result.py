"""The machine-readable benchmark result schema.

Every benchmark run — correctness, performance, or statistical — serialises to
this schema and lands in ``benchmarks/results/*.json``. The gate runner and the
progress dashboard both consume it; changing it is a breaking change for both.

Statistical discipline (master spec §16): quantitative claims carry an effect
size with a confidence interval, never a bare p-value; n is justified; seeds
are fixed and recorded.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EffectSize(BaseModel):
    """An effect size with its confidence interval — the unit of quantitative claim."""

    name: str
    value: float
    ci_low: float
    ci_high: float
    ci_level: float = 0.95
    method: str = Field(description="How the CI was obtained (bootstrap, analytic, HPD).")


class BenchmarkResult(BaseModel):
    """One benchmark outcome, self-describing and reproducible."""

    benchmark_id: str = Field(description="Stable id, e.g. 'reciprocity_potential'.")
    unit: str = Field(description="Gate unit this feeds, e.g. 'finite.response.reciprocity'.")
    kind: Literal["correctness", "performance", "statistical"]
    passed: bool
    metrics: dict[str, float] = Field(default_factory=dict)
    effect_sizes: list[EffectSize] = Field(default_factory=list)
    n: int | None = Field(default=None, description="Sample size used.")
    n_justification: str | None = Field(
        default=None,
        description="Power/precision reasoning for n. Required for kind='statistical'.",
    )
    seed: int | None = Field(default=None, description="RNG seed; required for reproducibility.")
    config_ref: str | None = Field(
        default=None, description="Path or hash of the resolved config that produced this."
    )
    library_version: str
    timestamp: str = Field(description="ISO-8601 UTC.")
    notes: str = ""
