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

from pydantic import BaseModel, ConfigDict, Field

# Pydantic defaults to extra="ignore", which silently DROPS misspelled keyword
# arguments: an experiment can pass `n_samples=` / `seeds=` / `interpretation=`
# and land an impoverished artifact with no error anywhere (observed while
# writing R9 — only the two REQUIRED fields raised, so three wrong names went
# through unnoticed). Forbidding extras turns every such typo into an immediate
# failure at the point of construction. Verified against every committed
# benchmarks/results/*.json before landing, so no existing artifact is
# invalidated by the stricter policy.
#
# ser_json_inf_nan="constants": pydantic's DEFAULT writes float('nan') out as
# JSON `null`, so a metric deliberately set to NaN ("no estimate — the model
# was rejected") lands on disk as a value the schema itself cannot read back.
# Two committed artifacts predate this and still carry nulls
# (electricity_lambda, frontier_lambda_c); nothing broke only because the gate
# runner reads raw JSON rather than validating. Writing NaN as `NaN` keeps the
# sentinel round-trippable.
_STRICT = ConfigDict(extra="forbid", ser_json_inf_nan="constants")


class EffectSize(BaseModel):
    """An effect size with its confidence interval — the unit of quantitative claim."""

    model_config = _STRICT

    name: str
    value: float
    ci_low: float
    ci_high: float
    ci_level: float = 0.95
    method: str = Field(description="How the CI was obtained (bootstrap, analytic, HPD).")


class BenchmarkResult(BaseModel):
    """One benchmark outcome, self-describing and reproducible."""

    model_config = _STRICT

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
