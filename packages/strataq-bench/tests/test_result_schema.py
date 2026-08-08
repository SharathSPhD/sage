"""Round-trip test for the benchmark result schema the gates and dashboard consume."""

from strataq_bench import BenchmarkResult, EffectSize


def test_result_round_trip() -> None:
    result = BenchmarkResult(
        benchmark_id="reciprocity_potential",
        unit="finite.response.reciprocity",
        kind="correctness",
        passed=True,
        metrics={"max_R": 3.2e-12},
        effect_sizes=[
            EffectSize(name="R_gap", value=0.4, ci_low=0.35, ci_high=0.45, method="bootstrap")
        ],
        n=200,
        n_justification="precision: CI half-width < 0.05 at n=200 in pilot",
        seed=1234,
        library_version="0.1.0.dev0",
        timestamp="2026-08-08T00:00:00Z",
    )
    restored = BenchmarkResult.model_validate_json(result.model_dump_json())
    assert restored == result
