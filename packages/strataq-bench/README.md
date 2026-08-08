# strataq-bench

Benchmark harness + statistical validation for strataq. Results are machine-readable JSON (`strataq_bench.result.BenchmarkResult`) written to `benchmarks/results/`, consumed by the gate runner and the progress dashboard. Performance regressions beyond 1.2× against committed baselines fail the gate.
