---
name: experiment-runner
description: Run a science experiment reproducibly. Invoke as /experiment <name> [overrides...]. Hydra composition, seed discipline, resolved-config capture, findings logging.
---

# Experiment runner

1. Compose: `uv run python -m experiments.<name> --config-name=<cfg> [hydra overrides]`. Multirun sweeps (e.g. the α×λ phase map) use `--multirun` with ranges from config, never inline literals.
2. Verify before accepting results: the output directory contains the **resolved config** (Hydra job.config saved beside results), the seed used, and the library version. An experiment output without its resolved config is discarded and re-run.
3. Results feeding gates are converted to `BenchmarkResult` JSON in `benchmarks/results/` (strataq-bench schema) so the gate runner and dashboard can read them.
4. Anything unexpected in the readings — a meter reading nonzero where theory says zero, structure in R(alpha), estimator divergence — goes to `memory/findings.md` immediately, with the config path that produced it and an initial chase-or-parked status. Anomalies are the product.
5. Long runs go to background tasks; never block a session on a sweep. Check `make reproduce` still covers any new gate artifact.
