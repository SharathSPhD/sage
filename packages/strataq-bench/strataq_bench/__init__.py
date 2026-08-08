"""strataq_bench — benchmark harness and statistical validation.

Three benchmark kinds, all gate-feeding (master spec §16):
correctness (vs pygambit / analytic potentials), performance
(pytest-benchmark, 1.2x regression threshold), statistical (effect sizes,
CIs, justified n). Every benchmark writes machine-readable JSON conforming to
:mod:`strataq_bench.result` into ``benchmarks/results/``.
"""

__version__ = "0.1.0.dev0"

from strataq_bench.result import BenchmarkResult, EffectSize

__all__ = ["BenchmarkResult", "EffectSize", "__version__"]
