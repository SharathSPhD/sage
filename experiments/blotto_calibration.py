"""Blotto calibration — regenerates the domains.blotto gate artifact.

Run: ``uv run python -m experiments.blotto_calibration``
One artifact: instrument readings across small Blotto instances, including the
degenerate null (equal values, budget 2 — constant payoffs, meters must read
exactly zero) and the budget-field probe (readings move when the budget moves).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import jax.numpy as jnp
import strataq
from strataq.core.solve.fixedpoint import logit_qre
from strataq.domains.blotto import BlottoOracle, blotto_game_tensors
from strataq.finite.decompose.hodge import alpha
from strataq.finite.games.tensor import DenseTensorGame
from strataq.finite.response.reciprocity import reciprocity_defect
from strataq.thermo.exact import thermo_read
from strataq_bench import BenchmarkResult

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
UNIT = "domains.blotto"
SEED = 20260808
LAM = 1.5


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _game(values: jnp.ndarray, budgets: tuple[int, int]) -> DenseTensorGame:
    u_a, u_b, _, _ = blotto_game_tensors(BlottoOracle(values), budgets)
    return DenseTensorGame((u_a, u_b))


def run() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, float] = {}

    # Symmetric budget-3, three equal fields: the canonical alpha>0 anchor.
    anchor = _game(jnp.array([1.0, 1.0, 1.0]), (3, 3))
    point = logit_qre(anchor, LAM, tol=1e-12, max_iter=100_000)
    metrics["alpha_b3_k3"] = alpha(anchor)
    metrics["R_b3_k3"] = float(reciprocity_defect(anchor, point))

    # Small asymmetric instance: dissipation + circulation (profile space 9).
    asym = _game(jnp.array([2.0, 1.0]), (2, 2))
    reading = thermo_read(asym, 2.0)
    metrics["epr_b2_k2_asym"] = float(reading.epr)
    metrics["max_current_b2_k2_asym"] = float(reading.max_current)

    # Degenerate null: equal values, budget 2 — constant payoffs, exact zeros.
    degenerate = _game(jnp.array([1.0, 1.0]), (2, 2))
    metrics["epr_degenerate_null"] = float(thermo_read(degenerate, 2.0).epr)

    # Budget field probe: R at budget 3 vs 4 (the conjugate field is live).
    bigger = _game(jnp.array([1.0, 1.0, 1.0]), (4, 4))
    b_point = logit_qre(bigger, LAM, tol=1e-12, max_iter=100_000)
    metrics["R_b4_k3"] = float(reciprocity_defect(bigger, b_point))
    metrics["alpha_b4_k3"] = alpha(bigger)

    passed = (
        metrics["alpha_b3_k3"] > 0.6
        and metrics["R_b3_k3"] > 0.1
        and metrics["epr_b2_k2_asym"] > 1e-4
        and metrics["epr_degenerate_null"] < 1e-12
        and metrics["R_b4_k3"] > 0.1
    )
    result = BenchmarkResult(
        benchmark_id="blotto_readings",
        unit=UNIT,
        kind="correctness",
        passed=passed,
        metrics=metrics,
        seed=SEED,
        library_version=strataq.__version__,
        timestamp=_now(),
        notes=(
            "The alpha > 0 anchor with free payoffs: high alpha and positive R on "
            "symmetric Blotto; positive dissipation/circulation on the asymmetric "
            "2-field instance; EXACT zero on the degenerate equal-values budget-2 "
            "instance (constant payoffs); budget field moves the grid and the readings."
        ),
    )
    (RESULTS / "blotto_readings.json").write_text(result.model_dump_json(indent=2) + "\n")
    print(f"[{'PASS' if passed else 'FAIL'}] blotto_readings {metrics}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(run())
