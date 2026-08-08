"""Solver validation — regenerates the solve.advanced gate artifacts.

Run: ``uv run python -m experiments.solver_validation``
Artifacts: gambit_agreement (fixed-λ QRE vs Gambit's homotopy),
solver_cross_agreement (damped vs mirror last-iterate),
implicit_chi_agreement (implicit-diff Jacobian vs Result 1 resolvent).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import strataq
from strataq.core.solve.fixedpoint import logit_qre
from strataq.core.solve.implicit import qre_sigma
from strataq.core.solve.mirror import logit_qre_mirror
from strataq.finite.games.library import matching_pennies, rock_paper_scissors
from strataq.finite.games.tensor import DenseTensorGame
from strataq.finite.response.susceptibility import chi_equilibrium
from strataq_bench import BenchmarkResult

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
UNIT = "solve.advanced"
SEED = 20260808
LAMS = (0.5, 1.5)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _write(result: BenchmarkResult) -> None:
    (RESULTS / f"{result.benchmark_id}.json").write_text(result.model_dump_json(indent=2) + "\n")
    print(f"[{'PASS' if result.passed else 'FAIL'}] {result.benchmark_id}")


def _random_games(n: int, shape: tuple[int, int]) -> list[DenseTensorGame]:
    key = jax.random.PRNGKey(SEED + 21)
    games = []
    for g in range(n):
        k1, k2 = jax.random.split(jax.random.fold_in(key, g))
        games.append(DenseTensorGame((jax.random.normal(k1, shape), jax.random.normal(k2, shape))))
    return games


def run() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    failures = 0

    # ---- gambit_agreement (skipped gracefully when pygambit absent) --------
    try:
        from strataq.core.solve.validate import gambit_qre_sigma, max_profile_gap

        games = [matching_pennies(), rock_paper_scissors(), *_random_games(30, (3, 3))]
        worst = 0.0
        for lam in LAMS:
            for game in games:
                ours = logit_qre(game, lam, tol=1e-14, max_iter=400_000)
                worst = max(worst, max_profile_gap(ours.sigma, gambit_qre_sigma(game, lam)))
        passed = worst < 1e-8
        failures += not passed
        _write(
            BenchmarkResult(
                benchmark_id="gambit_agreement",
                unit=UNIT,
                kind="correctness",
                passed=passed,
                metrics={"max_profile_gap": worst, "n_games": float(len(games) * len(LAMS))},
                seed=SEED,
                library_version=strataq.__version__,
                timestamp=_now(),
                notes="Fixed-lambda QRE matches pygambit's homotopy tracer to 1e-8 "
                "(32 games x 2 lambdas, incl. harmonic anchors).",
            )
        )
    except ImportError:
        print("[SKIP] gambit_agreement (pygambit not installed) — artifact left as-is")

    # ---- solver_cross_agreement -------------------------------------------
    worst = 0.0
    for lam in LAMS:
        for game in _random_games(30, (3, 3)):
            a = logit_qre(game, lam, tol=1e-13, max_iter=200_000)
            b = logit_qre_mirror(game, lam, tol=1e-13, max_iter=200_000)
            worst = max(
                worst,
                max(float(jnp.max(jnp.abs(x - y))) for x, y in zip(a.sigma, b.sigma, strict=True)),
            )
    passed = worst < 1e-9
    failures += not passed
    _write(
        BenchmarkResult(
            benchmark_id="solver_cross_agreement",
            unit=UNIT,
            kind="correctness",
            passed=passed,
            metrics={"max_profile_gap": worst},
            seed=SEED,
            library_version=strataq.__version__,
            timestamp=_now(),
            notes="Damped iteration and magnetic mirror descent land on the same QRE to 1e-9.",
        )
    )

    # ---- implicit_chi_agreement -------------------------------------------
    worst = 0.0
    lam_fd = 0.9
    for game in _random_games(20, (3, 3)):
        total = sum(game.num_actions)
        jac = jax.jacrev(lambda h, g=game: qre_sigma(g, h, lam_fd, 1e-13, 200_000))(
            jnp.zeros(total)
        )
        point = logit_qre(game, lam_fd, tol=1e-13, max_iter=200_000)
        chi = chi_equilibrium(game, point).chi_full
        worst = max(worst, float(jnp.max(jnp.abs(jac - chi))))
    passed = worst < 1e-8
    failures += not passed
    _write(
        BenchmarkResult(
            benchmark_id="implicit_chi_agreement",
            unit=UNIT,
            kind="correctness",
            passed=passed,
            metrics={"max_abs_gap": worst, "n_games": 20.0},
            seed=SEED,
            library_version=strataq.__version__,
            timestamp=_now(),
            notes="Custom-VJP Jacobian of the solved QRE equals the Result 1 resolvent "
            "(one operator, implemented once, reused).",
        )
    )
    return failures


if __name__ == "__main__":
    raise SystemExit(run())
