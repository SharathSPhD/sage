"""Toolkit acceptance artifact — unit product.toolkit (the PI's product directive).

Run: ``uv run python -m experiments.toolkit_verdicts``
Six cases, every one through the PUBLIC contract only (plain lists), each
expected to reproduce a committed research result or a designed behaviour:
whirlpool/landscape dashboards, the F-0011 matrix, a driven series, a
random walk, and the flat-likelihood honesty warning.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import jax
import numpy as np
import strataq
import strataq.toolkit as tk
import yaml
from strataq.core.dynamics.markov import glauber_generator
from strataq.core.dynamics.sample import sample_trajectories
from strataq.finite.games.library import matching_pennies
from strataq_bench import BenchmarkResult, EffectSize

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
CONFIG = REPO / "config" / "experiments" / "toolkit_verdicts.yaml"
UNIT = "product.toolkit"

RPS_U1 = [[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]]
RPS_U2 = [[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]]


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text())
    seed = int(cfg["seed"])
    (RESULTS / "toolkit_verdicts.resolved.yaml").write_text(yaml.safe_dump(cfg))
    checks: dict[str, float] = {}

    rps = tk.game_thermo([RPS_U1, RPS_U2])
    checks["rps_whirlpool"] = float(rps.alpha > 0.9 and "whirlpool" in rps.verdict)
    coord = tk.game_thermo([[[2.0, 0.0], [0.0, 2.0]], [[2.0, 0.0], [0.0, 2.0]]])
    checks["coordination_landscape"] = float(coord.alpha < 0.05 and coord.epr < 1e-9)

    read = tk.reciprocity_read(cfg["f0011_chi"])
    checks["f0011_matrix_reproduced"] = float(
        abs(read.r - float(cfg["f0011_r"])) < float(cfg["f0011_tol"])
    )

    ic = cfg["irreversibility"]
    gen = glauber_generator(matching_pennies(), 2.0)
    batch = sample_trajectories(
        gen, jax.random.PRNGKey(seed), n_steps=int(ic["driven_steps"]), n_trajectories=1
    )
    level = np.array([0.0, 1.0, 3.0, 2.0])
    driven = tk.irreversibility_test(
        level[np.asarray(batch.states[0])],
        n_bins=2,
        n_surrogates=int(ic["n_surrogates"]),
        seed=seed,
    )
    checks["driven_series_detected"] = float(driven.detected)
    walk = tk.irreversibility_test(
        np.cumsum(np.random.default_rng(seed).normal(size=int(ic["walk_steps"]))),
        n_bins=3,
        n_surrogates=int(ic["n_surrogates"]),
        seed=seed,
    )
    checks["random_walk_at_null"] = float(not walk.detected)

    flat = tk.estimate_rationality(
        [[[1.0, -1.0], [-1.0, 1.0]], [[-1.0, 1.0], [1.0, -1.0]]],
        [[500, 500], [500, 500]],
    )
    checks["flat_likelihood_warned"] = float(any("flat likelihood" in w for w in flat.warnings))

    passed = all(v == 1.0 for v in checks.values())
    _now = datetime.now(UTC).isoformat(timespec="seconds")
    result = BenchmarkResult(
        benchmark_id="toolkit_verdicts",
        unit=UNIT,
        kind="statistical",
        passed=bool(passed),
        metrics={
            **checks,
            "f0011_r_read": read.r,
            "driven_p": driven.p_value,
            "walk_p": walk.p_value,
        },
        effect_sizes=[
            EffectSize(
                name="acceptance cases passing (public plain-list contract only)",
                value=sum(checks.values()) / len(checks),
                ci_low=sum(checks.values()) / len(checks),
                ci_high=sum(checks.values()) / len(checks),
                method="deterministic acceptance suite (degenerate CI)",
            )
        ],
        n=len(checks),
        n_justification=(
            "six acceptance cases spanning all four facade entry points; each "
            "reproduces a committed research result or designed guard behaviour."
        ),
        seed=seed,
        config_ref=str(CONFIG.relative_to(REPO)),
        library_version=strataq.__version__,
        timestamp=_now,
        notes=(
            "The product claim P1: the facade exposes the gated instruments "
            "unchanged — F-0011's matrix reads its committed value through "
            "strataq.toolkit; the F-0009 null detects a driven series and stays "
            "quiet on a random walk; flat likelihoods warn instead of quoting."
        ),
    )
    path = RESULTS / "toolkit_verdicts.json"
    path.write_text(result.model_dump_json(indent=2) + "\n")
    print(f"[{'PASS' if passed else 'FAIL'}] toolkit_verdicts -> {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
