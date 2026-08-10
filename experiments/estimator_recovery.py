"""λ-estimator recovery — regenerates the estimate.lambda gate artifacts.

Run: ``uv run python -m experiments.estimator_recovery``
Two artifacts: recovery on well-specified synthetic data across λ* and α,
and the misspecification check (λ-mixture data must widen the estimator
spread — disagreement is the diagnostic, and it must actually fire).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import strataq
import yaml
from strataq.core.solve.fixedpoint import logit_qre
from strataq.estimate.lam import (
    agreement_protocol,
    lambda_dispersion,
    lambda_mle,
    lambda_moment_chi,
    sample_choices,
)
from strataq.finite.decompose.generate import make_family
from strataq.finite.games.library import congestion, rock_paper_scissors
from strataq.finite.games.tensor import DenseTensorGame
from strataq.finite.response.susceptibility import chi_equilibrium
from strataq_bench import BenchmarkResult, EffectSize

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
CONFIG = REPO / "config" / "experiments" / "estimator_recovery.yaml"
UNIT = "estimate.lambda"

COSTS = jnp.array([[1.0, 2.5, 4.0], [1.5, 2.0, 3.5], [0.5, 3.0, 5.0]])
ASYM = DenseTensorGame(
    (
        jnp.array([[3.0, 0.0, 1.5], [1.0, 2.0, 0.5], [0.0, 1.0, 2.5]]),
        jnp.array([[2.0, 1.0, 0.0], [0.5, 3.0, 1.0], [1.5, 0.0, 2.0]]),
    )
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _write(result: BenchmarkResult) -> None:
    path = RESULTS / f"{result.benchmark_id}.json"
    path.write_text(result.model_dump_json(indent=2) + "\n")
    print(f"[{'PASS' if result.passed else 'FAIL'}] {result.benchmark_id} -> {path.name}")


def run() -> int:
    cfg = yaml.safe_load(CONFIG.read_text())
    seed = int(cfg["seed"])
    n = int(cfg["n_samples"])
    thr = cfg["thresholds"]
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "estimator_recovery.resolved.yaml").write_text(
        yaml.safe_dump({"config": cfg, "library_version": strataq.__version__, "run_at": _now()})
    )
    failures = 0
    root = jax.random.key(seed)

    games = {"asym_3x3": ASYM}
    for alpha, game in zip(
        cfg["alphas"],
        make_family(
            congestion(2, COSTS),
            rock_paper_scissors(),
            [float(a) for a in cfg["alphas"]],
            scale=float(cfg["family_scale"]),
        ),
        strict=True,
    ):
        games[f"family_a{alpha:.2f}"] = game

    # ---- 1. estimator_recovery --------------------------------------------
    errs: dict[str, list[float]] = {"mle": [], "dispersion": [], "moment_chi": []}
    metrics: dict[str, float] = {}
    idx = 0
    for gname, game in games.items():
        for lam_star in (float(x) for x in cfg["lam_stars"]):
            idx += 1
            counts = sample_choices(game, lam_star, n, jax.random.fold_in(root, idx))
            mle = lambda_mle(game, counts)
            if not mle.warnings:  # identified cases only; warnings are their own check
                errs["mle"].append(abs(mle.lam - lam_star) / lam_star)
            disp = lambda_dispersion(game, counts)
            if not disp.warnings:
                errs["dispersion"].append(abs(disp.lam - lam_star) / lam_star)
            chi_obs = chi_equilibrium(game, logit_qre(game, lam_star)).chi_full
            mc = lambda_moment_chi(game, chi_obs)
            errs["moment_chi"].append(abs(mc.lam - lam_star) / lam_star)
            metrics[f"{gname}_lam{lam_star:g}_mle"] = mle.lam
            metrics[f"{gname}_lam{lam_star:g}_disp"] = disp.lam

    med = {k: float(np.median(v)) for k, v in errs.items() if v}
    ok = (
        med["mle"] < thr["mle_rel_err_max"]
        and med["dispersion"] < thr["dispersion_rel_err_max"]
        and med["moment_chi"] < thr["moment_chi_rel_err_max"]
    )
    failures += not ok
    rng = np.random.default_rng(seed)
    n_boot = int(cfg["bootstrap"]["n_resamples"])
    effect_sizes = []
    for name, v in errs.items():
        arr = np.array(v)
        boots = [float(np.median(arr[rng.integers(0, len(arr), len(arr))])) for _ in range(n_boot)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        effect_sizes.append(
            EffectSize(
                name=f"median relative error · {name}",
                value=float(np.median(arr)),
                ci_low=float(lo),
                ci_high=float(hi),
                method="bootstrap over cases",
            )
        )
        metrics[f"median_rel_err_{name}"] = float(np.median(arr))
    _write(
        BenchmarkResult(
            benchmark_id="estimator_recovery",
            unit=UNIT,
            kind="statistical",
            passed=ok,
            metrics=metrics,
            effect_sizes=effect_sizes,
            n=idx,
            n_justification=(
                f"{idx} (game, lambda*) cases; {n} choices per player per case gives "
                "multinomial SEs of ~0.3% per cell, well inside the recovery thresholds; "
                "unidentified cases (flagged by the estimators themselves) are excluded "
                "from error medians and checked separately."
            ),
            seed=seed,
            config_ref=str(CONFIG.relative_to(REPO)),
            library_version=strataq.__version__,
            timestamp=_now(),
            notes=(
                "All three data-side estimators recover lambda* on well-specified synthetic "
                "data (moment_chi from exact chi is the oracle anchor)."
            ),
        )
    )

    # ---- 2. estimator_misspecification ------------------------------------
    k1, k2, k3 = jax.random.split(jax.random.fold_in(root, 999), 3)
    clean = agreement_protocol(ASYM, sample_choices(ASYM, 1.2, 2 * n, k3))
    a = sample_choices(ASYM, 0.4, n, k1)
    b = sample_choices(ASYM, 4.0, n, k2)
    mixed = agreement_protocol(ASYM, tuple(x + y for x, y in zip(a, b, strict=True)))
    ratio = mixed.agreement_gap / max(clean.agreement_gap, 1e-9)
    rps = agreement_protocol(
        rock_paper_scissors(), sample_choices(rock_paper_scissors(), 1.5, 2 * n, k3)
    )
    unident = any("unidentified" in w or "flat" in w for w in rps.warnings)
    ok = ratio > thr["mixture_gap_ratio_min"] and unident
    failures += not ok
    _write(
        BenchmarkResult(
            benchmark_id="estimator_misspecification",
            unit=UNIT,
            kind="correctness",
            passed=ok,
            metrics={
                "clean_gap": clean.agreement_gap,
                "mixture_gap": mixed.agreement_gap,
                "gap_ratio": ratio,
                "rps_unidentified_warned": float(unident),
            },
            seed=seed,
            config_ref=str(CONFIG.relative_to(REPO)),
            library_version=strataq.__version__,
            timestamp=_now(),
            notes=(
                "The diagnostics fire when they must: lambda-mixture data widens the "
                "estimator spread by the required factor, and symmetric-RPS data (lambda "
                "unidentified from frequencies) produces a warning, not a number."
            ),
        )
    )
    return failures


if __name__ == "__main__":
    raise SystemExit(run())
