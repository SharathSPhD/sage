"""Dynamics/thermo calibration — regenerates the dynamics.exact gate artifacts.

Run: ``uv run python -m experiments.dynamics_calibration``
Four artifacts: Gibbs agreement, equilibrium-reads-zero, NESS-reads-positive,
and the cross-instrument chain co-movement (EPR and ℛ along α — conjecture C1's
first data).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import strataq
import yaml
from strataq.core.solve.fixedpoint import logit_qre
from strataq.finite.decompose.generate import make_family
from strataq.finite.games.library import (
    congestion,
    congestion_potential,
    coordination,
    matching_pennies,
    rock_paper_scissors,
)
from strataq.finite.games.tensor import DenseTensorGame
from strataq.finite.response.reciprocity import reciprocity_defect
from strataq.thermo.exact import gibbs_distribution, thermo_read
from strataq_bench import BenchmarkResult, EffectSize

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
CONFIG = REPO / "config" / "experiments" / "dynamics_calibration.yaml"
UNIT = "dynamics.exact"

COSTS = jnp.array([[1.0, 2.5, 4.0], [1.5, 2.0, 3.5], [0.5, 3.0, 5.0]])


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _write(result: BenchmarkResult) -> None:
    path = RESULTS / f"{result.benchmark_id}.json"
    path.write_text(result.model_dump_json(indent=2) + "\n")
    print(f"[{'PASS' if result.passed else 'FAIL'}] {result.benchmark_id} -> {path.name}")


def _spearman(x: jnp.ndarray, y: jnp.ndarray) -> float:
    def ranks(v: jnp.ndarray) -> jnp.ndarray:
        order = jnp.argsort(v)
        return jnp.zeros_like(v).at[order].set(jnp.arange(v.shape[0], dtype=v.dtype))

    rx = ranks(x) - (x.shape[0] - 1) / 2.0
    ry = ranks(y) - (y.shape[0] - 1) / 2.0
    return float((rx @ ry) / jnp.sqrt((rx @ rx) * (ry @ ry)))


def run() -> int:
    cfg = yaml.safe_load(CONFIG.read_text())
    seed = int(cfg["seed"])
    lam = float(cfg["lam"])
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "dynamics_calibration.resolved.yaml").write_text(
        yaml.safe_dump({"config": cfg, "library_version": strataq.__version__, "run_at": _now()})
    )
    failures = 0

    potential_cases = [
        ("congestion_n2", congestion(2, COSTS), congestion_potential(2, COSTS)),
        ("congestion_n3", congestion(3, COSTS), congestion_potential(3, COSTS)),
        ("coordination_2x3", coordination(2, 3, bonus=2.0), None),
    ]
    harmonic_cases = [("rps_3", rock_paper_scissors()), ("matching_pennies", matching_pennies())]

    # ---- 1. gibbs_agreement ------------------------------------------------
    worst_gap = 0.0
    for _name, game, phi in potential_cases:
        if phi is None:
            continue
        for lam_val in cfg["lam_grid"]:
            reading = thermo_read(game, float(lam_val))
            gap = float(jnp.max(jnp.abs(reading.pi - gibbs_distribution(phi, float(lam_val)))))
            worst_gap = max(worst_gap, gap)
    passed = worst_gap < 1e-10
    failures += not passed
    _write(
        BenchmarkResult(
            benchmark_id="gibbs_agreement",
            unit=UNIT,
            kind="correctness",
            passed=passed,
            metrics={"max_abs_gap": worst_gap},
            seed=seed,
            config_ref=str(CONFIG.relative_to(REPO)),
            library_version=strataq.__version__,
            timestamp=_now(),
            notes="Stationary pi of Glauber-logit matches exp(lambda*Phi)/Z to 1e-10 (K3).",
        )
    )

    # ---- 2. equilibrium_reads_zero ----------------------------------------
    max_epr = 0.0
    max_j = 0.0
    for _name, game, _phi in potential_cases:
        reading = thermo_read(game, lam)
        max_epr = max(max_epr, float(reading.epr))
        max_j = max(max_j, float(reading.max_current))
    passed = max_epr < 1e-12 and max_j < 1e-12
    failures += not passed
    _write(
        BenchmarkResult(
            benchmark_id="equilibrium_reads_zero",
            unit=UNIT,
            kind="correctness",
            passed=passed,
            metrics={"max_epr": max_epr, "max_current": max_j},
            seed=seed,
            config_ref=str(CONFIG.relative_to(REPO)),
            library_version=strataq.__version__,
            timestamp=_now(),
            notes="EPR and J* below 1e-12 on exact potential games (detailed balance).",
        )
    )

    # ---- 3. ness_reads_positive -------------------------------------------
    min_epr = float("inf")
    min_j = float("inf")
    for _name, game in harmonic_cases:
        reading = thermo_read(game, lam)
        min_epr = min(min_epr, float(reading.epr))
        min_j = min(min_j, float(reading.max_current))
    passed = min_epr > 1e-3 and min_j > 1e-4
    failures += not passed
    _write(
        BenchmarkResult(
            benchmark_id="ness_reads_positive",
            unit=UNIT,
            kind="correctness",
            passed=passed,
            metrics={"min_epr": min_epr, "min_current": min_j},
            seed=seed,
            config_ref=str(CONFIG.relative_to(REPO)),
            library_version=strataq.__version__,
            timestamp=_now(),
            notes="Harmonic games are NESS: positive dissipation and circulation.",
        )
    )

    # ---- 4. chain_comovement (statistical, C1 first data) ------------------
    sweep = cfg["sweep"]
    levels = [float(a) for a in sweep["levels"]]
    per_level = int(sweep["games_per_level"])
    shape = tuple(sweep["shape"])
    scale = float(sweep["scale"])

    alphas: list[float] = []
    eprs: list[float] = []
    rs: list[float] = []
    key = jax.random.PRNGKey(seed + 11)
    for level_idx, a in enumerate(levels):
        for g_idx in range(per_level):
            k = jax.random.fold_in(jax.random.fold_in(key, level_idx), g_idx)
            k1, k2, k3, k4 = jax.random.split(k, 4)
            pot_src = DenseTensorGame((jax.random.normal(k1, shape), jax.random.normal(k2, shape)))
            harm_src = DenseTensorGame((jax.random.normal(k3, shape), jax.random.normal(k4, shape)))
            try:
                game = make_family(pot_src, harm_src, [a], scale=scale)[0]
            except ValueError:
                continue
            reading = thermo_read(game, lam)
            point = logit_qre(game, lam, tol=1e-12, max_iter=100_000)
            alphas.append(a)
            eprs.append(float(reading.epr))
            rs.append(float(reciprocity_defect(game, point)))

    x = jnp.asarray(alphas)
    epr_arr = jnp.asarray(eprs)
    r_arr = jnp.asarray(rs)
    rho_epr_alpha = _spearman(x, epr_arr)
    rho_epr_r = _spearman(epr_arr, r_arr)
    n = len(alphas)

    # Within-level (alpha-conditional) coupling: the marginal rho is alpha-driven;
    # the conditional structure is the scientific content (findings F-0004).
    within = {}
    for level in levels:
        mask = [i for i, av in enumerate(alphas) if av == level]
        if len(mask) > 2:
            idx = jnp.asarray(mask)
            within[level] = _spearman(epr_arr[idx], r_arr[idx])

    boot_key = jax.random.PRNGKey(seed + 12)
    boot_a, boot_b = [], []
    for b in range(int(cfg["bootstrap"]["n_resamples"])):
        idx = jax.random.randint(jax.random.fold_in(boot_key, b), (n,), 0, n)
        boot_a.append(_spearman(x[idx], epr_arr[idx]))
        boot_b.append(_spearman(epr_arr[idx], r_arr[idx]))
    ci = lambda arr: (  # noqa: E731
        float(jnp.quantile(jnp.asarray(arr), 0.025)),
        float(jnp.quantile(jnp.asarray(arr), 0.975)),
    )
    a_lo, a_hi = ci(boot_a)
    b_lo, b_hi = ci(boot_b)

    passed = rho_epr_alpha > 0.9 and rho_epr_r > 0.8
    failures += not passed
    _write(
        BenchmarkResult(
            benchmark_id="chain_comovement",
            unit=UNIT,
            kind="statistical",
            passed=passed,
            metrics={
                "spearman_epr_alpha": rho_epr_alpha,
                "spearman_epr_R_marginal": rho_epr_r,
                "n_pairs": float(n),
                **{f"within_level_rho_alpha_{level}": v for level, v in within.items()},
                "within_level_rho_min": min(within.values()),
                "within_level_rho_max": max(within.values()),
            },
            effect_sizes=[
                EffectSize(
                    name="spearman_epr_alpha",
                    value=rho_epr_alpha,
                    ci_low=a_lo,
                    ci_high=a_hi,
                    method="bootstrap (2000 resamples, percentile)",
                ),
                EffectSize(
                    name="spearman_epr_R",
                    value=rho_epr_r,
                    ci_low=b_lo,
                    ci_high=b_hi,
                    method="bootstrap (2000 resamples, percentile)",
                ),
            ],
            n=n,
            n_justification=(
                f"{per_level} games x {len(levels)} levels: bootstrap CI half-width on "
                "Spearman rho < 0.02 at this n in pilot — an order of magnitude inside "
                "the gate margins (0.9 / 0.8)."
            ),
            seed=seed,
            config_ref=str(CONFIG.relative_to(REPO)),
            library_version=strataq.__version__,
            timestamp=_now(),
            notes=(
                "C1 data, honestly split: the MARGINAL rho(EPR, R) is alpha-driven. "
                "Within-level (alpha-conditional) coupling is strong (+0.8..0.88) for "
                "alpha <= 0.65, degrades above, and REVERSES SIGN at alpha = 0.95 "
                "(about -0.35): the meters decouple among near-pure-harmonic games — "
                "the C1 falsifier partially realised. Findings F-0004; discovered by "
                "red-team stratification, independently verified."
            ),
        )
    )
    return failures


if __name__ == "__main__":
    raise SystemExit(run())
