"""Reciprocity calibration — regenerates the finite.response.reciprocity gate artifacts.

Run: ``uv run python -m experiments.reciprocity_calibration``
Outputs: four BenchmarkResult JSONs in benchmarks/results/ plus the resolved
config beside them (reproducibility contract: everything regenerates from the
seed recorded here; `make reproduce` calls this).
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
    coordination,
    matching_pennies,
    rock_paper_scissors,
)
from strataq.finite.games.tensor import DenseTensorGame
from strataq.finite.response.reciprocity import reciprocity_defect
from strataq.finite.response.susceptibility import build_operators, chi_equilibrium, chi_fd
from strataq_bench import BenchmarkResult, EffectSize

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
CONFIG = REPO / "config" / "experiments" / "reciprocity_calibration.yaml"
UNIT = "finite.response.reciprocity"

COSTS = jnp.array([[1.0, 2.5, 4.0], [1.5, 2.0, 3.5], [0.5, 3.0, 5.0]])


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _write(result: BenchmarkResult) -> None:
    path = RESULTS / f"{result.benchmark_id}.json"
    path.write_text(result.model_dump_json(indent=2) + "\n")
    print(f"[{'PASS' if result.passed else 'FAIL'}] {result.benchmark_id} -> {path.name}")


def _solve_r(game: DenseTensorGame, lam: float, cfg: dict) -> float:
    point = logit_qre(game, lam, tol=cfg["solver"]["tol"], max_iter=cfg["solver"]["max_iter"])
    return float(reciprocity_defect(game, point))


def _spearman(x: jnp.ndarray, y: jnp.ndarray) -> float:
    def ranks(v: jnp.ndarray) -> jnp.ndarray:
        order = jnp.argsort(v)
        r = jnp.zeros_like(v).at[order].set(jnp.arange(v.shape[0], dtype=v.dtype))
        return r

    rx, ry = ranks(x), ranks(y)
    rx = rx - jnp.mean(rx)
    ry = ry - jnp.mean(ry)
    return float((rx @ ry) / jnp.sqrt((rx @ rx) * (ry @ ry)))


def _random_mixed_game(key, shape, scale) -> tuple[DenseTensorGame, DenseTensorGame]:
    k1, k2, k3, k4 = jax.random.split(key, 4)
    pot_src = DenseTensorGame((jax.random.normal(k1, shape), jax.random.normal(k2, shape)))
    harm_src = DenseTensorGame((jax.random.normal(k3, shape), jax.random.normal(k4, shape)))
    return pot_src, harm_src


def run() -> int:
    cfg = yaml.safe_load(CONFIG.read_text())
    seed = int(cfg["seed"])
    lam = float(cfg["lam"])
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "reciprocity_calibration.resolved.yaml").write_text(
        yaml.safe_dump({"config": cfg, "library_version": strataq.__version__, "run_at": _now()})
    )
    failures = 0

    # ---- 1. potential_reads_zero -------------------------------------------
    potential_games = [
        ("congestion_n2", congestion(2, COSTS)),
        ("congestion_n3", congestion(3, COSTS)),
        ("coordination_2x3", coordination(2, 3, bonus=2.0)),
        ("coordination_3x2", coordination(3, 2, bonus=1.5, mismatch=-0.5)),
        ("common_interest_random", None),  # filled below
    ]
    key = jax.random.PRNGKey(seed)
    v = jax.random.normal(key, (3, 3))
    from strataq.finite.games.library import common_interest

    potential_games[-1] = ("common_interest_random", common_interest(v))
    readings = {name: _solve_r(g, lam, cfg) for name, g in potential_games}
    max_r = max(readings.values())
    passed = max_r < 1e-10
    failures += not passed
    _write(
        BenchmarkResult(
            benchmark_id="reciprocity_potential",
            unit=UNIT,
            kind="correctness",
            passed=passed,
            metrics={f"R_{k}": val for k, val in readings.items()} | {"max_R": max_r},
            seed=seed,
            config_ref=str(CONFIG.relative_to(REPO)),
            library_version=strataq.__version__,
            timestamp=_now(),
            notes="R < 1e-10 required on 5 verified exact potential games.",
        )
    )

    # ---- 2. harmonic_reads_positive ----------------------------------------
    harmonic_games = [("rps_3", rock_paper_scissors(3)), ("rps_5", rock_paper_scissors(5))]
    if cfg["harmonic_games"]["include_matching_pennies"]:
        harmonic_games.append(("matching_pennies", matching_pennies()))
    h_readings = {name: _solve_r(g, lam, cfg) for name, g in harmonic_games}
    min_r = min(h_readings.values())
    passed = min_r > 0.1
    failures += not passed
    _write(
        BenchmarkResult(
            benchmark_id="reciprocity_harmonic",
            unit=UNIT,
            kind="correctness",
            passed=passed,
            metrics={f"R_{k}": val for k, val in h_readings.items()} | {"min_R": min_r},
            seed=seed,
            config_ref=str(CONFIG.relative_to(REPO)),
            library_version=strataq.__version__,
            timestamp=_now(),
            notes="R > 0.1 required on the RPS family (and matching pennies).",
        )
    )

    # ---- 3. monotone_in_alpha (statistical) --------------------------------
    sweep = cfg["alpha_sweep"]
    levels = [float(a) for a in sweep["levels"]]
    per_level = int(sweep["games_per_level"])
    shape = tuple(sweep["shape"])
    scale = float(sweep["scale"])

    alphas, r_values = [], []
    sweep_key = jax.random.PRNGKey(seed + 1)
    for level_idx, a in enumerate(levels):
        for g_idx in range(per_level):
            k = jax.random.fold_in(jax.random.fold_in(sweep_key, level_idx), g_idx)
            pot_src, harm_src = _random_mixed_game(k, shape, scale)
            try:
                game = make_family(pot_src, harm_src, [a], scale=scale)[0]
            except ValueError:
                continue  # degenerate zero-norm component draw
            alphas.append(a)
            r_values.append(_solve_r(game, lam, cfg))
    x = jnp.asarray(alphas)
    y = jnp.asarray(r_values)
    rho = _spearman(x, y)
    n = len(alphas)
    # Large-sample t approximation for Spearman under H0.
    t_stat = rho * ((n - 2) / max(1e-12, 1.0 - rho**2)) ** 0.5
    p_upper_bound = 1e-12 if t_stat > 8.0 else 0.05  # t>8 at n≈2000: p ≪ 1e-12

    boot_key = jax.random.PRNGKey(seed + 2)
    boot = []
    for b in range(int(cfg["bootstrap"]["n_resamples"])):
        idx = jax.random.randint(jax.random.fold_in(boot_key, b), (n,), 0, n)
        boot.append(_spearman(x[idx], y[idx]))
    boot_arr = jnp.asarray(boot)
    ci_low = float(jnp.quantile(boot_arr, 0.025))
    ci_high = float(jnp.quantile(boot_arr, 0.975))

    passed = rho > 0.9 and p_upper_bound < 0.01
    failures += not passed
    _write(
        BenchmarkResult(
            benchmark_id="reciprocity_alpha_sweep",
            unit=UNIT,
            kind="statistical",
            passed=passed,
            metrics={"spearman_rho": rho, "n_pairs": float(n), "t_stat": float(t_stat)},
            effect_sizes=[
                EffectSize(
                    name="spearman_rho_R_alpha",
                    value=rho,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    method="bootstrap (2000 resamples, percentile)",
                )
            ],
            n=n,
            n_justification=(
                f"{per_level} games x {len(levels)} alpha levels: pilot runs put the "
                "bootstrap CI half-width on Spearman rho below 0.01 at this n, an "
                "order of magnitude tighter than the 0.9 gate threshold margin."
            ),
            seed=seed,
            config_ref=str(CONFIG.relative_to(REPO)),
            library_version=strataq.__version__,
            timestamp=_now(),
            notes="Gate: Spearman rho(R, alpha) > 0.9, p < 0.01.",
        )
    )

    # ---- 4. oracle_agreement (chi_eq vs finite differences) ----------------
    fd_cfg = cfg["fd_check"]
    fd_key = jax.random.PRNGKey(seed + 3)
    worst = 0.0
    for g_idx in range(int(fd_cfg["n_games"])):
        k1, k2 = jax.random.split(jax.random.fold_in(fd_key, g_idx))
        game = DenseTensorGame(
            (
                float(fd_cfg["payoff_scale"]) * jax.random.normal(k1, tuple(fd_cfg["shape"])),
                float(fd_cfg["payoff_scale"]) * jax.random.normal(k2, tuple(fd_cfg["shape"])),
            )
        )
        point = logit_qre(game, lam, tol=1e-13, max_iter=200_000)
        resp = chi_equilibrium(game, point)
        gap = float(jnp.max(jnp.abs(resp.chi_full - chi_fd(game, lam))))
        worst = max(worst, gap)
    passed = worst < 1e-6
    failures += not passed
    _write(
        BenchmarkResult(
            benchmark_id="chi_fd_agreement",
            unit=UNIT,
            kind="correctness",
            passed=passed,
            metrics={"max_abs_gap": worst, "n_games": float(fd_cfg["n_games"])},
            seed=seed,
            config_ref=str(CONFIG.relative_to(REPO)),
            library_version=strataq.__version__,
            timestamp=_now(),
            notes="chi_eq must match central finite differences of logit_qre to 1e-6.",
        )
    )

    # ---- 5. spectrum_reality (N3 numerical leg) ----------------------------
    lam_grid = [0.5, 1.0, 2.0, 5.0, 10.0]
    max_rel_imag_potential = 0.0
    min_abs_imag_harmonic = float("inf")
    for _name, game in potential_games:
        for lam_val in lam_grid:
            point = logit_qre(
                game, lam_val, tol=cfg["solver"]["tol"], max_iter=cfg["solver"]["max_iter"]
            )
            ops = build_operators(game, point)
            eigs = jnp.linalg.eigvals(ops.s_tangent @ ops.b_tangent)
            scale = max(float(jnp.max(jnp.abs(eigs))), 1e-30)
            max_rel_imag_potential = max(
                max_rel_imag_potential, float(jnp.max(jnp.abs(jnp.imag(eigs)))) / scale
            )
    for _name, game in harmonic_games:
        for lam_val in lam_grid:
            point = logit_qre(
                game, lam_val, tol=cfg["solver"]["tol"], max_iter=cfg["solver"]["max_iter"]
            )
            ops = build_operators(game, point)
            eigs = jnp.linalg.eigvals(ops.s_tangent @ ops.b_tangent)
            min_abs_imag_harmonic = min(
                min_abs_imag_harmonic, float(jnp.max(jnp.abs(jnp.imag(eigs))))
            )
    passed = max_rel_imag_potential < 1e-8 and min_abs_imag_harmonic > 1e-3
    failures += not passed
    _write(
        BenchmarkResult(
            benchmark_id="spectrum_reality",
            unit=UNIT,
            kind="correctness",
            passed=passed,
            metrics={
                "max_rel_imag_potential": max_rel_imag_potential,
                "min_abs_imag_harmonic": min_abs_imag_harmonic,
                "lam_grid_max": max(lam_grid),
            },
            seed=seed,
            config_ref=str(CONFIG.relative_to(REPO)),
            library_version=strataq.__version__,
            timestamp=_now(),
            notes=(
                "N3 numerical leg: SB spectrum effectively real (rel. imag < 1e-8) on all "
                "potential games across lambda in {0.5..10}; visibly complex on harmonic "
                "games. A complex pair on a potential game would falsify N3 (claims ledger)."
            ),
        )
    )
    return failures


if __name__ == "__main__":
    raise SystemExit(run())
