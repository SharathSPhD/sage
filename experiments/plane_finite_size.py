"""R11 — the finite-size kill-shot on the two-axis (ℛ, EPR) claim.

The programme's flagship claim is that response asymmetry ℛ and dissipation
EPR share a zero but are otherwise INDEPENDENT coordinates. The entire
evidential base for the "otherwise independent" half (F-0004, F-0007, F-0010)
is the within-α-level collapse of ρ(EPR, ℛ) — and every reading of it is at
N=2 players, m=3 actions. The single most likely way the claim is wrong is
that the collapse is a FINITE-SIZE ARTEFACT of a 4-dimensional tangent space
and a 9-state chain, weakening as m grows.

This experiment sweeps m ∈ {3, 4, 5, 6} at fixed N=2 and fixed λ, with the same
100-games-per-level design as F-0004 (so the m=3 row is a replication on a
fresh seed stream), and adjudicates the criteria T1–T4 registered in
``config/experiments/plane_finite_size.yaml`` BEFORE this file existed
(commit a6533e7). A registered scale control D1 repeats the two extreme α
levels under a constant per-entry-payoff-RMS rule, because the fixed-norm
family generator makes larger m mean a colder game.

Run: ``uv run python -m experiments.plane_finite_size``
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import strataq
import yaml
from strataq.core.solve.fixedpoint import logit_qre
from strataq.finite.decompose.generate import make_family
from strataq.finite.games.tensor import DenseTensorGame
from strataq.finite.response.reciprocity import reciprocity_defect_of
from strataq.finite.response.susceptibility import chi_equilibrium
from strataq.thermo.exact import thermo_read
from strataq_bench import BenchmarkResult, EffectSize

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
CONFIG = REPO / "config" / "experiments" / "plane_finite_size.yaml"
UNIT = "science.plane"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _spearman(x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Rank correlation, same helper as ``experiments/decoupling_mechanism.py``."""

    def ranks(v: jnp.ndarray) -> jnp.ndarray:
        order = jnp.argsort(v)
        return jnp.zeros_like(v).at[order].set(jnp.arange(v.shape[0], dtype=v.dtype))

    rx = ranks(x) - (x.shape[0] - 1) / 2.0
    ry = ranks(y) - (y.shape[0] - 1) / 2.0
    return float((rx @ ry) / jnp.sqrt((rx @ rx) * (ry @ ry)))


def _cell(
    key: jnp.ndarray, *, m: int, alpha: float, n_games: int, scale: float, lam: float
) -> dict[str, Any]:
    """One (m, α) cell: n_games seeded games → ℛ, EPR and the registered diagnostics.

    PRNG is threaded explicitly: ``key`` is already folded on (m, α); each game
    folds in its own index, so a cell's draws are reproducible independently of
    the loop order around it.
    """
    shape = (m, m)
    rs: list[float] = []
    eprs: list[float] = []
    dists: list[float] = []
    ranges: list[float] = []
    lam_norms: list[float] = []
    near_critical = 0
    non_converged = 0
    for g_idx in range(n_games):
        k = jax.random.fold_in(key, g_idx)
        k1, k2, k3, k4 = jax.random.split(k, 4)
        pot = DenseTensorGame((jax.random.normal(k1, shape), jax.random.normal(k2, shape)))
        harm = DenseTensorGame((jax.random.normal(k3, shape), jax.random.normal(k4, shape)))
        game = make_family(pot, harm, [alpha], scale=scale)[0]
        point = logit_qre(game, lam)
        non_converged += int(not bool(point.converged))
        resp = chi_equilibrium(game, point)
        near_critical += int(bool(resp.near_critical))
        dists.append(float(resp.distance_to_criticality))
        rs.append(float(reciprocity_defect_of(resp.chi_tangent)))
        eprs.append(float(thermo_read(game, lam).epr))
        ranges.append(float(point.payoff_range))
        lam_norms.append(float(point.lambda_normalised[0]))
    r_arr = jnp.asarray(rs)
    epr_arr = jnp.asarray(eprs)
    return {
        "r": r_arr,
        "epr": epr_arr,
        "rho": _spearman(epr_arr, r_arr),
        "n": int(r_arr.shape[0]),
        "near_critical_frac": near_critical / n_games,
        "non_converged": non_converged,
        "median_distance_to_criticality": float(jnp.median(jnp.asarray(dists))),
        "median_payoff_range": float(jnp.median(jnp.asarray(ranges))),
        "median_lambda_normalised": float(jnp.median(jnp.asarray(lam_norms))),
        "median_epr": float(jnp.median(epr_arr)),
        "median_r": float(jnp.median(r_arr)),
    }


def _bootstrap_ci(
    x: jnp.ndarray, y: jnp.ndarray, *, key: jnp.ndarray, n_resamples: int, ci_level: float
) -> tuple[float, float]:
    """Percentile bootstrap CI on ρ(x, y), resampling games within the level (T3)."""
    n = x.shape[0]
    boot = []
    for b in range(n_resamples):
        idx = jax.random.randint(jax.random.fold_in(key, b), (n,), 0, n)
        boot.append(_spearman(x[idx], y[idx]))
    tail = (1.0 - ci_level) / 2.0
    arr = jnp.asarray(boot)
    return float(jnp.quantile(arr, tail)), float(jnp.quantile(arr, 1.0 - tail))


def _arm(
    *,
    root_key: jnp.ndarray,
    m_values: list[int],
    levels: list[float],
    n_games: int,
    lam: float,
    scale_of_m: dict[int, float],
    label: str,
) -> dict[int, dict[float, dict[str, Any]]]:
    """Run every (m, α) cell of one arm. ``scale_of_m`` carries the D1 rule."""
    cells: dict[int, dict[float, dict[str, Any]]] = {}
    for m_idx, m in enumerate(m_values):
        cells[m] = {}
        for level_idx, alpha in enumerate(levels):
            key = jax.random.fold_in(jax.random.fold_in(root_key, m_idx), level_idx)
            cell = _cell(
                key,
                m=m,
                alpha=alpha,
                n_games=n_games,
                scale=scale_of_m[m],
                lam=lam,
            )
            cells[m][alpha] = cell
            print(
                f"[{label}] m={m} alpha={alpha:.2f}: rho(EPR,R)={cell['rho']:+.3f}  "
                f"near_crit={cell['near_critical_frac']:.2f}  "
                f"lam_norm={cell['median_lambda_normalised']:.2f}"
            )
    return cells


def _adjudicate(
    cells: dict[int, dict[float, dict[str, Any]]],
    cis: dict[int, tuple[float, float]],
    crit: dict[str, Any],
    max_near_critical_frac: float,
    m_values: list[int],
) -> dict[str, Any]:
    """Apply the registered T1–T4 exactly as written in the config."""
    a_lo = float(crit["alpha_lo"])
    a_hi = float(crit["alpha_hi"])
    rho_hi = [cells[m][a_hi]["rho"] for m in m_values]
    rho_lo = [cells[m][a_lo]["rho"] for m in m_values]
    collapse = [h - low for h, low in zip(rho_hi, rho_lo, strict=True)]
    delta = rho_hi[-1] - rho_hi[0]

    mono = all(rho_hi[k + 1] >= rho_hi[k] - float(crit["mono_tol"]) for k in range(len(rho_hi) - 1))
    ci_low = [cis[m][0] for m in m_values]
    ci_high = [cis[m][1] for m in m_values]
    ends_disjoint = ci_low[-1] > ci_high[0]
    all_overlap = max(ci_low) <= min(ci_high)

    ceiling = float(crit["survive_ceiling"])
    t1_refute = mono and delta > float(crit["recovery_delta_refute"]) and ends_disjoint
    t1_survive = (
        all(r <= ceiling for r in rho_hi)
        and delta <= float(crit["recovery_delta_survive"])
        and all(h < ceiling for h in ci_high)
    )

    baseline_ok = all(r >= float(crit["baseline_rho_min"]) for r in rho_lo)
    t2 = all(c <= float(crit["collapse_drop_max"]) for c in collapse) and baseline_ok

    guard_fired = any(
        cell["near_critical_frac"] > max_near_critical_frac or cell["non_converged"] > 0
        for per_m in cells.values()
        for cell in per_m.values()
    )
    t4 = {
        "i_nonmonotone_with_drift": (not mono)
        and abs(delta) > float(crit["recovery_delta_survive"]),
        "ii_all_cis_overlap_and_no_ceiling": all_overlap and not all(h < ceiling for h in ci_high),
        "iii_baseline_absent": not baseline_ok,
        "iv_diagnostic_guard": guard_fired,
    }
    indeterminate = any(t4.values())

    if indeterminate:
        verdict = "INDETERMINATE"
    elif t1_refute:
        verdict = "REFUTED"
    elif t1_survive and t2:
        verdict = "SURVIVES"
    else:
        verdict = "INDETERMINATE"
    return {
        "verdict": verdict,
        "rho_hi": rho_hi,
        "rho_lo": rho_lo,
        "collapse": collapse,
        "delta": delta,
        "mono": mono,
        "ends_disjoint": ends_disjoint,
        "all_overlap": all_overlap,
        "t1_refute": t1_refute,
        "t1_survive": t1_survive,
        "t2": t2,
        "t2_baseline_ok": baseline_ok,
        "t4": t4,
    }


def run() -> int:
    cfg = yaml.safe_load(CONFIG.read_text())
    seed = int(cfg["seed"])
    lam = float(cfg["lam"])
    levels = [float(a) for a in cfg["sweep"]["levels"]]
    per_level = int(cfg["sweep"]["games_per_level"])
    m_values = [int(m) for m in cfg["sweep"]["m_values"]]
    scale = float(cfg["sweep"]["scale"])
    crit = cfg["criteria"]
    boot_cfg = cfg["bootstrap"]
    n_resamples = int(boot_cfg["n_resamples"])
    ci_level = float(boot_cfg["ci_level"])
    max_near = float(cfg["diagnostics"]["max_near_critical_frac"])
    sc_cfg = cfg["scale_control"]
    a_lo = float(crit["alpha_lo"])
    a_hi = float(crit["alpha_hi"])

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "plane_finite_size.resolved.yaml").write_text(
        yaml.safe_dump(
            {"config": cfg, "library_version": strataq.__version__, "run_at": _now()},
            sort_keys=False,
        )
    )

    # --- main arm: fixed scale (house precedent, comparable to F-0004/F-0010) ---
    main = _arm(
        root_key=jax.random.PRNGKey(seed),
        m_values=m_values,
        levels=levels,
        n_games=per_level,
        lam=lam,
        scale_of_m=dict.fromkeys(m_values, scale),
        label="main",
    )

    # --- T3: percentile bootstrap CI on rho_hi(m), explicit seed stream ---
    boot_key = jax.random.PRNGKey(seed + int(boot_cfg["seed_offset"]))
    cis: dict[int, tuple[float, float]] = {}
    for m_idx, m in enumerate(m_values):
        cell = main[m][a_hi]
        cis[m] = _bootstrap_ci(
            cell["epr"],
            cell["r"],
            key=jax.random.fold_in(boot_key, m_idx),
            n_resamples=n_resamples,
            ci_level=ci_level,
        )
        print(f"[T3] m={m}: rho_hi={cell['rho']:+.3f}  CI=[{cis[m][0]:+.3f}, {cis[m][1]:+.3f}]")

    verdict_main = _adjudicate(main, cis, crit, max_near, m_values)

    # --- D1: registered scale control, two extreme levels, constant per-entry RMS ---
    d1_levels = [float(a) for a in sc_cfg["levels"]]
    m_min = min(m_values)
    d1_scale_of_m = {m: scale * m / m_min for m in m_values}
    d1 = _arm(
        root_key=jax.random.PRNGKey(seed + int(sc_cfg["seed_offset"])),
        m_values=m_values,
        levels=d1_levels,
        n_games=per_level,
        lam=lam,
        scale_of_m=d1_scale_of_m,
        label="D1",
    )
    d1_key = jax.random.PRNGKey(seed + int(sc_cfg["seed_offset"]) + int(boot_cfg["seed_offset"]))
    d1_cis: dict[int, tuple[float, float]] = {}
    for m_idx, m in enumerate(m_values):
        cell = d1[m][a_hi]
        d1_cis[m] = _bootstrap_ci(
            cell["epr"],
            cell["r"],
            key=jax.random.fold_in(d1_key, m_idx),
            n_resamples=n_resamples,
            ci_level=ci_level,
        )
    verdict_d1 = _adjudicate(d1, d1_cis, crit, max_near, m_values)

    # Registered in the config BEFORE the run: if the two arms disagree in
    # verdict, the unit is INDETERMINATE and the disagreement is reported.
    arms_agree = verdict_d1["verdict"] == verdict_main["verdict"]
    verdict = verdict_main["verdict"] if arms_agree else "INDETERMINATE"

    metrics: dict[str, float] = {
        "verdict_survives": float(verdict == "SURVIVES"),
        "verdict_refuted": float(verdict == "REFUTED"),
        "verdict_indeterminate": float(verdict == "INDETERMINATE"),
        "t1_refute_main": float(verdict_main["t1_refute"]),
        "t1_survive_main": float(verdict_main["t1_survive"]),
        "t1_monotone_main": float(verdict_main["mono"]),
        "t1_delta_rho_hi_main": verdict_main["delta"],
        "t2_main": float(verdict_main["t2"]),
        "t2_baseline_ok_main": float(verdict_main["t2_baseline_ok"]),
        "t3_ends_disjoint_main": float(verdict_main["ends_disjoint"]),
        "t3_all_cis_overlap_main": float(verdict_main["all_overlap"]),
        "t4_i_nonmonotone_with_drift": float(verdict_main["t4"]["i_nonmonotone_with_drift"]),
        "t4_ii_all_cis_overlap_and_no_ceiling": float(
            verdict_main["t4"]["ii_all_cis_overlap_and_no_ceiling"]
        ),
        "t4_iii_baseline_absent": float(verdict_main["t4"]["iii_baseline_absent"]),
        "t4_iv_diagnostic_guard": float(verdict_main["t4"]["iv_diagnostic_guard"]),
        "d1_arms_agree": float(arms_agree),
        "d1_t1_refute": float(verdict_d1["t1_refute"]),
        "d1_t1_survive": float(verdict_d1["t1_survive"]),
        "d1_t2": float(verdict_d1["t2"]),
        "d1_delta_rho_hi": verdict_d1["delta"],
    }
    for arm_name, arm_cells in (("main", main), ("d1", d1)):
        for m, per_m in arm_cells.items():
            for alpha, cell in per_m.items():
                pre = f"{arm_name}_m{m}_a{alpha:.2f}"
                metrics[f"{pre}_rho_epr_r"] = cell["rho"]
                metrics[f"{pre}_near_critical_frac"] = cell["near_critical_frac"]
                metrics[f"{pre}_non_converged"] = float(cell["non_converged"])
                metrics[f"{pre}_median_dist_crit"] = cell["median_distance_to_criticality"]
                metrics[f"{pre}_median_lambda_normalised"] = cell["median_lambda_normalised"]
                metrics[f"{pre}_median_payoff_range"] = cell["median_payoff_range"]
                metrics[f"{pre}_median_epr"] = cell["median_epr"]
                metrics[f"{pre}_median_r"] = cell["median_r"]
    for m, coll in zip(m_values, verdict_main["collapse"], strict=True):
        metrics[f"main_collapse_m{m}"] = coll
    for m, coll in zip(m_values, verdict_d1["collapse"], strict=True):
        metrics[f"d1_collapse_m{m}"] = coll

    effects = [
        EffectSize(
            name=f"rho_hi_main_m{m}",
            value=main[m][a_hi]["rho"],
            ci_low=cis[m][0],
            ci_high=cis[m][1],
            ci_level=ci_level,
            method=f"percentile bootstrap ({n_resamples} resamples over games within level)",
        )
        for m in m_values
    ] + [
        EffectSize(
            name=f"rho_hi_scale_control_m{m}",
            value=d1[m][a_hi]["rho"],
            ci_low=d1_cis[m][0],
            ci_high=d1_cis[m][1],
            ci_level=ci_level,
            method=f"percentile bootstrap ({n_resamples} resamples over games within level)",
        )
        for m in m_values
    ]

    n_total = per_level * len(levels) * len(m_values) + per_level * len(d1_levels) * len(m_values)
    passed = verdict == "SURVIVES"
    result = BenchmarkResult(
        benchmark_id="plane_finite_size",
        unit=UNIT,
        kind="statistical",
        passed=passed,
        metrics=metrics,
        effect_sizes=effects,
        n=n_total,
        n_justification=" ".join(str(cfg["n_justification"]).split()),
        seed=seed,
        config_ref=str(CONFIG.relative_to(REPO)),
        library_version=strataq.__version__,
        timestamp=_now(),
        notes=(
            f"R11 finite-size kill-shot on the two-axis (R, EPR) claim. VERDICT: {verdict}. "
            f"Criteria T1-T4 registered in {CONFIG.name} at commit a6533e7, BEFORE this "
            "experiment file existed; nothing in the config was amended after the run and "
            "the design was executed exactly as registered (100 games per cell, m in "
            f"{m_values}, 10 alpha levels, lam={lam}). rho_hi(m) [alpha={a_hi}] = "
            + ", ".join(
                f"m={m}: {main[m][a_hi]['rho']:+.3f} [{cis[m][0]:+.3f}, {cis[m][1]:+.3f}]"
                for m in m_values
            )
            + "; rho_lo(m) [alpha="
            + f"{a_lo}] = "
            + ", ".join(f"m={m}: {main[m][a_lo]['rho']:+.3f}" for m in m_values)
            + f". T1 monotone-non-decreasing={verdict_main['mono']}, "
            f"delta=rho_hi(m_max)-rho_hi(m_min)={verdict_main['delta']:+.3f}, "
            f"T1_refute={verdict_main['t1_refute']}, T1_survive={verdict_main['t1_survive']}; "
            f"T2={verdict_main['t2']}; T3 end-CIs disjoint={verdict_main['ends_disjoint']}, "
            f"all CIs overlap={verdict_main['all_overlap']}; T4 triggers={verdict_main['t4']}. "
            f"Registered scale control D1 (constant per-entry payoff RMS, scale(m)=scale*m/"
            f"{m_min}) verdict={verdict_d1['verdict']}, arms_agree={arms_agree}. "
            "Scope, stated as registered: N=2 throughout, m<=6, single lambda — this is a "
            "test of m-scaling only and says nothing about N-scaling."
        ),
    )
    (RESULTS / "plane_finite_size.json").write_text(result.model_dump_json(indent=2) + "\n")
    print(f"[{'PASS' if passed else 'FAIL'}] plane_finite_size  VERDICT={verdict}")
    print(f"  main rho_hi: {[round(r, 3) for r in verdict_main['rho_hi']]}")
    print(f"  main rho_lo: {[round(r, 3) for r in verdict_main['rho_lo']]}")
    print(f"  D1   rho_hi: {[round(r, 3) for r in verdict_d1['rho_hi']]}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(run())
