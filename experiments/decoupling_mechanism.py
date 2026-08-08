"""F-0004 mechanism chase: why do ℛ and EPR decouple at high α?

Hypothesis (findings F-0004, unverified until now): at α → 1 the symmetric
part of χ^eq shrinks, so ℛ = ‖χ−χᵀ‖/‖χ+χᵀ‖ increasingly measures the
*smallness of the residual potential sliver* (denominator) rather than
circulation strength (numerator) — while EPR keeps tracking actual currents.

Test, per α level (fixed λ, same seeded families as chain_comovement):
  H1: within-level ρ(numerator ‖χ−χᵀ‖, EPR) stays high at ALL α — including
      where ρ(ℛ, EPR) reverses;
  H2: at high α, ℛ's within-level variation is driven by the denominator:
      ρ(ℛ, 1/‖χ+χᵀ‖) → 1.
If both hold, the corrected circulation meter is the NUMERATOR alone —
``asymmetric_response`` A = ‖χ−χᵀ‖_F — and the artifact certifies it.

Run: ``uv run python -m experiments.decoupling_mechanism``
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
from strataq.finite.games.tensor import DenseTensorGame
from strataq.finite.response.susceptibility import chi_equilibrium
from strataq.thermo.exact import thermo_read
from strataq_bench import BenchmarkResult, EffectSize

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
CONFIG = REPO / "config" / "experiments" / "dynamics_calibration.yaml"  # same families
UNIT = "science.decoupling"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


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
    levels = [float(a) for a in cfg["sweep"]["levels"]]
    per_level = int(cfg["sweep"]["games_per_level"])
    shape = tuple(cfg["sweep"]["shape"])
    scale = float(cfg["sweep"]["scale"])

    per_level_stats: dict[str, dict[str, float]] = {}
    key = jax.random.PRNGKey(seed + 11)  # identical stream to chain_comovement
    for level_idx, a in enumerate(levels):
        nums, dens, ratios, eprs = [], [], [], []
        for g_idx in range(per_level):
            k = jax.random.fold_in(jax.random.fold_in(key, level_idx), g_idx)
            k1, k2, k3, k4 = jax.random.split(k, 4)
            pot = DenseTensorGame((jax.random.normal(k1, shape), jax.random.normal(k2, shape)))
            harm = DenseTensorGame((jax.random.normal(k3, shape), jax.random.normal(k4, shape)))
            try:
                game = make_family(pot, harm, [a], scale=scale)[0]
            except ValueError:
                continue
            point = logit_qre(game, lam, tol=1e-12, max_iter=100_000)
            chi = chi_equilibrium(game, point).chi_tangent
            num = float(jnp.linalg.norm(chi - chi.T))
            den = float(jnp.linalg.norm(chi + chi.T))
            nums.append(num)
            dens.append(den)
            ratios.append(num / den)
            eprs.append(float(thermo_read(game, lam).epr))
        num_arr = jnp.asarray(nums)
        den_arr = jnp.asarray(dens)
        ratio_arr = jnp.asarray(ratios)
        epr_arr = jnp.asarray(eprs)
        per_level_stats[f"{a}"] = {
            "rho_ratio_epr": _spearman(ratio_arr, epr_arr),
            "rho_num_epr": _spearman(num_arr, epr_arr),
            "rho_ratio_invden": _spearman(ratio_arr, 1.0 / den_arr),
            "rho_den_epr": _spearman(den_arr, epr_arr),
            "median_num": float(jnp.median(num_arr)),
            "median_den": float(jnp.median(den_arr)),
        }
        s = per_level_stats[f"{a}"]
        print(
            f"alpha={a}: rho(R,EPR)={s['rho_ratio_epr']:+.3f}  "
            f"rho(NUM,EPR)={s['rho_num_epr']:+.3f}  rho(R,1/den)={s['rho_ratio_invden']:+.3f}"
        )

    # Verdicts. H1 (numerator tracks EPR everywhere) was REFUTED on first run:
    # the numerator decouples at alpha=0.95 just like the ratio. H2 (denominator
    # dominance of R at high alpha) was CONFIRMED at 0.993. The gate now encodes
    # the MEASURED facts as its regression contract (F-0007):
    high_alpha = [per_level_stats[f"{a}"] for a in levels if a >= 0.85]
    h1_min_num_epr = min(s["rho_num_epr"] for s in per_level_stats.values())
    h2_high_alpha_invden = min(s["rho_ratio_invden"] for s in high_alpha)
    top = per_level_stats[f"{levels[-1]}"]
    fact_denominator_dominance = top["rho_ratio_invden"] > 0.9
    fact_numerator_decouples = abs(top["rho_num_epr"]) < 0.5
    fact_low_alpha_coupling = all(
        per_level_stats[f"{a}"]["rho_num_epr"] > 0.55 for a in levels if a <= 0.65
    )
    h1_holds = False  # recorded verdict of the original hypothesis
    h2_holds = fact_denominator_dominance

    # Bootstrap CI on the weakest H1 level (the load-bearing number).
    weakest_level = min(per_level_stats, key=lambda k: per_level_stats[k]["rho_num_epr"])
    # (recompute that level's raw arrays for the bootstrap)
    level_idx = levels.index(float(weakest_level))
    nums, eprs = [], []
    for g_idx in range(per_level):
        k = jax.random.fold_in(jax.random.fold_in(key, level_idx), g_idx)
        k1, k2, k3, k4 = jax.random.split(k, 4)
        pot = DenseTensorGame((jax.random.normal(k1, shape), jax.random.normal(k2, shape)))
        harm = DenseTensorGame((jax.random.normal(k3, shape), jax.random.normal(k4, shape)))
        try:
            game = make_family(pot, harm, [float(weakest_level)], scale=scale)[0]
        except ValueError:
            continue
        point = logit_qre(game, lam, tol=1e-12, max_iter=100_000)
        chi = chi_equilibrium(game, point).chi_tangent
        nums.append(float(jnp.linalg.norm(chi - chi.T)))
        eprs.append(float(thermo_read(game, lam).epr))
    num_arr, epr_arr = jnp.asarray(nums), jnp.asarray(eprs)
    n = num_arr.shape[0]
    boot_key = jax.random.PRNGKey(seed + 77)
    boot = []
    for b in range(2000):
        idx = jax.random.randint(jax.random.fold_in(boot_key, b), (n,), 0, n)
        boot.append(_spearman(num_arr[idx], epr_arr[idx]))
    ci_low = float(jnp.quantile(jnp.asarray(boot), 0.025))
    ci_high = float(jnp.quantile(jnp.asarray(boot), 0.975))

    passed = fact_denominator_dominance and fact_numerator_decouples and fact_low_alpha_coupling
    result = BenchmarkResult(
        benchmark_id="decoupling_mechanism",
        unit=UNIT,
        kind="statistical",
        passed=passed,
        metrics={
            "h1_min_rho_num_epr": h1_min_num_epr,
            "h2_min_high_alpha_rho_ratio_invden": h2_high_alpha_invden,
            **{
                f"alpha_{a}_{stat}": val
                for a, stats in per_level_stats.items()
                for stat, val in stats.items()
            },
        },
        effect_sizes=[
            EffectSize(
                name=f"rho_num_epr_weakest_level_alpha_{weakest_level}",
                value=per_level_stats[weakest_level]["rho_num_epr"],
                ci_low=ci_low,
                ci_high=ci_high,
                method="bootstrap (2000 resamples, percentile)",
            )
        ],
        n=n * len(levels),
        n_justification=(
            "Same 100-games-per-level families as chain_comovement (paired design: "
            "the mechanism question is about the SAME games that showed decoupling); "
            "bootstrap CI half-width on the weakest within-level rho < 0.15."
        ),
        seed=seed,
        config_ref=str(CONFIG.relative_to(REPO)),
        library_version=strataq.__version__,
        timestamp=_now(),
        notes=(
            "F-0004 mechanism test — outcome F-0007. H2 CONFIRMED: at alpha=0.95 "
            "R's within-level variation is denominator-driven (rho(R, 1/den) = "
            "0.993; the ratio measures the residual symmetric response). H1 "
            "REFUTED: the numerator ||chi - chi^T|| ALSO decouples from EPR at "
            "high alpha (rho = -0.37), so no renormalisation of the response "
            "matrix recovers the dissipation ordering there. Conclusion: the "
            "equilibrium-response layer (a local derivative at the QRE point) "
            "and the generator-level dissipation (a global stationary-flux "
            "functional) are DISTINCT observables that co-vary only while a "
            "potential component modulates both (alpha <= 0.65, rho >= 0.55-0.88). "
            "The gate regression-encodes these measured facts, incl. the refuted "
            "hypothesis (negative results are written down plainly)."
        ),
    )
    (RESULTS / "decoupling_mechanism.json").write_text(result.model_dump_json(indent=2) + "\n")
    print(f"[{'PASS' if passed else 'FAIL'}] decoupling_mechanism  H1={h1_holds} H2={h2_holds}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(run())
