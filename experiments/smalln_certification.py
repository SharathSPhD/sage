"""R8 — can a month of market data be quoted? (unit thermo.hs_estimator.smalln)

Run: ``uv run python -m experiments.smalln_certification``
Registered S1–S3 in config/experiments/smalln_certification.yaml (commit
verified landed before this ran). The unit EXTENDS the certification to the
smallest n that passes all three criteria, or publishes the refusal
boundary — both are acceptable outcomes and neither is pre-declared.
Artifact: ``smalln_certification.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import strataq
import yaml
from strataq.core.dynamics.markov import glauber_generator
from strataq.finite.decompose.generate import make_family
from strataq.finite.games.library import coordination, matching_pennies
from strataq.thermo.hs_estimator import HSEstimate, hs_y_estimate, sample_quench_states
from strataq.thermo.protocols import QuenchProtocol, hatano_sasa_exact
from strataq_bench import BenchmarkResult, EffectSize


@dataclass(frozen=True)
class Cell:
    """One (n, interval-method) screening cell."""

    cover: int
    agree: int
    flips: int
    width_ratio: float  # median CI half-width / bootstrap SE (S1b)
    seeds: int


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
CONFIG = REPO / "config" / "experiments" / "smalln_certification.yaml"
UNIT = "thermo.hs_estimator.smalln"


def main() -> int:
    cfg: dict[str, Any] = yaml.safe_load(CONFIG.read_text())
    seed = int(cfg["seed"])
    (RESULTS / "smalln_certification.resolved.yaml").write_text(yaml.safe_dump(cfg))

    game = make_family(
        coordination(2, 2, bonus=2.0), matching_pennies(), [float(cfg["family"]["alpha"])]
    )[0]
    lams = jnp.array([float(x) for x in cfg["protocol"]["lambdas"]])
    tau = float(cfg["protocol"]["tau"])
    proto = QuenchProtocol(lambdas=lams, taus=jnp.full((len(lams) - 1,), tau))
    exact = float(hatano_sasa_exact(game, proto)[1])
    n_seeds = int(cfg["n_seeds"])
    methods = list(cfg["interval_candidates"])
    rng = np.random.default_rng(seed)

    def windows_for(n: int, s: int) -> list[np.ndarray]:
        return sample_quench_states(
            game,
            proto,
            n_trajectories=n,
            steps_per_unit_time=int(cfg["steps_per_unit_time"]),
            seed=seed + 97 * s + n,
        )

    def est_for(w: list[np.ndarray], method: str) -> HSEstimate:
        return hs_y_estimate(
            w,
            n_states=4,
            hold_durations=[tau] * len(w),
            pseudocount=float(cfg["pseudocount"]),
            relax_safety=float(cfg["relax_safety"]),
            interval_method=method,
        )

    # reference decisions at the certified n
    ref_admit = []
    for s in range(n_seeds):
        ref_admit.append(est_for(windows_for(int(cfg["n_reference"]), s), "percentile").usable)

    max_w = float(cfg["max_width_times_se"])

    def run_cell(n: int, method: str, seed_offset: int = 0, seeds: int | None = None) -> Cell:
        """One (n, method) cell: coverage, gate agreement, flag flips, and the
        median half-width / bootstrap-SE ratio (S1b)."""
        n_s = seeds if seeds is not None else n_seeds
        cover = agree = flips = 0
        ratios: list[float] = []
        for s in range(n_s):
            w = windows_for(n, s + seed_offset)
            e = est_for(w, method)
            cover += int(e.mean_y_ci_low <= exact <= e.mean_y_ci_high)
            if seed_offset == 0 and s < len(ref_admit):
                agree += int(e.usable == ref_admit[s])
            half = 0.5 * (e.mean_y_ci_high - e.mean_y_ci_low)
            ratios.append(half / e.boot_se if e.boot_se > 0 else float("inf"))
            # S3: trajectory-order permutation (null for the physics; see the
            # config's S3_interpretation — it also stirs the 4-way SE split)
            perm = rng.permutation(n)
            e_shuf = est_for([x[perm] for x in w], method)
            flips += int(e_shuf.usable != e.usable)
        return Cell(cover, agree, flips, float(np.median(ratios)), n_s)

    metrics: dict[str, float] = {"exact_mean_y": exact}
    results: dict[tuple[int, str], Cell] = {}
    for n in [int(x) for x in cfg["n_grid"]]:
        for method in methods:
            c = run_cell(n, method)
            results[(n, method)] = c
            key = f"n{n}_{method}"
            metrics[f"{key}_coverage"] = float(c.cover)
            metrics[f"{key}_gate_agreement"] = float(c.agree)
            metrics[f"{key}_flag_flips"] = float(c.flips)
            metrics[f"{key}_width_over_se"] = c.width_ratio

    # ---- C-1 / C-2 diagnostics mandated by the results red-team ----------
    diag = cfg["diagnostics"]
    d_method = str(diag["method"])
    # TRUE settling status from the generator's exact gaps: no sampling noise,
    # so a disagreement cannot be blamed on the finite-sample reference.
    gaps = []
    for lam in [float(x) for x in cfg["protocol"]["lambdas"]][1:]:
        ev = np.linalg.eigvals(np.asarray(glauber_generator(game, lam)))
        gaps.append(float(-np.sort(ev.real)[::-1][1]))
    true_settled = all(tau >= float(cfg["relax_safety"]) / g for g in gaps)
    metrics["true_settled_primary"] = float(true_settled)
    metrics["true_max_relax_time"] = float(max(1.0 / g for g in gaps))

    def within_group_perm(n: int, rng_local: np.random.Generator) -> np.ndarray:
        """Permute only INSIDE each i::4 residue class, so the SE split's
        composition is preserved while the permutation stays physically null."""
        idx = np.arange(n)
        out = idx.copy()
        for r in range(4):
            grp = idx[r::4]
            out[r::4] = rng_local.permutation(grp)
        return out

    drng = np.random.default_rng(int(cfg["seed"]) + 31337)
    for n in [int(x) for x in diag["n_grid"]]:
        agree_true = flips_within = 0
        for s in range(n_seeds):
            w = windows_for(n, s)
            e = est_for(w, d_method)
            agree_true += int(e.usable == true_settled)
            wg = within_group_perm(n, drng)
            e_wg = est_for([x[wg] for x in w], d_method)
            flips_within += int(e_wg.usable != e.usable)
        metrics[f"n{n}_agree_true_settling"] = float(agree_true)
        metrics[f"n{n}_flips_within_group"] = float(flips_within)

    lo_band, hi_band = (int(x) for x in cfg["coverage_band"])
    agree_min = int(cfg["gate_agreement_min"])
    flip_max = int(cfg["flag_flip_max"])

    def passes(c: Cell) -> bool:
        """S1 (coverage in band) AND S1b (width guard) AND S2 AND S3."""
        return (
            lo_band <= c.cover <= hi_band
            and c.width_ratio <= max_w
            and c.agree >= agree_min
            and c.flips <= flip_max
        )

    passing = {k: v for k, v in results.items() if passes(v)}
    # the certification is the SMALLEST n that passes with at least one method
    certified_n = min((n for (n, _method) in passing), default=0)
    certified_methods = sorted({m for (n, m) in passing if n == certified_n})
    metrics["screened_min_n"] = float(certified_n)
    metrics["n_passing_cells"] = float(len(passing))

    # SELECTION VALIDATION: the smallest-of-15 winner must repeat on a
    # disjoint seed stream, or the certification is refused outright.
    holdout_ok = False
    if certified_n:
        hc = run_cell(
            certified_n, certified_methods[0], seed_offset=int(cfg["holdout_seed_offset"])
        )
        holdout_ok = (
            lo_band <= hc.cover <= hi_band and hc.width_ratio <= max_w and hc.flips <= flip_max
        )
        metrics["holdout_coverage"] = float(hc.cover)
        metrics["holdout_width_over_se"] = hc.width_ratio
        metrics["holdout_flag_flips"] = float(hc.flips)
    metrics["holdout_confirmed"] = float(holdout_ok)

    # SCOPE VALIDATION: coverage must also hold on a DIFFERENT game at a
    # settled hold (transfer, not just a different tau — see the config note).
    scope_ok = False
    if certified_n and holdout_ok:
        sec = cfg["s1_scope_validation"]["secondary"]
        game2 = make_family(
            coordination(2, 2, bonus=2.0), matching_pennies(), [float(sec["alpha"])]
        )[0]
        tau2 = float(sec["tau"])
        proto2 = QuenchProtocol(lambdas=lams, taus=jnp.full((len(lams) - 1,), tau2))
        exact2 = float(hatano_sasa_exact(game2, proto2)[1])
        sec_seeds = int(sec["n_seeds"])
        cover2 = 0
        for s in range(sec_seeds):
            w2 = sample_quench_states(
                game2,
                proto2,
                n_trajectories=certified_n,
                steps_per_unit_time=int(sec["steps_per_unit_time"]),
                seed=seed + 7919 * s,
            )
            e2 = hs_y_estimate(
                w2,
                n_states=4,
                hold_durations=[tau2] * len(w2),
                pseudocount=float(cfg["pseudocount"]),
                relax_safety=float(cfg["relax_safety"]),
                interval_method=certified_methods[0],
            )
            cover2 += int(e2.mean_y_ci_low <= exact2 <= e2.mean_y_ci_high)
        frac = cover2 / sec_seeds
        scope_ok = frac >= float(cfg["s1_scope_validation"]["secondary_coverage_min_fraction"])
        metrics["secondary_coverage"] = float(cover2)
        metrics["secondary_coverage_fraction"] = frac
    metrics["scope_confirmed"] = float(scope_ok)

    if not (holdout_ok and scope_ok):
        certified_n = 0
    metrics["certified_min_n"] = float(certified_n)

    if certified_n:
        verdict = (
            f"EXTENDED: n >= {certified_n} certified via {certified_methods[0]} "
            "(screened, held-out-confirmed on a disjoint seed stream, and "
            "transfer-validated on a second game)"
        )
    elif passing:
        verdict = (
            f"REFUSED: n = {min(n for (n, _mm) in passing)} passed the screen but failed "
            f"{'held-out confirmation' if not holdout_ok else 'transfer validation'} — "
            "the screen alone is a best-of-15 selection and does not certify"
        )
    else:
        verdict = (
            "REFUSED at every tested n — no interval method holds S1/S1b/S2/S3 below "
            "the existing n >= 200 certification"
        )
    fallback = results.get((200, "percentile"))
    boundary_cell = (
        results[(certified_n, certified_methods[0])]
        if certified_n
        else (fallback if fallback is not None else Cell(0, 0, 0, 0.0, n_seeds))
    )
    res = BenchmarkResult(
        benchmark_id="smalln_certification",
        unit=UNIT,
        kind="statistical",
        passed=True,  # ran per registration; extend-or-refuse is the datum
        metrics=metrics,
        effect_sizes=[
            EffectSize(
                name=f"coverage at the certified boundary n={certified_n or 200}",
                value=float(boundary_cell.cover) / n_seeds,
                ci_low=float(lo_band) / n_seeds,
                ci_high=1.0,
                method=(
                    f"{n_seeds} seeds per (n, method) cell; registered band "
                    f"[{lo_band}, {hi_band}]/{n_seeds}, gate agreement >= {agree_min}, "
                    f"flag flips <= {flip_max}"
                ),
            )
        ],
        n=n_seeds * len(cfg["n_grid"]) * len(methods),
        n_justification=(
            f"{n_seeds} seeds x {len(cfg['n_grid'])} n-levels x {len(methods)} interval "
            "methods, each seed also re-read under a physically-null permutation "
            "(S3) — the flag-stability criterion that failed at n~30 in F-0017."
        ),
        seed=seed,
        config_ref=str(CONFIG.relative_to(REPO)),
        library_version=strataq.__version__,
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        notes=(
            f"R8 verdict: {verdict}. Criteria S1 (coverage in band), S2 (gate "
            "decision agrees with the n=400 reference), S3 (anomaly flag stable "
            "under a physically-null permutation) were registered before the run; "
            "the smallest n passing all three with any interval method is the new "
            "certified floor, and a refusal is an equally acceptable outcome."
        ),
    )
    path = RESULTS / "smalln_certification.json"
    path.write_text(res.model_dump_json(indent=2) + "\n")
    print(f"[PASS] smalln_certification -> {path.name}")
    print(f"  {verdict}")
    print("  diagnostics (percentile; true_settled=" + str(true_settled) + "):")
    for n in [int(x) for x in diag["n_grid"]]:
        print(
            f"    n={n:>3} agree-with-TRUE {metrics[f'n{n}_agree_true_settling']:.0f}/{n_seeds}"
            f"  flips(full perm) {metrics[f'n{n}_{d_method}_flag_flips']:.0f}"
            f"  flips(within-group) {metrics[f'n{n}_flips_within_group']:.0f}"
        )
    for (n, m), c in sorted(results.items()):
        print(
            f"  n={n:>3} {m:<12} coverage {c.cover:>2}/{n_seeds} "
            f"width/SE {c.width_ratio:>4.2f} agree {c.agree:>2} flips {c.flips:>2}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
