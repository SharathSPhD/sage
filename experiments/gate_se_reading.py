"""R9 — a split-independent SE for the relaxation gate (unit
thermo.hs_estimator.gate_se).

Run: ``uv run python -m experiments.gate_se_reading``
Registered G1-G5 in config/experiments/gate_se.yaml (both commits verified
landed before this ran: the criteria, then the G5 aggregation/oracle-stream
resolution). The unit either adopts a candidate that passes ALL FIVE or
publishes which came closest and the residual mechanism — both are
acceptable outcomes and neither is pre-declared.
Artifact: ``gate_se_read.json``.

PROCESS NOTE: a first invocation was STOPPED mid-flight and its output was
never read, in order to add Jeffreys intervals to the reported proportions
and the oracle's own relative SE while still blind to any result. No
criterion changed; the additions are reporting only. (Same discipline as
R8, whose pre-amendment run was likewise stopped before producing a verdict.)
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
from scipy.stats import beta as sp_beta
from strataq.core.dynamics.markov import glauber_generator
from strataq.finite.decompose.generate import make_family
from strataq.finite.games.library import coordination, matching_pennies
from strataq.thermo.hs_estimator import relaxation_gate, sample_quench_states
from strataq.thermo.protocols import QuenchProtocol
from strataq_bench import BenchmarkResult, EffectSize

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
CONFIG = REPO / "config" / "experiments" / "gate_se.yaml"
UNIT = "thermo.hs_estimator.gate_se"


@dataclass(frozen=True)
class MethodCell:
    """One (n, se_method) cell across the registered criteria."""

    flips: int  # G1: flag flips under the FULL trajectory permutation
    agree: int  # G2/G3: agreement with the EXACT settling status
    se_ratio_max: float  # G5: worst-window median-SE / oracle-SE
    se_ratio_med: float  # G5 companion: median over windows
    seeds: int


def main() -> int:
    cfg: dict[str, Any] = yaml.safe_load(CONFIG.read_text())
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "gate_se.resolved.yaml").write_text(yaml.safe_dump(cfg))

    seed = int(cfg["seed"])
    game = make_family(
        coordination(2, 2, bonus=2.0), matching_pennies(), [float(cfg["family"]["alpha"])]
    )[0]
    lams = [float(x) for x in cfg["protocol"]["lambdas"]]
    tau = float(cfg["protocol"]["tau"])
    spu = int(cfg["steps_per_unit_time"])
    safety = float(cfg["relax_safety"])
    methods = list(cfg["se_candidates"])
    n_seeds = int(cfg["n_seeds"])

    # TRUE settling status from the generator's exact spectral gaps — no
    # sampling noise, so a disagreement cannot be blamed on the reference
    # (R8's C-1, reused verbatim so R9's G2 is on the same footing).
    gaps = [
        float(-np.sort(np.linalg.eigvals(np.asarray(glauber_generator(game, lam))).real)[::-1][1])
        for lam in lams[1:]
    ]
    true_max_relax = max(1.0 / g for g in gaps)
    true_settled = all(tau >= safety / g for g in gaps)

    def windows_for(n: int, s: int, hold: float = tau) -> list[np.ndarray]:
        p = QuenchProtocol(lambdas=jnp.array(lams), taus=jnp.full((len(lams) - 1,), hold))
        return sample_quench_states(
            game, p, n_trajectories=n, steps_per_unit_time=spu, seed=seed + 97 * s + n
        )

    def gate_for(w: list[np.ndarray], method: str, hold: float = tau):
        return relaxation_gate(
            w,
            n_states=4,
            hold_durations=[hold] * len(w),
            relax_safety=safety,
            se_method=method,
            bootstrap_resamples=int(cfg["bootstrap_resamples"]),
        )

    metrics: dict[str, float] = {
        "true_settled_primary": float(true_settled),
        "true_max_relax_time": true_max_relax,
        "gate_requires_tau_at_least": safety * true_max_relax,
    }

    # ---- G5 oracle: the TRUE sampling SE of tau_hat, from independent draws
    # on a DISJOINT seed stream (resolved at registration). One per window.
    oracle: dict[int, np.ndarray] = {}
    off = int(cfg["oracle_seed_offset"])
    reps = int(cfg["oracle_replicates"])
    all_n = [int(x) for x in cfg["g_grid"]] + [int(x) for x in cfg["g_grid_large"]]
    for n in all_n:
        draws = [gate_for(windows_for(n, off + r), "delta").tau_hats for r in range(reps)]
        oracle[n] = np.std(np.asarray(draws), axis=0, ddof=1)
        metrics[f"n{n}_oracle_se_med"] = float(np.median(oracle[n]))

    # ---- G1 / G2 / G3 / G5: one pass per n, windows sampled ONCE and shared
    # across candidates so the comparison is on identical data
    prng = np.random.default_rng(seed + 4242)
    cells: dict[tuple[int, str], MethodCell] = {}
    for n in all_n:
        draws = [windows_for(n, s) for s in range(n_seeds)]
        perms = [prng.permutation(n) for _ in range(n_seeds)]
        for method in methods:
            flips = agree = 0
            ses = []
            for w, perm in zip(draws, perms, strict=True):
                g = gate_for(w, method)
                agree += int(g.ok == true_settled)
                ses.append(g.ses)
                # G1: the physically-null full trajectory permutation
                g_perm = gate_for([x[perm] for x in w], method)
                flips += int(g_perm.ok != g.ok)
            med_se = np.median(np.asarray(ses), axis=0)
            ratio = np.abs(med_se / np.maximum(oracle[n], 1e-12) - 1.0)
            cell = MethodCell(
                flips=flips,
                agree=agree,
                se_ratio_max=float(np.max(ratio)),
                se_ratio_med=float(np.median(ratio)),
                seeds=n_seeds,
            )
            cells[(n, method)] = cell
            key = f"n{n}_{method}"
            metrics[f"{key}_flag_flips"] = float(cell.flips)
            metrics[f"{key}_agree_true_settling"] = float(cell.agree)
            metrics[f"{key}_se_ratio_max"] = cell.se_ratio_max
            metrics[f"{key}_se_ratio_med"] = cell.se_ratio_med

    # ---- G4: refusal power on genuinely unsettled holds. This is the guard
    # that stops G2 being satisfied by driving the SE to zero.
    refusals: dict[tuple[int, str], int] = {}
    for n in [int(x) for x in cfg["unsettled_n_grid"]]:
        for hold in [float(x) for x in cfg["unsettled_taus"]]:
            draws = [windows_for(n, s, hold) for s in range(n_seeds)]
            for method in methods:
                got = sum(int(not gate_for(w, method, hold).ok) for w in draws)
                refusals[(n, method)] = refusals.get((n, method), 0) + got
                metrics[f"n{n}_tau{hold:g}_{method}_refusals"] = float(got)

    # ---- adjudication, strictly by the registered thresholds -------------
    flip_max = int(cfg["flag_flip_max"])
    agree_min = int(cfg["agreement_min"])
    agree_min_lg = int(cfg["agreement_min_large"])
    tol = float(cfg["se_accuracy_tol"])
    need_ref = int(cfg["unsettled_refusal_required"]) * len(cfg["unsettled_taus"])
    small = [int(x) for x in cfg["g_grid"]]
    large = [int(x) for x in cfg["g_grid_large"]]

    verdicts: dict[str, dict[str, Any]] = {}
    for method in methods:
        g1 = all(cells[(n, method)].flips <= flip_max for n in small)
        g2 = all(cells[(n, method)].agree >= agree_min for n in small)
        g3 = all(cells[(n, method)].agree >= agree_min_lg for n in large)
        g4 = all(
            refusals[(n, method)] >= need_ref for n in [int(x) for x in cfg["unsettled_n_grid"]]
        )
        g5 = all(cells[(n, method)].se_ratio_max <= tol for n in small)
        # margin: how much slack the WORST criterion has, normalised per
        # criterion so the tie-break is comparable across dimensions
        margins = [
            min((flip_max - cells[(n, method)].flips) / max(flip_max, 1) for n in small),
            min((cells[(n, method)].agree - agree_min) / n_seeds for n in small),
            min((cells[(n, method)].agree - agree_min_lg) / n_seeds for n in large),
            min(
                (refusals[(n, method)] - need_ref) / need_ref
                for n in [int(x) for x in cfg["unsettled_n_grid"]]
            ),
            min((tol - cells[(n, method)].se_ratio_max) / tol for n in small),
        ]
        verdicts[method] = {
            "G1_order_invariance": g1,
            "G2_accuracy_small_n": g2,
            "G3_no_regression": g3,
            "G4_refusal_power": g4,
            "G5_se_accuracy": g5,
            "passes_all": bool(g1 and g2 and g3 and g4 and g5),
            "min_margin": float(min(margins)),
            "eligible": method != "split",  # incumbent is baseline only
        }
        metrics[f"{method}_passes_all"] = float(verdicts[method]["passes_all"])
        metrics[f"{method}_min_margin"] = float(verdicts[method]["min_margin"])

    winners = [m for m, v in verdicts.items() if v["passes_all"] and v["eligible"]]
    # decision rule fixed before results: highest minimum margin, ties to the
    # cheaper method (candidate order in config is cheap-to-expensive)
    adopted = (
        max(winners, key=lambda m: (verdicts[m]["min_margin"], -methods.index(m)))
        if winners
        else ""
    )
    metrics["n_candidates_passing"] = float(len(winners))
    verdict = (
        f"ADOPT {adopted}: split-independent SE passes G1-G5"
        if adopted
        else "REFUSED: no candidate passes all of G1-G5 (see per-criterion flags)"
    )

    # 20 seeds is a small denominator: every reported proportion carries a
    # Jeffreys binomial interval so the criteria's counts cannot be read as
    # more precise than they are. The CRITERIA remain counts vs the registered
    # thresholds — these intervals are reporting, not adjudication.
    def jeffreys(k: int, n: int) -> tuple[float, float]:
        return (
            float(sp_beta.ppf(0.025, k + 0.5, n - k + 0.5)),
            float(sp_beta.ppf(0.975, k + 0.5, n - k + 0.5)),
        )

    # the G5 oracle is itself an SD from a finite sample: an SD estimated from
    # R draws carries relative SE ~ 1/sqrt(2(R-1)), reported so the tolerance
    # can be read against the reference's own noise
    oracle_rel_se = float(1.0 / np.sqrt(2.0 * (reps - 1)))
    metrics["oracle_relative_se"] = oracle_rel_se

    r8_flip_baseline = {30: 6, 50: 5, 100: 3}
    effects = []
    for n in small + large:
        for method in methods:
            c = cells[(n, method)]
            lo, hi = jeffreys(c.flips, n_seeds)
            effects.append(
                EffectSize(
                    name=f"n{n}_flag_flip_rate_{method}",
                    value=c.flips / n_seeds,
                    ci_low=lo,
                    ci_high=hi,
                    method=f"Jeffreys binomial interval over {n_seeds} seeds",
                    interpretation=(
                        f"G1 at n={n}, {method}: {c.flips}/{n_seeds} flag flips under the "
                        f"physically-null full trajectory permutation (bar <= {flip_max}"
                        + (
                            f"; R8 baseline with the incumbent split: {r8_flip_baseline[n]}"
                            if n in r8_flip_baseline
                            else ""
                        )
                        + ")"
                    ),
                )
            )
            alo, ahi = jeffreys(c.agree, n_seeds)
            effects.append(
                EffectSize(
                    name=f"n{n}_agreement_rate_{method}",
                    value=c.agree / n_seeds,
                    ci_low=alo,
                    ci_high=ahi,
                    method=f"Jeffreys binomial interval over {n_seeds} seeds",
                    interpretation=(
                        f"G2/G3 at n={n}, {method}: {c.agree}/{n_seeds} agreement with the "
                        "EXACT spectral-gap settling status (bar "
                        f"{agree_min_lg if n in large else agree_min}; R8 baseline with the "
                        "incumbent split: 10/15/16 at n=30/50/100, 19 at n=200)"
                    ),
                )
            )
            effects.append(
                EffectSize(
                    name=f"n{n}_se_over_oracle_{method}",
                    value=c.se_ratio_max,
                    ci_low=max(0.0, c.se_ratio_max - oracle_rel_se),
                    ci_high=c.se_ratio_max + oracle_rel_se,
                    method=(
                        f"worst-window |median SE / oracle SE - 1|; the interval spans the "
                        f"oracle's own relative SE (1/sqrt(2*({reps}-1)) = {oracle_rel_se:.3f}) "
                        "since the reference is itself an SD from a finite sample"
                    ),
                    interpretation=(
                        f"G5 at n={n}, {method}: worst-window SE deviation "
                        f"{c.se_ratio_max:.3f} (bar <= {tol}; median over windows "
                        f"{c.se_ratio_med:.3f}). A value ABOVE the bar overstates the "
                        "uncertainty, one below understates it — both fail"
                    ),
                )
            )

    res = BenchmarkResult(
        benchmark_id="gate_se_read",
        unit=UNIT,
        kind="statistical",
        passed=True,  # ran per registration; the adjudication is the finding
        metrics=metrics,
        effect_sizes=effects,
        n_samples=n_seeds,
        seeds=[seed],
        notes=(
            f"{verdict}. Registered G1-G5 (config/experiments/gate_se.yaml, commits "
            "verified landed before the run). Per-candidate criterion flags in "
            f"metrics. Baseline (R8): agreement 10/15/16 at n=30/50/100, flips "
            f"6/5/3. True settling status={true_settled} (tau={tau} vs required "
            f"{safety * true_max_relax:.2f}). G5 oracle = SD of tau_hat over "
            f"{reps} independent draws on a disjoint seed stream; the statistic "
            "is the WORST window (the gate refuses on its worst window). "
            f"strataq {strataq.__version__}, generated {datetime.now(UTC).isoformat()}"
        ),
    )
    (RESULTS / "gate_se_read.json").write_text(res.to_json())

    print(verdict)
    print(f"\ntrue_settled={true_settled}  tau={tau}  requires>={safety * true_max_relax:.2f}")
    hdr = f"{'n':>5} {'method':>10} {'flips':>6} {'agree':>6} {'se/oracle-1 max':>16}"
    print(f"\n{hdr}\n{'-' * len(hdr)}")
    for n in all_n:
        for method in methods:
            c = cells[(n, method)]
            print(f"{n:>5} {method:>10} {c.flips:>6} {c.agree:>6} {c.se_ratio_max:>16.3f}")
    print(f"\n{'method':>10} {'G1':>4} {'G2':>4} {'G3':>4} {'G4':>4} {'G5':>4} {'margin':>8}")
    for method in methods:
        v = verdicts[method]
        flags = " ".join(
            "  Y " if v[k] else "  n "
            for k in (
                "G1_order_invariance",
                "G2_accuracy_small_n",
                "G3_no_regression",
                "G4_refusal_power",
                "G5_se_accuracy",
            )
        )
        print(f"{method:>10} {flags} {v['min_margin']:>8.3f}")
    for n in [int(x) for x in cfg["unsettled_n_grid"]]:
        for method in methods:
            print(f"G4 n={n} {method}: {refusals[(n, method)]}/{need_ref} refusals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
