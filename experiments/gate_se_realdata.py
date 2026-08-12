"""R9 downstream obligation — does the split-independent SE change REAL reads?

Run: ``uv run python -m experiments.gate_se_realdata``
The registered decision rule in config/experiments/gate_se.yaml requires that
adopting a new SE method be checked against the affected artifacts, "including
if a previously admitted read becomes refused". This script re-reads R7's
CAISO day-pair panel (unit domains.electricity.quench, F-0017) under EVERY SE
candidate and reports what moves.

It also answers the objection that G1 is vacuous for the exactly
order-invariant candidates: their invariance is a theorem, so passing G1 is
arithmetic rather than evidence. The evidence has to come from real data —
from the rate at which a physically-null day-order shuffle flips a month's
anomaly verdict.

A v1 of this script measured that with ONE shuffle per month, copying R7's
design, and found 0 flips under every method including the incumbent. That
was not a result, it was a coin flip: R7's reported "1 of 7 months UNSTABLE"
rests on one draw per month too, and a different draw gives 0. So the count
in F-0017 carried unstated sampling error. This version measures a FLIP RATE
over 20 shuffles per month, with the same shuffle set shared across methods.

NO CRITERION IS ATTACHED. This is a declared diagnostic: it reports label
changes, it does not certify monthly windows. Doing that needs its own
registered unit re-establishing small-n coverage (R8's C-3 stands).
Artifact: ``gate_se_realdata.json``.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import strataq
import yaml
from strataq.domains.electricity.caiso import fetch_dam_lmp
from strataq.thermo.hs_estimator import SE_METHODS, hs_y_estimate
from strataq_bench import BenchmarkResult

from experiments.day_quench_reading import day_windows

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
UNIT = "thermo.hs_estimator.gate_se"


def main() -> int:
    cfg: dict[str, Any] = yaml.safe_load(
        (REPO / "config" / "experiments" / "day_quench.yaml").read_text()
    )
    dc = cfg["data"]
    y, m, d = (int(x) for x in str(dc["start"]).split("-"))
    ts, prices = fetch_dam_lmp(date(y, m, d), int(dc["days"]))
    bounds = [int(b) for b in cfg["holds"]["hour_boundaries"]]
    durations = [float(b2 - b1) for b1, b2 in pairwise(bounds)]
    durations = [*durations, durations[0]]
    windows, n_states, _dropped, month_of = day_windows(
        ts, prices, bounds, int(cfg["embedding"]["n_price_bins"])
    )
    n_days = windows[0].shape[0]

    def read(ws: list[np.ndarray], method: str):
        return hs_y_estimate(
            ws,
            n_states=n_states,
            hold_durations=durations,
            pseudocount=float(cfg["estimator"]["pseudocount"]),
            relax_safety=float(cfg["estimator"]["relax_safety"]),
            relax_se_method=method,
        )

    # MANY shuffles per month, not one. A v1 of this script used a single
    # permutation per month (R7's own design) and measured 0 UNSTABLE months
    # under EVERY method including the incumbent — because with one draw the
    # label is a coin flip, not a measurement. R7 reported 1 of 7 months
    # UNSTABLE from one draw each; a different draw gives 0. The count
    # therefore had unstated sampling error, and the honest statistic is a
    # FLIP RATE over many physically-null day-order shuffles, sharing the same
    # shuffle set across methods so the comparison is on identical data.
    n_shuffles = 20
    rng = np.random.default_rng(int(cfg["seed"]))
    months = sorted({dd.month for dd in month_of})
    sel_of = {
        mth: np.array([i for i, dd in enumerate(month_of) if dd.month == mth]) for mth in months
    }
    months = [mth for mth in months if len(sel_of[mth]) >= 20]
    shuffles = {mth: [rng.permutation(sel_of[mth]) for _ in range(n_shuffles)] for mth in months}
    metrics: dict[str, float] = {
        "n_days": float(n_days),
        "n_states": float(n_states),
        "n_shuffles_per_month": float(n_shuffles),
    }
    table: list[tuple[str, str, float, str, str]] = []

    def subset(idx: np.ndarray) -> list[np.ndarray]:
        return [w[idx] for w in windows]

    def anomaly(idx: np.ndarray, method: str) -> bool:
        est = read(subset(idx), method)
        return any("ANOMALY" in msg for msg in est.warnings)

    for method in SE_METHODS:
        full = read(windows, method)
        metrics[f"full_{method}_usable"] = float(full.usable)
        metrics[f"full_{method}_mean_y"] = full.mean_y
        metrics[f"full_{method}_relax_refused"] = float(
            any("relaxation gate" in msg for msg in full.warnings)
        )
        table.append(("full", method, full.mean_y, str(full.usable), ""))
        total_flips = 0
        n_admitted = 0
        for mth in months:
            em = read(subset(sel_of[mth]), method)
            anom = any("ANOMALY" in msg for msg in em.warnings)
            # flip rate: how often a permutation that cannot change any
            # physical property nonetheless changes the anomaly verdict
            flips = sum(int(anomaly(p, method) != anom) for p in shuffles[mth])
            total_flips += flips
            n_admitted += int(em.usable)
            metrics[f"m{mth}_{method}_anomaly"] = float(anom)
            metrics[f"m{mth}_{method}_flip_rate"] = flips / n_shuffles
            metrics[f"m{mth}_{method}_usable"] = float(em.usable)
            metrics[f"m{mth}_{method}_ift"] = em.ift_estimate
            table.append(
                (
                    f"month {mth:02d}",
                    method,
                    em.ift_estimate,
                    str(em.usable),
                    f"{flips}/{n_shuffles} flips",
                )
            )
        metrics[f"{method}_total_flip_rate"] = total_flips / (n_shuffles * len(months))
        metrics[f"{method}_n_months_admitted"] = float(n_admitted)

    changes = [
        f"{m}: flip rate {metrics[f'{m}_total_flip_rate']:.3f}, "
        f"{metrics[f'{m}_n_months_admitted']:.0f}/{len(months)} months admitted "
        f"(split: {metrics['split_total_flip_rate']:.3f}, "
        f"{metrics['split_n_months_admitted']:.0f}/{len(months)})"
        for m in SE_METHODS
        if m != "split"
    ]
    res = BenchmarkResult(
        benchmark_id="gate_se_realdata",
        unit=UNIT,
        kind="statistical",
        passed=True,
        metrics=metrics,
        n=n_days,
        n_justification=(
            f"{n_days} CAISO day-pairs, the same panel R7 read; the monthly cells carry "
            "~30 day-pairs each, which is exactly the small-n regime R8 refused. No "
            "criterion is attached, so no power claim is made"
        ),
        seed=int(cfg["seed"]),
        config_ref="config/experiments/day_quench.yaml",
        library_version=strataq.__version__,
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        notes=(
            "R9 downstream obligation on REAL data (R7's CAISO day-pair panel, "
            f"{n_days} day-pairs): every SE candidate re-reads each monthly window and "
            f"{n_shuffles} physically-null day-order shuffles of it, so the anomaly "
            "flag's instability is a measured RATE rather than a single draw. This also "
            "corrects a v1 of this diagnostic (and, retroactively, R7's own design) "
            "which used one shuffle per month and so reported an UNSTABLE COUNT with "
            "unstated sampling error. Diagnostic only — NO criterion attached, and it "
            "does not certify monthly windows (R8's C-3 stands). "
            + "; ".join(changes)
            + f". strataq {strataq.__version__}, generated {datetime.now(UTC).isoformat()}"
        ),
    )
    (RESULTS / "gate_se_realdata.json").write_text(res.model_dump_json(indent=2) + "\n")

    hdr = f"{'window':>10} {'method':>10} {'mean_y/ift':>11} {'usable':>7} {'null-shuffle':>14}"
    print(f"{hdr}\n{'-' * len(hdr)}")
    for w, method, val, ok, cause in table:
        print(f"{w:>10} {method:>10} {val:>11.4f} {ok:>7} {cause:>14}")
    print()
    for c in changes:
        print(c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
