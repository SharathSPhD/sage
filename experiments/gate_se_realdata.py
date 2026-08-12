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
specifically from F-0017's UNSTABLE months, where the anomaly flag flipped
under a physically-null day-order shuffle. If those labels stabilise here,
the fix does something; if they do not, the residual is elsewhere.

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

    rng = np.random.default_rng(int(cfg["seed"]))
    metrics: dict[str, float] = {"n_days": float(n_days), "n_states": float(n_states)}
    table: list[tuple[str, str, float, bool, str]] = []
    months = sorted({dd.month for dd in month_of})
    # one shuffle per month, drawn ONCE and shared across methods so the
    # comparison is on identical data (R7 drew inside its own loop)
    shuffles = {
        mth: rng.permutation(np.array([i for i, dd in enumerate(month_of) if dd.month == mth]))
        for mth in months
    }

    for method in SE_METHODS:
        full = read(windows, method)
        metrics[f"full_{method}_usable"] = float(full.usable)
        metrics[f"full_{method}_mean_y"] = full.mean_y
        metrics[f"full_{method}_relax_refused"] = float(
            any("relaxation gate" in w for w in full.warnings)
        )
        table.append(("full", method, full.mean_y, full.usable, ""))
        for mth in months:
            sel = np.array([i for i, dd in enumerate(month_of) if dd.month == mth])
            if len(sel) < 20:
                continue
            em = read([w[sel] for w in windows], method)
            es = read([w[shuffles[mth]] for w in windows], method)
            anom = any("ANOMALY" in w for w in em.warnings)
            anom_s = any("ANOMALY" in w for w in es.warnings)
            # R7's four-way cause labels, unchanged so the comparison is direct
            if anom:
                cause = "drift" if not anom_s else "not_drift"
            else:
                cause = "UNSTABLE" if anom_s else "clean"
            metrics[f"m{mth}_{method}_anomaly"] = float(anom)
            metrics[f"m{mth}_{method}_anomaly_shuffled"] = float(anom_s)
            metrics[f"m{mth}_{method}_unstable"] = float(cause == "UNSTABLE")
            metrics[f"m{mth}_{method}_ift"] = em.ift_estimate
            table.append((f"month {mth:02d}", method, em.ift_estimate, em.usable, cause))
        n_unstable = sum(1 for mth in months if metrics.get(f"m{mth}_{method}_unstable", 0.0) > 0.5)
        metrics[f"{method}_n_unstable_months"] = float(n_unstable)

    base_u = metrics["split_n_unstable_months"]
    changes = [
        f"{m}: {metrics[f'{m}_n_unstable_months']:.0f} UNSTABLE months (split: {base_u:.0f})"
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
            f"{n_days} day-pairs): every SE candidate re-reads the monthly scan with "
            "the day-order shuffle control. Diagnostic only — NO criterion attached, "
            "and it does not certify monthly windows (R8's C-3 stands). "
            + "; ".join(changes)
            + f". strataq {strataq.__version__}, generated {datetime.now(UTC).isoformat()}"
        ),
    )
    (RESULTS / "gate_se_realdata.json").write_text(res.to_json())

    hdr = f"{'window':>10} {'method':>10} {'mean_y/ift':>11} {'usable':>7} {'cause':>10}"
    print(f"{hdr}\n{'-' * len(hdr)}")
    for w, method, val, ok, cause in table:
        print(f"{w:>10} {method:>10} {val:>11.4f} {ok!s:>7} {cause:>10}")
    print()
    for c in changes:
        print(c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
