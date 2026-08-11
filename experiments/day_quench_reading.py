"""R7 — the trading day as a repeated quench (unit domains.electricity.quench).

Run: ``uv run python -m experiments.day_quench_reading``
Registered design D1–D4 in config/experiments/day_quench.yaml (commit
verified landed; NO outcome declared — the certified estimator's own
verdicts are the finding). Artifact: ``day_quench_read.json``.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
import strataq
import yaml
from strataq.domains.electricity import phase_embed
from strataq.domains.electricity.caiso import fetch_dam_lmp
from strataq.thermo.hs_estimator import HSEstimate, hs_y_estimate
from strataq_bench import BenchmarkResult, EffectSize

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
CONFIG = REPO / "config" / "experiments" / "day_quench.yaml"
UNIT = "domains.electricity.quench"


def day_windows(
    ts: list, prices: list[float], boundaries: list[int], n_bins: int
) -> tuple[list[np.ndarray], int, int, list]:
    """(windows per hold, n_states, n_days_dropped): days as trajectories.

    Each complete 24-hour day is phase-embedded on its own (23 samples after
    the delta); hour-of-day blocks become the hold windows. Days with any
    missing hour are dropped and counted (D1).
    """
    by_day: dict[date, dict[int, float]] = {}
    for t, p in zip(ts, prices, strict=True):
        by_day.setdefault(t.date(), {})[t.hour] = p
    complete = {d: hs for d, hs in by_day.items() if len(hs) == 24}
    dropped = len(by_day) - len(complete)
    per_day_states = []
    n_states = 2 * n_bins
    for d in sorted(complete):
        series = [complete[d][h] for h in range(24)]
        states, n_states = phase_embed(series, n_bins)  # 23 states samples
        per_day_states.append(states)
    arr = np.asarray(per_day_states, dtype=np.int64)  # (n_days, 23)
    # hour boundaries are in original-hour terms; after the delta the sample
    # at index i corresponds to hour i+1, so block [a, b) maps to cols a-1:b-1
    windows = [arr[:, a - 1 : b - 1] for a, b in pairwise(boundaries)]
    # CLOSE THE CYCLE (red-team round 1, CRITICAL): the diurnal protocol is
    # circular — without the wrap-around jump (peak block -> the NEXT day's
    # night block) the daily loop is half-counted (measured: +3.7 nats/day).
    # Trajectories become day-pairs: drop the last day so every trajectory
    # has its following-day night window.
    night_next = windows[0][1:]  # next day's night block
    windows = [w[:-1] for w in windows] + [night_next]
    return windows, n_states, dropped, sorted(complete)[:-1]


def read(windows: list[np.ndarray], n_states: int, durations: list[float], cfg: dict) -> HSEstimate:
    return hs_y_estimate(
        windows,
        n_states=n_states,
        hold_durations=durations,
        pseudocount=float(cfg["estimator"]["pseudocount"]),
        relax_safety=float(cfg["estimator"]["relax_safety"]),
    )


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text())
    (RESULTS / "day_quench.resolved.yaml").write_text(yaml.safe_dump(cfg))
    dc = cfg["data"]
    y, m, d = (int(x) for x in str(dc["start"]).split("-"))
    ts, prices = fetch_dam_lmp(date(y, m, d), int(dc["days"]))
    bounds = [int(b) for b in cfg["holds"]["hour_boundaries"]]
    durations = [float(b2 - b1) for b1, b2 in pairwise(bounds)]
    durations = [*durations, durations[0]]  # the closure window (next night)
    windows, n_states, dropped, month_of = day_windows(
        ts, prices, bounds, int(cfg["embedding"]["n_price_bins"])
    )
    n_days = windows[0].shape[0]

    full = read(windows, n_states, durations, cfg)
    half = n_days // 2
    first = read([w[:half] for w in windows], n_states, durations, cfg)
    second = read([w[half:] for w in windows], n_states, durations, cfg)

    # day-BLOCK bootstrap (red-team: per-day Y lag-1 rho ~ 0.6 — iid CI too
    # narrow): resample 7-day blocks, recompute the full pipeline mean
    rng = np.random.default_rng(int(cfg["seed"]))
    block_len = 7
    n_blocks = n_days // block_len
    block_means = []
    for _ in range(200):
        starts = rng.integers(0, n_days - block_len, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_len) for s in starts])
        eb = read([w[idx] for w in windows], n_states, durations, cfg)
        block_means.append(eb.mean_y)
    blk_lo, blk_hi = (float(q) for q in np.quantile(block_means, [0.025, 0.975]))

    # monthly anomaly scan (red-team: regime mixing) — group day indices by month
    months = sorted({d.month for d in month_of})
    monthly = {}
    for mth in months:
        sel = np.array([i for i, d in enumerate(month_of) if d.month == mth])
        if len(sel) < 20:
            continue
        em = read([w[sel] for w in windows], n_states, durations, cfg)
        monthly[mth] = (em.ift_estimate, float(any("ANOMALY" in x for x in em.warnings)))

    def block(e: HSEstimate, label: str) -> dict[str, float]:
        return {
            f"{label}_usable": float(e.usable),
            f"{label}_mean_y": e.mean_y,
            f"{label}_ci_low": e.mean_y_ci_low,
            f"{label}_ci_high": e.mean_y_ci_high,
            f"{label}_ift": e.ift_estimate,
            f"{label}_ift_lo": e.ift_ci_low,
            f"{label}_ift_hi": e.ift_ci_high,
            f"{label}_relax_refused": float(any("relaxation gate" in w for w in e.warnings)),
            f"{label}_anomaly": float(any("ANOMALY" in w for w in e.warnings)),
            f"{label}_thin": float(any("thin" in w for w in e.warnings)),
        }

    verdict = (
        "admitted: first data-side excess-dissipation read"
        if full.usable
        else "REFUSED: the trading day is not stepwise-stationary at hour-block "
        "resolution (the quantified finding — see gate flags)"
    )
    res = BenchmarkResult(
        benchmark_id="day_quench_read",
        unit=UNIT,
        kind="statistical",
        passed=True,  # ran per registration; the verdict is data
        metrics={
            "n_days": float(n_days),
            "n_days_dropped": float(dropped),
            "n_states": float(n_states),
            **block(full, "full"),
            **block(first, "h1"),
            **block(second, "h2"),
            "blockboot_ci_low": blk_lo,
            "blockboot_ci_high": blk_hi,
            **{f"month{m:02d}_ift": v[0] for m, v in monthly.items()},
            **{f"month{m:02d}_anomaly": v[1] for m, v in monthly.items()},
        },
        effect_sizes=[
            EffectSize(
                name=f"day-quench mean Y_hat ({verdict[:40]})",
                value=full.mean_y,
                ci_low=full.mean_y_ci_low,
                ci_high=full.mean_y_ci_high,
                method=(
                    f"hs_y_estimate over {n_days} day-pair trajectories, CLOSED "
                    f"cycle; day-block bootstrap CI [{blk_lo:.2f}, {blk_hi:.2f}] "
                    "is the honest interval (serial days); NOTE: 6 states is "
                    "outside the estimator's validated scope (4 states) — this "
                    "application extends, not inherits, the certification"
                ),
            )
        ],
        n=n_days,
        n_justification=(
            f"{n_days} complete days ({dropped} dropped for missing hours) — the "
            "estimator's validated scope needs >= 200 trajectories; the seasonal "
            "halves are ~half that and labelled indicative-only per D4."
        ),
        seed=int(cfg["seed"]),
        config_ref=str(CONFIG.relative_to(REPO)),
        library_version=strataq.__version__,
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        notes=(
            f"R7 verdict: {verdict}. Full-window warnings: {full.warnings}. "
            "Registered design D1-D4; the certified instrument's gates were in "
            "charge and their flags are reported verbatim."
        ),
    )
    path = RESULTS / "day_quench_read.json"
    path.write_text(res.model_dump_json(indent=2) + "\n")
    print(f"[PASS] day_quench_read -> {path.name}")
    print(f"  verdict: {verdict}")
    print(
        f"  full: usable={full.usable} Y={full.mean_y:.4f} "
        f"[{full.mean_y_ci_low:.4f},{full.mean_y_ci_high:.4f}] IFT={full.ift_estimate:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
