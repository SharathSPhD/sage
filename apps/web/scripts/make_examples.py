#!/usr/bin/env python
"""Regenerate the bundled example series for /diagnose.

    /home/claude/sage/.venv/bin/python apps/web/scripts/make_examples.py

Three files, and they are NOT all the same kind of thing. The difference is stated
here, on the button in the UI, and in the CSV header itself:

  caiso.csv         REAL. 840 hourly CAISO SP15 day-ahead prices, July 2026, copied
                    straight out of the committed, gate-checked artifact
                    `apps/web/data/electricity_series.json` (unit domains.electricity)
                    that the /markets page already renders. No resampling, no noise,
                    no rounding beyond what is in the artifact. This is the series
                    behind F-0008/F-0009.

  random-walk.csv   SYNTHETIC. A driftless Gaussian random walk. It has no cycle and
                    no preferred direction in time, so the irreversibility test should
                    NOT escape its detailed-balance null. It is here as the negative
                    control: an instrument that fires on this is broken.

  whirlpool.csv     SYNTHETIC. An Edgeworth price cycle -- a slow undercutting ramp
                    down, then a jump back up in one step. Irreversible by construction:
                    played backwards it is a slow ramp UP and a collapse DOWN, which is
                    a visibly different process, and that asymmetry is what the test
                    picks up.

There is deliberately no "Dominick's" example. This repository has no committed
Dominick's Finer Foods series, and the honest options were to ship real data or to
ship something clearly labelled as generated -- never to generate a series and give
it a real retailer's name. The R = 0.00112 Dominick's reading still appears on the
plane as a reference mark, where it comes from the actual finding.

Every generated file is seeded; running this twice gives byte-identical output.
Every file carries a first column that is NOT a counter (the real one carries
timestamps), because a counter is a perfect ramp and is the most irreversible series
there is -- see the column-inference note in app/diagnose/page.tsx.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

SEED = 20260812
WEB = Path(__file__).resolve().parents[1]
OUT = WEB / "public" / "examples"
ARTIFACT = WEB / "data" / "electricity_series.json"


def write(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    print(f"{path}  ({len(rows)} rows)")


def caiso_real() -> None:
    """Copy the committed real series out to CSV. Nothing is computed here."""
    art = json.loads(ARTIFACT.read_text())
    hours, prices = art["hours"], art["prices"]
    if len(hours) != len(prices):
        raise SystemExit("artifact is malformed: hours and prices differ in length")
    write(
        OUT / "caiso.csv",
        ["timestamp_utc", "price_usd_per_mwh"],
        [[h, f"{p:g}"] for h, p in zip(hours, prices, strict=True)],
    )


def random_walk(rng: np.random.Generator, n: int = 500) -> None:
    """Driftless Gaussian random walk -- the reversible negative control."""
    x = np.cumsum(rng.normal(0.0, 1.0, n))
    write(
        OUT / "random-walk.csv",
        ["step_label", "value"],
        [[f"t{i + 1:04d}", f"{v:.4f}"] for i, v in enumerate(x)],
    )


def whirlpool(rng: np.random.Generator, n: int = 600) -> None:
    """Edgeworth sawtooth: slow undercutting ramp, then a one-step jump back up."""
    ceiling, floor_ = 1.92, 1.31
    x = np.empty(n)
    p = ceiling
    for i in range(n):
        x[i] = p
        if p <= floor_ + rng.uniform(0.0, 0.05):
            p = ceiling - rng.uniform(0.0, 0.06)  # someone restores the price, in one step
        else:
            p -= rng.uniform(0.012, 0.048)  # undercut by a small, noisy amount
    x = np.clip(x + rng.normal(0.0, 0.006, n), 1.0, 2.5)
    write(
        OUT / "whirlpool.csv",
        ["period_label", "price"],
        [[f"p{i + 1:04d}", f"{v:.3f}"] for i, v in enumerate(x)],
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    caiso_real()
    random_walk(rng)
    whirlpool(rng)
    stale = OUT / "dominicks.csv"
    if stale.exists():
        stale.unlink()
        print(f"removed {stale} (fabricated retail data -- see the module docstring)")


if __name__ == "__main__":
    main()
