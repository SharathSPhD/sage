"""CAISO OASIS day-ahead LMP loader — the electricity domain's data contact.

Plan v2 R2 named ERCOT, but ERCOT's public endpoints sit behind bot-blocking
(HTTP 403 to programmatic clients); CAISO's OASIS API serves day-ahead LMPs
as zipped CSV with no key and no registration, so the electricity domain's
first data contact is CAISO (deviation recorded in the unit gate).

Whole-hour DAM LMPs for a trading-hub node, cached on disk, deterministic
parse. No science here — just honest data plumbing (validated row counts,
explicit gaps).
"""

from __future__ import annotations

import csv
import io
import time
import zipfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from urllib.request import urlopen

CACHE = Path.home() / ".cache" / "strataq" / "caiso"
OASIS = "https://oasis.caiso.com/oasisapi/SingleZip"
DEFAULT_NODE = "TH_SP15_GEN-APND"  # SP15 trading hub


_MARKETS = {
    "DAM": ("PRC_LMP", "DAM", 24),
    "RTM": ("PRC_INTVL_LMP", "RTM", 288),  # 5-minute intervals
}


def _fetch_day(node: str, day: date, market: str = "DAM") -> bytes:
    queryname, run_id, _ = _MARKETS[market]
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f"{market}_{node}_{day.isoformat()}.zip"
    if cached.exists():
        return cached.read_bytes()
    start = f"{day:%Y%m%d}T08:00-0000"
    end = f"{day + timedelta(days=1):%Y%m%d}T08:00-0000"
    url = (
        f"{OASIS}?queryname={queryname}&startdatetime={start}&enddatetime={end}"
        f"&version=1&market_run_id={run_id}&node={node}&resultformat=6"
    )
    # OASIS rate-limits bursts: pace uncached fetches and back off on 429.
    from urllib.error import HTTPError

    delay = 5.0
    for attempt in range(6):
        try:
            time.sleep(delay if attempt else 5.0)
            with urlopen(url, timeout=120) as resp:
                blob = bytes(resp.read())
            cached.write_bytes(blob)
            return blob
        except HTTPError as exc:
            if exc.code != 429:
                raise
            delay = min(delay * 2, 120.0)
    raise TimeoutError(f"CAISO OASIS rate limit persisted after retries: {url}")


def _parse_zip(blob: bytes) -> dict[datetime, float]:
    out: dict[datetime, float] = {}
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for name in zf.namelist():
            if not name.endswith(".csv"):
                continue
            text = zf.read(name).decode()
            for row in csv.DictReader(io.StringIO(text)):
                if row.get("LMP_TYPE") != "LMP":
                    continue
                ts = datetime.fromisoformat(row["INTERVALSTARTTIME_GMT"]).astimezone(UTC)
                out[ts] = float(row["MW"])
    return out


def fetch_dam_lmp(
    start: date, days: int, node: str = DEFAULT_NODE, market: str = "DAM"
) -> tuple[list[datetime], list[float]]:
    """LMPs for `days` consecutive days (DAM hourly or RTM 5-minute), sorted.

    Returns (timestamps, prices). Raises if coverage falls below 90% of the
    expected intervals — silent gaps are how bad readings happen.
    """
    prices: dict[datetime, float] = {}
    for i in range(days):
        prices.update(_parse_zip(_fetch_day(node, start + timedelta(days=i), market)))
    ts = sorted(prices)
    expected = days * _MARKETS[market][2]
    if len(ts) < 0.9 * expected:
        raise ValueError(f"CAISO coverage {len(ts)}/{expected} intervals below 90% — refusing")
    return ts, [prices[t] for t in ts]


def discretize_quantile(prices: list[float], n_bins: int) -> tuple[list[int], list[float]]:
    """Quantile-bin a price series into states 0..n_bins-1.

    Quantile edges (not uniform) so every state is visited — the KLD
    estimator's plug-in needs occupancy everywhere.
    """
    srt = sorted(prices)
    edges = [srt[int(len(srt) * k / n_bins)] for k in range(1, n_bins)]
    states = []
    for p in prices:
        s = 0
        for e in edges:
            if p >= e:
                s += 1
        states.append(s)
    return states, edges


def phase_embed(prices: list[float], n_price_bins: int) -> tuple[list[int], int]:
    """Embed a scalar series as (price bin, sign of last change) states.

    A periodic drive (the diurnal cycle) retraces the same 1-D price path up
    and down, so price-VALUE discretization is structurally blind to its
    irreversibility: the time-reversed path visits the same value transitions.
    In phase space (p, Δp-sign) the loop no longer retraces itself — the
    standard position/velocity embedding. Returns (states, n_states) with
    n_states = 2 * n_price_bins; the first sample is dropped (no Δ).
    """
    bins, _ = discretize_quantile(prices, n_price_bins)
    states = []
    for i in range(1, len(prices)):
        up = 1 if prices[i] >= prices[i - 1] else 0
        states.append(bins[i] * 2 + up)
    return states, 2 * n_price_bins
