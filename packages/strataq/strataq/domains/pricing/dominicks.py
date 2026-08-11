"""Dominick's canned-soup panel — the pricing domain's data contact.

Source: the PI's processed DreamPrice dataset on Hugging Face
(``qbz506/dreamprice-dominicks-cso``; original data by the Kilts Center,
Chicago Booth; CC-BY-NC-4.0 — derived artifacts inherit that licence).
The HF splits are a RANDOM row subsample of the store×UPC×week panel
(~45 of 399 weeks per store-item), so everything downstream is designed
gap-tolerant: brand-level store-week indices rather than item-level
first differences.

Wholesale cost is recovered from the gross margin:
``cost = price * (1 - PROFIT/100)`` (the Dominick's manual's definition);
rows with implausible margins or missing prices are dropped and counted.
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import polars as pl

CACHE = Path.home() / ".cache" / "strataq" / "dominicks"
BASE = (
    "https://huggingface.co/datasets/qbz506/dreamprice-dominicks-cso/resolve/"
    "refs%2Fconvert%2Fparquet/default"
)
SPLITS = ("train", "validation", "test")
COLUMNS = ["STORE", "UPC", "WEEK", "MOVE", "PRICE", "PROFIT", "SALE", "DESCRIP", "OK"]

# margin sanity window (fractions): outside this the PROFIT field is
# unreliable per the Dominick's manual's known glitches
MARGIN_LO = -0.5
MARGIN_HI = 0.9


def _fetch(split: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{split}.parquet"
    if not path.exists():
        urlretrieve(f"{BASE}/{split}/0000.parquet", path)
    return path


def load_panel() -> pl.DataFrame:
    """The cleaned long panel: store, upc, mfr, week, price, cost, sale flag.

    Drops rows with non-positive price, out-of-window margins, or OK=0, and
    records the drop count as a column-attached metadata frame is overkill —
    callers get (panel, n_dropped) via :func:`load_panel_with_stats` if they
    need the audit number.
    """
    return load_panel_with_stats()[0]


def load_panel_with_stats() -> tuple[pl.DataFrame, int]:
    frames = [pl.read_parquet(_fetch(s), columns=COLUMNS) for s in SPLITS]
    raw = pl.concat(frames)
    n_raw = len(raw)
    df = (
        raw.filter(
            (pl.col("PRICE") > 0)
            & (pl.col("OK") == 1)
            & (pl.col("PROFIT") / 100.0 > MARGIN_LO)
            & (pl.col("PROFIT") / 100.0 < MARGIN_HI)
        )
        .with_columns(
            (pl.col("UPC") // 100000).alias("MFR"),
            (pl.col("PRICE") * (1.0 - pl.col("PROFIT") / 100.0)).alias("COST"),
            pl.col("SALE").is_not_null().alias("ON_SALE"),
        )
        .select("STORE", "UPC", "MFR", "WEEK", "PRICE", "COST", "MOVE", "ON_SALE")
    )
    return df, n_raw - len(df)


def brand_index(panel: pl.DataFrame, mfr: int, *, exclude_sale: bool = True) -> pl.DataFrame:
    """Store-week brand indices: mean log price and mean log cost over the
    manufacturer's UPCs present that week (gap-tolerant by construction)."""
    df = panel.filter(pl.col("MFR") == mfr)
    if exclude_sale:
        df = df.filter(~pl.col("ON_SALE"))
    return (
        df.with_columns(pl.col("PRICE").log().alias("LOGP"), pl.col("COST").log().alias("LOGC"))
        .group_by("STORE", "WEEK")
        .agg(pl.mean("LOGP"), pl.mean("LOGC"), pl.len().alias("N_ITEMS"))
    )


def category_price_series(panel: pl.DataFrame, store: int) -> tuple[list[int], list[float]]:
    """One store's weekly category price index (mean log price, all items)."""
    df = (
        panel.filter(pl.col("STORE") == store)
        .with_columns(pl.col("PRICE").log().alias("LOGP"))
        .group_by("WEEK")
        .agg(pl.mean("LOGP"))
        .sort("WEEK")
    )
    return df["WEEK"].to_list(), df["LOGP"].to_list()
