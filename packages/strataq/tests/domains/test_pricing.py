"""Dominick's pricing loader (unit domains.pricing)."""

import polars as pl
from strataq.domains.pricing import brand_index, category_price_series, load_panel_with_stats

CAMPBELL = 51000
PROGRESSO = 41196


class TestPanel:
    def test_load_and_clean(self):
        panel, dropped = load_panel_with_stats()
        assert len(panel) > 400_000
        assert dropped < 0.1 * (len(panel) + dropped)  # <10% dropped, counted
        assert panel["PRICE"].min() > 0
        # cost recovery: cost = price*(1-margin); margins inside the window
        margins = 1.0 - panel["COST"] / panel["PRICE"]
        assert float(margins.min()) > -0.5 and float(margins.max()) < 0.9

    def test_brand_indices_cover_most_store_weeks(self):
        panel, _ = load_panel_with_stats()
        camp = brand_index(panel, CAMPBELL)
        prog = brand_index(panel, PROGRESSO)
        joined = camp.join(prog, on=["STORE", "WEEK"], suffix="_P")
        assert len(joined) > 15_000  # both brands priced in most store-weeks
        assert float(joined["N_ITEMS"].mean()) > 3  # dense Campbell index

    def test_category_series_long_enough(self):
        panel, _ = load_panel_with_stats()
        store = int(panel.group_by("STORE").agg(pl.len()).sort("len", descending=True)["STORE"][0])
        weeks, logp = category_price_series(panel, store)
        assert len(weeks) > 300
        assert weeks == sorted(weeks)
        assert len(logp) == len(weeks)
