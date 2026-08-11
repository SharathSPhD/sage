"""Dominick's pricing readings — unit domains.pricing (plan-v2 R4).

Run: ``uv run python -m experiments.pricing_reading``
Two artifacts: (1) the programme's headline empirical promise — the
reciprocity read from cross-brand cost pass-through (N2/C2), with the
single-retailer prediction stated ex-ante; (2) an Edgeworth-cycle scan:
per-store weekly category-price irreversibility against the
reversibilized-Markov null established in F-0009.
"""

from __future__ import annotations

from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
import polars as pl
import strataq
import yaml
from scipy.stats import beta as sp_beta
from strataq.core.dynamics.sample import trajectory_from_series
from strataq.domains.electricity import phase_embed
from strataq.domains.pricing import brand_index, category_price_series, load_panel_with_stats
from strataq.thermo.estimators import kld_epr
from strataq_bench import BenchmarkResult, EffectSize

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
CONFIG = REPO / "config" / "experiments" / "pricing_reading.yaml"
UNIT = "domains.pricing"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _write(result: BenchmarkResult) -> None:
    path = RESULTS / f"{result.benchmark_id}.json"
    path.write_text(result.model_dump_json(indent=2) + "\n")
    print(f"[{'PASS' if result.passed else 'FAIL'}] {result.benchmark_id} -> {path.name}")


def _two_way_demean(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Approximate two-way (store, week) within transform by sequential demeaning."""
    out = df
    for _ in range(3):  # a few alternating projections converge fast here
        for key in ("STORE", "WEEK"):
            means = out.group_by(key).agg([pl.mean(c).alias(f"_m_{c}") for c in cols])
            out = out.join(means, on=key)
            out = out.with_columns([(pl.col(c) - pl.col(f"_m_{c}")).alias(c) for c in cols]).drop(
                [f"_m_{c}" for c in cols]
            )
    return out


def _chi(joined: pl.DataFrame) -> np.ndarray:
    """2x2 pass-through matrix from the demeaned brand panel by OLS."""
    y_c = joined["LOGP"].to_numpy()
    y_p = joined["LOGP_P"].to_numpy()
    x = np.column_stack([joined["LOGC"].to_numpy(), joined["LOGC_P"].to_numpy()])
    beta_c, *_ = np.linalg.lstsq(x, y_c, rcond=None)
    beta_p, *_ = np.linalg.lstsq(x, y_p, rcond=None)
    # rows: which price responds; cols: [campbell cost, progresso cost] —
    # both regressions share the same regressor order, so both rows keep it.
    return np.array([[beta_c[0], beta_c[1]], [beta_p[0], beta_p[1]]])


def _r_of(chi: np.ndarray) -> float:
    return float(np.linalg.norm(chi - chi.T) / max(np.linalg.norm(chi + chi.T), 1e-12))


def passthrough(cfg: dict, panel: pl.DataFrame, seed: int) -> int:
    br = cfg["brands"]
    camp = brand_index(panel, int(br["campbell"]), exclude_sale=bool(cfg["exclude_sale"]))
    prog = brand_index(panel, int(br["progresso"]), exclude_sale=bool(cfg["exclude_sale"]))
    joined = camp.join(prog, on=["STORE", "WEEK"], suffix="_P").drop_nulls()
    cols = ["LOGP", "LOGC", "LOGP_P", "LOGC_P"]
    demeaned = _two_way_demean(joined.select(["STORE", "WEEK", *cols]), cols)

    chi = _chi(demeaned)
    r_hat = _r_of(chi)
    asym = float(chi[0, 1] - chi[1, 0])

    rng = np.random.default_rng(seed)
    stores = demeaned["STORE"].unique().to_list()
    raw = joined.select(["STORE", "WEEK", *cols])
    boots_r, boots_asym = [], []
    for _ in range(int(cfg["passthrough"]["bootstrap"])):
        pick = rng.choice(stores, size=len(stores), replace=True)
        # relabel duplicated draws as distinct clusters so the within
        # transform is recomputed on each resample (standard cluster
        # bootstrap; demeaning is part of the estimator, so it resamples too)
        sub = pl.concat(
            [
                raw.filter(pl.col("STORE") == s).with_columns(pl.lit(i).alias("STORE"))
                for i, s in enumerate(pick)
            ]
        )
        c = _chi(_two_way_demean(sub, cols))
        boots_r.append(_r_of(c))
        boots_asym.append(float(c[0, 1] - c[1, 0]))
    r_lo, r_hi = np.percentile(boots_r, [2.5, 97.5])
    a_lo, a_hi = np.percentile(boots_asym, [2.5, 97.5])

    symmetric = r_hat < float(cfg["passthrough"]["symmetric_R_max"]) and (a_lo <= 0 <= a_hi)
    _write(
        BenchmarkResult(
            benchmark_id="pricing_passthrough_R",
            unit=UNIT,
            kind="statistical",
            passed=True,  # pass = the estimation ran honestly; the verdict is data
            metrics={
                "chi_cc": float(chi[0, 0]),
                "chi_cp": float(chi[0, 1]),
                "chi_pc": float(chi[1, 0]),  # dP_prog/dC_camp
                "chi_pp": float(chi[1, 1]),
                "R_empirical": r_hat,
                "R_ci_low": float(r_lo),
                "R_ci_high": float(r_hi),
                "asymmetry": asym,
                "asymmetry_ci_low": float(a_lo),
                "asymmetry_ci_high": float(a_hi),
                "symmetric_verdict": float(symmetric),
                "n_store_weeks": float(len(joined)),
                "n_stores": float(len(stores)),
            },
            effect_sizes=[
                EffectSize(
                    name="empirical reciprocity defect R (2-brand pass-through)",
                    value=r_hat,
                    ci_low=float(r_lo),
                    ci_high=float(r_hi),
                    method="cluster bootstrap over stores (500 resamples)",
                )
            ],
            n=len(joined),
            n_justification=(
                f"{len(joined)} store-weeks with both brand indices present across "
                f"{len(stores)} stores; two-way (store, week) demeaning; cluster "
                "bootstrap respects within-store dependence."
            ),
            seed=seed,
            config_ref=str(CONFIG.relative_to(REPO)),
            library_version=strataq.__version__,
            timestamp=_now(),
            notes=(
                "The programme's N2 promise on real data: R from cross-brand wholesale-"
                "cost pass-through. IDENTIFICATION HONESTY: this is ONE retailer pricing "
                "both brands (category management), so the ex-ante prediction — stated in "
                "the config before the run — is near-SYMMETRIC cross-pass-through "
                "(potential-like, small R with CI covering zero asymmetry); a large R "
                "would falsify the single-objective model of category pricing. Regular "
                "(non-sale) prices only; brand indices are store-week means over each "
                "manufacturer's UPCs (gap-tolerant to the HF row subsample)."
            ),
        )
    )
    return 0


def edgeworth(cfg: dict, panel: pl.DataFrame, seed: int) -> int:
    ec = cfg["edgeworth"]
    n_bins = int(ec["n_bins"])
    rng = np.random.default_rng(seed)
    best = (
        panel.group_by("STORE").agg(pl.len()).sort("len", descending=True).head(int(ec["n_stores"]))
    )["STORE"].to_list()
    detections = 0
    stats = []
    for store in best:
        _weeks, logp = category_price_series(panel, int(store))
        if len(logp) < 200:
            continue
        estates, e_n = phase_embed([float(x) for x in logp], n_bins)
        eseq = np.asarray(estates)
        s = float(kld_epr(trajectory_from_series(eseq, e_n), k=1))
        counts = np.zeros((e_n, e_n))
        for a, b in pairwise(eseq):
            counts[a, b] += 1.0
        sym = (counts + counts.T) / 2.0
        rs = sym.sum(axis=1, keepdims=True)
        p_rev = np.divide(sym, rs, out=np.full_like(sym, 1.0 / e_n), where=rs > 0)
        null = []
        for _ in range(int(ec["surrogates"])):
            out = np.empty(len(eseq), dtype=np.int64)
            out[0] = eseq[0]
            for i in range(1, len(eseq)):
                out[i] = rng.choice(e_n, p=p_rev[out[i - 1]])
            null.append(float(kld_epr(trajectory_from_series(out, e_n), k=1)))
        q99 = float(np.quantile(null, 1.0 - float(ec["alpha"])))
        det = s > q99
        detections += int(det)
        stats.append((int(store), s, q99, det))
    n_tested = len(stats)
    expected_fp = float(ec["alpha"]) * n_tested
    _write(
        BenchmarkResult(
            benchmark_id="pricing_edgeworth_scan",
            unit=UNIT,
            kind="statistical",
            passed=True,  # the scan's verdict is a finding either way
            metrics={
                "n_stores_tested": float(n_tested),
                "n_detections": float(detections),
                "expected_false_positives": expected_fp,
                "detection_rate": detections / max(n_tested, 1),
                **{f"store_{s}_stat": v for s, v, _, _ in stats[:8]},
            },
            effect_sizes=[
                EffectSize(
                    name="Edgeworth detection rate (per-store weekly category indices)",
                    value=detections / max(n_tested, 1),
                    ci_low=float(sp_beta.ppf(0.025, detections + 0.5, n_tested - detections + 0.5)),
                    ci_high=float(
                        sp_beta.ppf(0.975, detections + 0.5, n_tested - detections + 0.5)
                    ),
                    method=f"Jeffreys binomial interval over {n_tested} stores",
                )
            ],
            n=n_tested,
            n_justification=(
                f"{n_tested} best-covered stores, ~350-400 weekly category-price points "
                "each; per-store reversibilized-Markov null (the F-0009 machinery) at "
                f"alpha={ec['alpha']}; the store count makes the binomial comparison "
                "against the false-positive expectation meaningful."
            ),
            seed=seed,
            config_ref=str(CONFIG.relative_to(REPO)),
            library_version=strataq.__version__,
            timestamp=_now(),
            notes=(
                "Edgeworth-cycle scan: detections >> alpha*n means retail category "
                "prices carry asymmetric-cycle irreversibility (the Maskin-Tirole "
                "signature); detections ~ alpha*n is a certified at-null for weekly "
                "category indices at this resolution. Week-number gaps from the HF "
                "subsample are treated as consecutive observations (noted limitation: "
                "gaps blur transitions toward reversibility, so detections are "
                "conservative)."
            ),
        )
    )
    return 0


def run() -> int:
    cfg = yaml.safe_load(CONFIG.read_text())
    seed = int(cfg["seed"])
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "pricing_reading.resolved.yaml").write_text(
        yaml.safe_dump({"config": cfg, "library_version": strataq.__version__, "run_at": _now()})
    )
    panel, dropped = load_panel_with_stats()
    print(f"panel {len(panel)} rows ({dropped} dropped by cleaning)")
    failures = 0
    failures += passthrough(cfg, panel, seed)
    failures += edgeworth(cfg, panel, seed)
    return failures


if __name__ == "__main__":
    raise SystemExit(run())
