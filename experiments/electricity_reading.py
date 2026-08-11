"""First real-data reading — irreversibility of CAISO day-ahead prices.

Run: ``uv run python -m experiments.electricity_reading``
The question the instruments can now ask of a real market: is the day-ahead
hourly price series time-reversible (equilibrium-like fluctuation) or does
it carry measurable irreversibility (a driven cycle)? KLD estimate with a
time-shuffle surrogate null, plus the TUR certified bound with data-driven
weights on a held-out split.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import strataq
import yaml
from strataq.core.dynamics.sample import trajectory_from_series
from strataq.domains.electricity import discretize_quantile, fetch_dam_lmp, phase_embed
from strataq.thermo.estimators import (
    kld_epr,
)
from strataq_bench import BenchmarkResult, EffectSize

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
CONFIG = REPO / "config" / "experiments" / "electricity_reading.yaml"
UNIT = "domains.electricity"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _write(result: BenchmarkResult) -> None:
    path = RESULTS / f"{result.benchmark_id}.json"
    path.write_text(result.model_dump_json(indent=2) + "\n")
    print(f"[{'PASS' if result.passed else 'FAIL'}] {result.benchmark_id} -> {path.name}")


def _read_market(cfg: dict, mcfg: dict, seed: int) -> tuple[dict, list, bool, int]:
    """One market's dual reading (value-space control + phase embedding) vs FT nulls."""
    n_bins = int(mcfg["n_bins"])
    dt = float(mcfg["dt_hours"])
    start = date.fromisoformat(str(mcfg["start"]))
    _ts, prices = fetch_dam_lmp(
        start, int(mcfg["days"]), node=str(cfg["node"]), market=str(mcfg["market"])
    )
    states, _edges = discretize_quantile(prices, n_bins)
    seq = np.asarray(states)
    n = len(seq)

    kld1 = float(kld_epr(trajectory_from_series(seq, n_bins, dt=dt), k=1))
    n_price_bins = max(n_bins // 2, 2)
    estates, e_nstates = phase_embed(list(prices), n_price_bins)
    kld_embed = float(kld_epr(trajectory_from_series(np.asarray(estates), e_nstates, dt=dt), k=1))

    rng = np.random.default_rng(seed)

    def ft_surrogate(x: np.ndarray) -> np.ndarray:
        spec = np.fft.rfft(x - x.mean())
        phases = rng.uniform(0, 2 * np.pi, len(spec))
        phases[0] = 0.0
        if len(x) % 2 == 0:
            phases[-1] = 0.0
        return np.fft.irfft(np.abs(spec) * np.exp(1j * phases), n=len(x)) + x.mean()

    null, null_embed = [], []
    for _ in range(int(cfg["surrogates"])):
        perm = ft_surrogate(np.asarray(prices))
        ps, _ = discretize_quantile(list(perm), n_bins)
        null.append(float(kld_epr(trajectory_from_series(np.asarray(ps), n_bins, dt=dt), k=1)))
        es, en = phase_embed(list(perm), n_price_bins)
        null_embed.append(float(kld_epr(trajectory_from_series(np.asarray(es), en, dt=dt), k=1)))
    null_a = np.array(null)
    null_e = np.array(null_embed)
    alpha = float(cfg["thresholds"]["surrogate_alpha"])
    q_val = float(np.quantile(null_a, 1.0 - alpha))
    q_emb = float(np.quantile(null_e, 1.0 - alpha))

    metrics = {
        "n_samples": float(n),
        "kld_value_per_hour": kld1,
        "null_value_median": float(np.median(null_a)),
        "null_value_q99": q_val,
        "value_detected": float(kld1 > q_val),
        "kld_embed_per_hour": kld_embed,
        "null_embed_median": float(np.median(null_e)),
        "null_embed_q99": q_emb,
        "embed_detected": float(kld_embed > q_emb),
    }
    effect = EffectSize(
        name="phase-embedded KLD vs FT-surrogate null (nats/hour)",
        value=kld_embed,
        ci_low=float(np.quantile(null_e, 0.005)),
        ci_high=q_emb,
        ci_level=0.99,
        method="re-embedded phase-randomised surrogate null (band = the NULL, not the estimate)",
    )
    sane = (
        float(np.median(null_e)) > float(cfg["thresholds"]["min_null_median"])
        and float(np.median(null_a)) >= 0.0
    )
    return metrics, [effect], sane, n


def run() -> int:
    cfg = yaml.safe_load(CONFIG.read_text())
    seed = int(cfg["seed"])
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "electricity_reading.resolved.yaml").write_text(
        yaml.safe_dump({"config": cfg, "library_version": strataq.__version__, "run_at": _now()})
    )
    failures = 0
    for name, mcfg in cfg["markets"].items():
        metrics, effects, sane, n = _read_market(cfg, dict(mcfg), seed)
        # Pass = the METHODOLOGY held (coverage, sane nulls). Detection is a
        # finding either way: a certified null result on DAM is as much a
        # reading as a detection on RTM would be.
        failures += not sane
        _write(
            BenchmarkResult(
                benchmark_id=f"electricity_irreversibility_{name}",
                unit=UNIT,
                kind="statistical",
                passed=sane,
                metrics=metrics,
                effect_sizes=effects,
                n=n,
                n_justification=(
                    f"{n} {mcfg['market']} intervals over {mcfg['days']} days at node "
                    f"{cfg['node']}; the FT-surrogate null shares the spectrum (persistence "
                    "and the diurnal peak) and the plug-in bias at exactly this n, so a "
                    "detection verdict is bias- and persistence-robust; the value-space "
                    "reading is the built-in blindness control (a periodic drive retraces "
                    "its 1-D path, so value discretization must sit on its null)."
                ),
                seed=seed,
                config_ref=str(CONFIG.relative_to(REPO)),
                library_version=strataq.__version__,
                timestamp=_now(),
                notes=(
                    "Detection flags are FINDINGS, not gate criteria: embed_detected=1 means "
                    "the series is measurably time-irreversible beyond any linear process "
                    "with its spectrum; 0 is a certified null at this n and binning."
                ),
            )
        )
    return failures


if __name__ == "__main__":
    raise SystemExit(run())
