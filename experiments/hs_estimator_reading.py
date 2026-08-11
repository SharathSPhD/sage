"""HS-estimator validation sweep — unit thermo.hs_estimator.

Run: ``uv run python -m experiments.hs_estimator_reading``
Pre-registered P1–P3 (config committed and verified landed; the regime
change from the powerless alpha=0.5 setting is recorded in the config,
made during TDD before this experiment ever ran). Artifact:
``hs_estimator_sweep.json``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import jax.numpy as jnp
import strataq
import yaml
from strataq.finite.decompose.generate import make_family
from strataq.finite.games.library import coordination, matching_pennies
from strataq.thermo.hs_estimator import hs_y_estimate, sample_quench_states
from strataq.thermo.protocols import QuenchProtocol, hatano_sasa_exact
from strataq_bench import BenchmarkResult, EffectSize

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
CONFIG = REPO / "config" / "experiments" / "hs_estimator_v2.yaml"
UNIT = "thermo.hs_estimator"


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text())
    seed = int(cfg["seed"])
    (RESULTS / "hs_estimator.resolved.yaml").write_text(yaml.safe_dump(cfg))

    game = make_family(
        coordination(2, 2, bonus=2.0), matching_pennies(), [float(cfg["family"]["alpha"])]
    )[0]
    lams = jnp.array([float(x) for x in cfg["protocol"]["lambdas"]])
    taus = [float(t) for t in cfg["taus"]]

    rows = []
    for i, tau in enumerate(taus):
        proto = QuenchProtocol(lambdas=lams, taus=jnp.full((len(lams) - 1,), tau))
        exact = float(hatano_sasa_exact(game, proto)[1])
        windows = sample_quench_states(
            game,
            proto,
            n_trajectories=int(cfg["n_trajectories"]),
            steps_per_unit_time=int(cfg["steps_per_unit_time"]),
            seed=seed + i,
        )
        est = hs_y_estimate(
            windows,
            n_states=4,
            hold_durations=[tau] * len(windows),
            pseudocount=float(cfg["pseudocount"]),
            relax_safety=float(cfg.get("relax_safety", 3.0)),
            ift_tolerance=float(cfg.get("ift_tolerance", 0.05)),
        )
        rows.append((tau, exact, est))

    usable_rows = [(t, ex, e) for t, ex, e in rows if e.usable]
    usable_seq = [e.usable for _, _, e in rows]  # taus ascending
    c1 = usable_seq == sorted(usable_seq)
    c2 = bool(usable_rows) and all(
        e.mean_y_ci_low <= ex <= e.mean_y_ci_high for _, ex, e in usable_rows
    )
    c3 = all(not e.usable for t, _, e in rows if t <= 2.0)

    # C4: cross-game (alpha=0, slow basin hopping)
    cg = cfg["cross_game"]
    g0 = make_family(coordination(2, 2, bonus=2.0), matching_pennies(), [float(cg["alpha"])])[0]
    refuse_ok = True
    for t_ref in cg["taus_refuse"]:
        proto0 = QuenchProtocol(lambdas=lams, taus=jnp.full((len(lams) - 1,), float(t_ref)))
        w0 = sample_quench_states(
            g0,
            proto0,
            n_trajectories=int(cg["n_trajectories"]),
            steps_per_unit_time=int(cg["steps_per_unit_time"]),
            seed=seed,
        )
        e0 = hs_y_estimate(w0, n_states=4, hold_durations=[float(t_ref)] * len(w0))
        refuse_ok = refuse_ok and not e0.usable
    proto_a = QuenchProtocol(lambdas=lams, taus=jnp.full((len(lams) - 1,), float(cg["tau_admit"])))
    exact_a = float(hatano_sasa_exact(g0, proto_a)[1])
    admit_us = admit_cov = 0
    for s in range(int(cg["n_seeds"])):
        wa = sample_quench_states(
            g0,
            proto_a,
            n_trajectories=int(cg["n_trajectories"]),
            steps_per_unit_time=int(cg["steps_per_unit_time"]),
            seed=seed + s,
        )
        ea = hs_y_estimate(wa, n_states=4, hold_durations=[float(cg["tau_admit"])] * len(wa))
        admit_us += int(ea.usable)
        if ea.usable:
            admit_cov += int(ea.mean_y_ci_low <= exact_a <= ea.mean_y_ci_high)
    c4 = refuse_ok and admit_us * 2 >= int(cg["n_seeds"]) and admit_cov == admit_us

    # C5: 20-seed calibration at tau=32
    cal = cfg["calibration"]
    proto_c = QuenchProtocol(lambdas=lams, taus=jnp.full((len(lams) - 1,), float(cal["tau"])))
    exact_c = float(hatano_sasa_exact(game, proto_c)[1])
    cov_c = 0
    for s in range(int(cal["n_seeds"])):
        wc = sample_quench_states(
            game,
            proto_c,
            n_trajectories=int(cfg["n_trajectories"]),
            steps_per_unit_time=int(cfg["steps_per_unit_time"]),
            seed=seed + 1000 + s,
        )
        ec = hs_y_estimate(wc, n_states=4, hold_durations=[float(cal["tau"])] * len(wc))
        cov_c += int(ec.mean_y_ci_low <= exact_c <= ec.mean_y_ci_high)
    lo_band, hi_band = (int(x) for x in cal["coverage_band"])
    c5 = lo_band <= cov_c <= hi_band

    metrics: dict[str, float] = {
        "c1_monotone_boundary": float(c1),
        "c2_admitted_holds_cover": float(c2),
        "c3_short_holds_refused": float(c3),
        "c4_cross_game_alpha0": float(c4),
        "c5_calibration_20seed": float(c5),
        "c4_admit_usable": float(admit_us),
        "c5_coverage_count": float(cov_c),
    }
    for t, ex, e in rows:
        key = str(t).replace(".", "p")
        metrics[f"tau{key}_exact"] = ex
        metrics[f"tau{key}_est"] = e.mean_y
        metrics[f"tau{key}_ift"] = e.ift_estimate
        metrics[f"tau{key}_usable"] = float(e.usable)

    t_long, ex_long, e_long = (usable_rows or rows)[-1]
    res = BenchmarkResult(
        benchmark_id="hs_estimator_sweep",
        unit=UNIT,
        kind="statistical",
        passed=bool(c1 and c2 and c3 and c4 and c5),
        metrics=metrics,
        effect_sizes=[
            EffectSize(
                name=f"mean Y_hat at tau={t_long} (exact {ex_long:.4f})",
                value=e_long.mean_y,
                ci_low=e_long.mean_y_ci_low,
                ci_high=e_long.mean_y_ci_high,
                method=f"CLT over {cfg['n_trajectories']} sampled quench trajectories",
            )
        ],
        n=int(cfg["n_trajectories"]) * len(taus),
        n_justification=(
            f"{cfg['n_trajectories']} trajectories x {len(taus)} hold lengths: CI "
            "half-width ~0.03 nats resolves the exact <Y> ~ 0.09-0.25 signal; the "
            "originally registered alpha=0.5 regime had <Y> ~ 0.01 (powerless) and "
            "was replaced during TDD, recorded in config, before this first run."
        ),
        seed=seed,
        config_ref=str(CONFIG.relative_to(REPO)),
        library_version=strataq.__version__,
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        notes=(
            "The data-facing Hatano-Sasa meter: plug-in pi_hat from burned-in hold "
            "windows; the IFT is the instrument's own validity diagnostic. HONESTY "
            "LIMIT (measured during design, in docs): the diagnostic is necessary "
            "but NOT sufficient — <e^{-Y_hat}> can sit at 1 while <Y_hat> retains "
            "residual plug-in bias ~0.03 nats; quote mean_y with its CI, never alone."
        ),
    )
    path = RESULTS / "hs_estimator_sweep.json"
    path.write_text(res.model_dump_json(indent=2) + "\n")
    verdicts = [f"tau={t}: {'OK' if e.usable else 'flagged'}" for t, _, e in rows]
    print(f"[{'PASS' if res.passed else 'FAIL'}] hs_estimator_sweep -> {path.name}")
    print("  " + "; ".join(verdicts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
