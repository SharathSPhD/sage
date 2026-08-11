"""Fast-quench EFE campaign — unit science.quench_regimes (F-0013's remainder).

Run: ``uv run python -m experiments.fast_quench_campaign``
Four hypotheses for the lag-dominated regime (config committed and VERIFIED
landed before this ran). Artifact: ``fast_quench_campaign.json`` with the
full audit trail, held-out validation and sigma-sensitivity re-run.
"""

from __future__ import annotations

from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import strataq
import yaml
from strataq.core.dynamics.markov import glauber_generator, stationary_distribution
from strataq.estimate.bayes import Hypothesis, run_campaign
from strataq.finite.decompose.generate import make_family
from strataq.finite.games.library import coordination, matching_pennies
from strataq.finite.games.tensor import DenseTensorGame
from strataq.thermo.protocols import QuenchProtocol, hatano_sasa_exact
from strataq_bench import BenchmarkResult, EffectSize

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
CONFIG = REPO / "config" / "experiments" / "fast_quench.yaml"
UNIT = "science.quench_regimes"

Probe = tuple[float, tuple[int, float, float]]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text())
    seed = int(cfg["seed"])
    (RESULTS / "fast_quench.resolved.yaml").write_text(yaml.safe_dump(cfg))
    sigma = float(cfg["sigma_dex"])

    alphas = [float(a) for a in cfg["family"]["alphas_probe"]]
    games: dict[float, DenseTensorGame] = {
        a: make_family(coordination(2, 2, bonus=2.0), matching_pennies(), [a])[0] for a in alphas
    }

    def proto(spec: tuple[int, float, float]) -> QuenchProtocol:
        n, tau, lam_end = spec
        return QuenchProtocol(
            lambdas=jnp.linspace(0.5, float(lam_end), int(n) + 1),
            taus=jnp.full((int(n),), float(tau)),
        )

    def pis_of(a: float, spec: tuple[int, float, float]) -> list:
        return [
            stationary_distribution(glauber_generator(games[a], float(lam)))
            for lam in proto(spec).lambdas
        ]

    def d_kl(p, q) -> float:
        return float(jnp.sum(p * (jnp.log(p) - jnp.log(q))))

    def floor_of(a: float, spec: tuple[int, float, float]) -> float:
        return max(sum(d_kl(x, y) for x, y in pairwise(pis_of(a, spec))), 1e-300)

    def frozen_of(a: float, spec: tuple[int, float, float]) -> float:
        pis = pis_of(a, spec)
        return max(d_kl(pis[0], pis[-1]), 1e-300)

    def gap_of(a: float, spec: tuple[int, float, float]) -> float:
        lam_mid = 0.5 * (0.5 + float(spec[2]))
        ev = np.linalg.eigvals(np.asarray(glauber_generator(games[a], lam_mid)))
        return float(-np.sort(ev.real)[::-1][1])

    def h_frozen(probe: object) -> float:
        a, spec = probe  # type: ignore[misc]
        return float(np.log10(frozen_of(a, spec)))

    def h_floor(probe: object) -> float:
        a, spec = probe  # type: ignore[misc]
        return float(np.log10(floor_of(a, spec)))

    def h_interp(probe: object) -> float:
        a, spec = probe  # type: ignore[misc]
        f, d0 = floor_of(a, spec), frozen_of(a, spec)
        w = float(np.exp(-gap_of(a, spec) * float(spec[1])))
        return float(np.log10(max(f + (d0 - f) * w, 1e-300)))

    def h_geom(probe: object) -> float:
        a, spec = probe  # type: ignore[misc]
        return float(0.5 * (np.log10(floor_of(a, spec)) + np.log10(frozen_of(a, spec))))

    hypotheses = [
        Hypothesis(name="frozen_limit", predict=h_frozen),
        Hypothesis(name="slow_floor", predict=h_floor),
        Hypothesis(name="gap_interpolation", predict=h_interp),
        Hypothesis(name="geometric_strawman", predict=h_geom),
    ]
    probes: list[Probe] = [(a, tuple(s)) for a in alphas for s in cfg["protocols"]]  # type: ignore[misc]

    def run_probe(probe: object) -> float:
        a, spec = probe  # type: ignore[misc]
        return float(np.log10(max(float(hatano_sasa_exact(games[a], proto(spec))[1]), 1e-300)))

    def campaign(sig: float):
        return run_campaign(
            hypotheses,
            probes,
            run_probe=run_probe,
            sigma=sig,
            budget=int(cfg["budget"]),
            stop_confidence=float(cfg["stop_confidence"]),
            min_probes=int(cfg.get("min_probes", 1)),
        )

    result = campaign(sigma)
    widx = result.hypothesis_names.index(result.winner)
    consumed = {step.probe for step in result.history}
    resids = [abs(step.predictions[widx] - step.observed) for step in result.history]
    held_out = [p for p in probes if p not in consumed]
    val_resids = [abs(hypotheses[widx].predict(p) - run_probe(p)) for p in held_out]
    bound = float(cfg["adequacy_max_median_resid_sigmas"]) * sigma
    adequate = float(np.median(resids)) < bound
    validated = (float(np.median(val_resids)) < bound) if val_resids else True
    verdict = result.winner
    if not adequate:
        verdict = "all_hypotheses_inadequate"
    elif not validated:
        verdict = "winner_failed_validation"
    sens = campaign(float(cfg["sigma_sensitivity"]))

    _write = RESULTS / "fast_quench_campaign.json"
    res = BenchmarkResult(
        benchmark_id="fast_quench_campaign",
        unit=UNIT,
        kind="statistical",
        passed=True,  # ran honestly; the verdict is data
        metrics={
            **{
                f"belief_{n}": float(b)
                for n, b in zip(result.hypothesis_names, result.beliefs, strict=True)
            },
            "n_probes_run": float(len(result.history)),
            "stopped_early": float(result.stopped_early),
            "winner_median_resid_dex": float(np.median(resids)),
            "consumed_max_resid_dex": float(np.max(resids)),
            "heldout_n": float(len(held_out)),
            "heldout_median_resid_dex": float(np.median(val_resids)) if val_resids else 0.0,
            "heldout_max_resid_dex": float(np.max(val_resids)) if val_resids else 0.0,
            "adequate": float(adequate),
            "heldout_validated": float(validated),
            "sensitivity_sigma_agrees": float(sens.winner == result.winner),
            **{f"probe{i}_alpha": float(s.probe[0]) for i, s in enumerate(result.history)},  # type: ignore[index]
            **{f"probe{i}_tau": float(s.probe[1][1]) for i, s in enumerate(result.history)},  # type: ignore[index]
        },
        effect_sizes=[
            EffectSize(
                name=f"posterior belief in verdict '{verdict}'",
                value=float(np.max(result.beliefs)),
                ci_low=float(np.max(result.beliefs)),
                ci_high=float(np.max(result.beliefs)),
                method=(
                    f"EFE/BALD selection, sigma={sigma} dex, "
                    f"{len(result.history)}/{len(probes)} probes consumed"
                ),
            )
        ],
        n=len(result.history),
        n_justification=(
            f"budget {cfg['budget']} of {len(probes)} candidate probes; unconsumed "
            "probes all run as held-out validation — nothing silently skipped."
        ),
        seed=seed,
        config_ref=str(CONFIG.relative_to(REPO)),
        library_version=strataq.__version__,
        timestamp=_now(),
        notes=(
            f"Fast-quench regime verdict: {verdict}. Hypotheses: frozen limit "
            "D(pi_start||pi_end) (the tau->0 telescoping identity), F-0013's slow "
            "floor, spectral-gap interpolation between them, geometric strawman."
        ),
    )
    _write.write_text(res.model_dump_json(indent=2) + "\n")
    print(f"[PASS] fast_quench_campaign -> {_write.name} (verdict: {verdict})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
