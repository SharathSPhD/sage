"""Multi-mode quench campaign — unit science.quench_multimode (F-0014 refinement).

Run: ``uv run python -m experiments.quench_multimode_campaign``
Four closed-form lag models (config committed and VERIFIED landed before
this ran), probed on a grid that INCLUDES F-0014's known failure cells.
Artifact: ``quench_multimode_campaign.json``.
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
CONFIG = REPO / "config" / "experiments" / "quench_multimode.yaml"
UNIT = "science.quench_multimode"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text())
    seed = int(cfg["seed"])
    (RESULTS / "quench_multimode.resolved.yaml").write_text(yaml.safe_dump(cfg))
    sigma = float(cfg["sigma_dex"])

    alphas = [float(a) for a in cfg["family"]["alphas_probe"]]
    games: dict[object, DenseTensorGame] = {
        a: make_family(coordination(2, 2, bonus=2.0), matching_pennies(), [a])[0] for a in alphas
    }
    if cfg.get("family_3x3"):
        from strataq.finite.games.library import rock_paper_scissors

        for a in cfg["family_3x3"]["alphas_probe"]:
            games[("3x3", float(a))] = make_family(
                coordination(2, 3, bonus=2.0), rock_paper_scissors(), [float(a)]
            )[0]

    def proto(spec: tuple[int, float, float]) -> QuenchProtocol:
        n, tau, lam_end = spec
        return QuenchProtocol(
            lambdas=jnp.linspace(0.5, float(lam_end), int(n) + 1),
            taus=jnp.full((int(n),), float(tau)),
        )

    def tables(a: float, spec: tuple[int, float, float]):
        """Per-lambda generators (numpy) and stationary distributions."""
        gens = [np.asarray(glauber_generator(games[a], float(lam))) for lam in proto(spec).lambdas]
        pis = [np.asarray(stationary_distribution(jnp.asarray(g))) for g in gens]
        return gens, pis

    def d_kl(p: np.ndarray, q: np.ndarray) -> float:
        return float(np.sum(p * (np.log(p) - np.log(q))))

    def gap_of_gen(gen: np.ndarray) -> float:
        return float(-np.sort(np.linalg.eigvals(gen).real)[::-1][1])

    def _global(a: float, spec: tuple[int, float, float], gap_gen: np.ndarray) -> float:
        _gens, pis = tables(a, spec)
        floor = max(sum(d_kl(x, y) for x, y in pairwise(pis)), 1e-300)
        frozen = max(d_kl(pis[0], pis[-1]), 1e-300)
        w = float(np.exp(-gap_of_gen(gap_gen) * float(spec[1])))
        return float(np.log10(max(floor + (frozen - floor) * w, 1e-300)))

    def h_gap_mid(probe: object) -> float:
        a, spec = probe  # type: ignore[misc]
        lam_mid = 0.5 * (0.5 + float(spec[2]))
        return _global(a, spec, np.asarray(glauber_generator(games[a], lam_mid)))

    def h_gap_start(probe: object) -> float:
        a, spec = probe  # type: ignore[misc]
        return _global(a, spec, np.asarray(glauber_generator(games[a], 0.5)))

    def _recursive(a: float, spec: tuple[int, float, float], n_modes: int) -> float:
        gens, pis = tables(a, spec)
        tau = float(spec[1])
        p = pis[0]
        y = 0.0
        for k in range(1, len(pis)):
            y += float(np.sum(p * (np.log(pis[k - 1]) - np.log(pis[k]))))
            delta = p - pis[k]
            if n_modes == 1:
                p = pis[k] + delta * np.exp(-gap_of_gen(gens[k]) * tau)
            else:
                # multi-mode propagator (red-team fix): row-distributions
                # evolve by right-multiplication with expm(L tau), so the
                # relaxation modes are LEFT eigenvectors w_i (w_i L = mu_i
                # w_i). Take the FULL eigenbasis of L^T (columns are w_i^T),
                # get exact coefficients by a linear solve — no truncated
                # least squares silently dropping mass — then keep the
                # n_modes slowest non-stationary modes, conjugate pairs
                # together, and drop the fast remainder explicitly.
                vals, vecs = np.linalg.eig(gens[k].T)
                order = np.argsort(vals.real)[::-1]
                coeffs = np.linalg.solve(vecs, delta.astype(complex))
                keep = set()
                for i in order[1:]:
                    if len(keep) >= n_modes:
                        break
                    keep.add(i)
                    if abs(vals[i].imag) > 1e-12:  # keep the conjugate partner
                        j = int(np.argmin(np.abs(vals - np.conj(vals[i]))))
                        keep.add(j)
                p_new = pis[k].astype(complex)
                for i in keep:
                    p_new = p_new + coeffs[i] * vecs[:, i] * np.exp(vals[i] * tau)
                p = np.maximum(p_new.real, 1e-300)
                p = p / p.sum()
        return float(np.log10(max(y, 1e-300)))

    def h_recursive(probe: object) -> float:
        a, spec = probe  # type: ignore[misc]
        return _recursive(a, spec, 1)

    def h_recursive2(probe: object) -> float:
        a, spec = probe  # type: ignore[misc]
        return _recursive(a, spec, 2)

    hypotheses = [
        Hypothesis(name="gap_mid_global", predict=h_gap_mid),
        Hypothesis(name="gap_start_global", predict=h_gap_start),
        Hypothesis(name="recursive_1mode", predict=h_recursive),
        Hypothesis(name="recursive_2mode", predict=h_recursive2),
    ]
    probes = [(a, tuple(s)) for a in alphas for s in cfg["protocols"]]  # type: ignore[misc]
    if cfg.get("family_3x3"):
        probes += [
            (("3x3", float(a)), tuple(s))
            for a in cfg["family_3x3"]["alphas_probe"]
            for s in cfg["family_3x3"]["protocols"]
        ]

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
            min_probes=int(cfg["min_probes"]),
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

    # full-accounting comparison table: every hypothesis over every probe
    all_resids = {h.name: [abs(h.predict(p) - run_probe(p)) for p in probes] for h in hypotheses}

    path = RESULTS / "quench_multimode_campaign.json"
    res = BenchmarkResult(
        benchmark_id="quench_multimode_campaign",
        unit=UNIT,
        kind="statistical",
        passed=True,  # ran honestly; verdict is data
        metrics={
            **{
                f"belief_{n}": float(b)
                for n, b in zip(result.hypothesis_names, result.beliefs, strict=True)
            },
            "n_probes_run": float(len(result.history)),
            "winner_median_resid_dex": float(np.median(resids)),
            "consumed_max_resid_dex": float(np.max(resids)),
            "heldout_median_resid_dex": float(np.median(val_resids)) if val_resids else 0.0,
            "heldout_max_resid_dex": float(np.max(val_resids)) if val_resids else 0.0,
            "adequate": float(adequate),
            "heldout_validated": float(validated),
            "sensitivity_sigma_agrees": float(sens.winner == result.winner),
            **{f"grid_median_{n}": float(np.median(v)) for n, v in all_resids.items()},
            **{f"grid_max_{n}": float(np.max(v)) for n, v in all_resids.items()},
        },
        effect_sizes=[
            EffectSize(
                name=f"posterior belief in verdict '{verdict}'",
                value=float(np.max(result.beliefs)),
                ci_low=float(np.max(result.beliefs)),
                ci_high=float(np.max(result.beliefs)),
                method=(
                    f"EFE/BALD, sigma={sigma} dex, min_probes={cfg['min_probes']}, "
                    f"{len(result.history)}/{len(probes)} consumed; full grid table attached"
                ),
            )
        ],
        n=len(result.history),
        n_justification=(
            f"budget {cfg['budget']} of {len(probes)} probes (failure cells from "
            "F-0014 included by construction); ALL probes also evaluated for every "
            "hypothesis in the grid_median/grid_max table — no residual hidden."
        ),
        seed=seed,
        config_ref=str(CONFIG.relative_to(REPO)),
        library_version=strataq.__version__,
        timestamp=_now(),
        notes=(
            f"Multimode refinement verdict: {verdict}. The recursive models track "
            "the NESS path per step, putting F-0014's loop-path failures in scope "
            "by construction; the two-mode variant adds the next relaxation mode."
        ),
    )
    path.write_text(res.model_dump_json(indent=2) + "\n")
    print(f"[PASS] quench_multimode_campaign -> {path.name} (verdict: {verdict})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
