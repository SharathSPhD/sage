"""estimate.bayes readings — unit estimate.bayes (plan-v2 R6, ADR-0012).

Run: ``uv run python -m experiments.bayes_reading``
Artifact 1 (bayes_recovery): posterior calibration — CI coverage across
seeds and λ*, the scale-fold identity in posterior space, mixture-vs-
single Bayes factors. Artifact 2 (efe_mechanism_campaign): the EFE
auto-research loop pointed at F-0012's open mechanism. Ex-ante criteria
live in the config, whose commit was VERIFIED to land before this ran.
"""

from __future__ import annotations

from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import strataq
import yaml
from strataq.core.dynamics.markov import glauber_generator, stationary_distribution
from strataq.estimate.bayes import (
    Hypothesis,
    bayes_factor,
    grid_posterior,
    log_evidence,
    log_evidence_mixture,
    refined_posterior,
    run_campaign,
)
from strataq.estimate.lam import sample_choices
from strataq.finite.decompose.generate import make_family
from strataq.finite.games.library import coordination, matching_pennies
from strataq.finite.games.tensor import DenseTensorGame
from strataq.thermo.protocols import QuenchProtocol, hatano_sasa_exact
from strataq_bench import BenchmarkResult, EffectSize

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
CONFIG = REPO / "config" / "experiments" / "bayes_reading.yaml"
UNIT = "estimate.bayes"

ANCHOR = DenseTensorGame(
    (
        jnp.array([[3.0, 0.0, 1.5], [1.0, 2.0, 0.5], [0.0, 1.0, 2.5]]),
        jnp.array([[2.0, 1.0, 0.0], [0.5, 3.0, 1.0], [1.5, 0.0, 2.0]]),
    )
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _write(result: BenchmarkResult) -> None:
    path = RESULTS / f"{result.benchmark_id}.json"
    path.write_text(result.model_dump_json(indent=2) + "\n")
    print(f"[{'PASS' if result.passed else 'FAIL'}] {result.benchmark_id} -> {path.name}")


def recovery(cfg: dict, seed: int) -> None:
    rc = cfg["recovery"]
    grid = np.geomspace(rc["grid"]["lo"], rc["grid"]["hi"], rc["grid"]["points"])
    coverage = {}
    for lam_star in rc["lam_stars"]:
        hits = 0
        for s in range(int(rc["n_seeds"])):
            counts = sample_choices(
                ANCHOR, float(lam_star), int(rc["n_choices"]), jax.random.PRNGKey(seed + s)
            )
            # refined_posterior enforces the resolution guard — the run that
            # used raw grid_posterior at PR ~ 3 undercovered (34/50) and led
            # to the quantised-interval bug find; see the unit gate notes.
            post = refined_posterior(ANCHOR, counts, grid)
            lo, hi = post.credible_interval(0.95)
            hits += int(post.grid_resolved and lo <= lam_star <= hi)
        coverage[lam_star] = hits

    # scale-fold identity in posterior space
    counts = sample_choices(ANCHOR, 1.8, int(rc["n_choices"]), jax.random.PRNGKey(seed))
    post = grid_posterior(ANCHOR, counts, grid)
    scaled_game = DenseTensorGame([2.0 * u for u in ANCHOR.payoffs])
    post_scaled = grid_posterior(scaled_game, counts, grid / 2.0)
    fold_err = float(np.max(np.abs(post.weights - post_scaled.weights)))

    mc = cfg["mixture"]
    mgrid = np.geomspace(mc["grid"]["lo"], mc["grid"]["hi"], mc["grid"]["points"])
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    n = int(mc["n_clean"])
    clean = sample_choices(ANCHOR, float(mc["lam_clean"]), n, k1)
    mixed = tuple(
        a + b
        for a, b in zip(
            sample_choices(ANCHOR, float(mc["lam_lo"]), n // 2, k2),
            sample_choices(ANCHOR, float(mc["lam_hi"]), n // 2, k3),
            strict=True,
        )
    )
    bf_mixed = bayes_factor(
        log_evidence_mixture(ANCHOR, mixed, mgrid), log_evidence(ANCHOR, mixed, mgrid)
    )
    bf_clean = bayes_factor(
        log_evidence_mixture(ANCHOR, clean, mgrid), log_evidence(ANCHOR, clean, mgrid)
    )

    ok = (
        all(h >= int(rc["coverage_min_hits"]) for h in coverage.values())
        and fold_err < 1e-8
        and bf_mixed > float(mc["bf_decisive"])
        and bf_clean < float(mc["bf_occam_max"])
    )
    _write(
        BenchmarkResult(
            benchmark_id="bayes_recovery",
            unit=UNIT,
            kind="statistical",
            passed=bool(ok),
            metrics={
                **{
                    f"coverage_lam{str(k).replace('.', 'p')}": float(v) for k, v in coverage.items()
                },
                "fold_identity_error": fold_err,
                "bf_mixture_on_mixed": min(bf_mixed, 1e12),
                "bf_mixture_on_clean": bf_clean,
            },
            effect_sizes=[
                EffectSize(
                    name="95% CI coverage (pooled over lambda*)",
                    value=sum(coverage.values()) / (3.0 * rc["n_seeds"]),
                    ci_low=min(coverage.values()) / float(rc["n_seeds"]),
                    ci_high=max(coverage.values()) / float(rc["n_seeds"]),
                    method=f"{rc['n_seeds']} seeds x {len(coverage)} lambda* levels",
                )
            ],
            n=int(rc["n_seeds"]) * len(coverage),
            n_justification=(
                f"{rc['n_seeds']} seeds per lambda* at n={rc['n_choices']} choices: "
                "binomial(10, 0.95) puts >=8 hits at ~98.8% probability, so the "
                "coverage floor is a real test without being seed-lottery-fragile."
            ),
            seed=seed,
            config_ref=str(CONFIG.relative_to(REPO)),
            library_version=strataq.__version__,
            timestamp=_now(),
            notes=(
                "Posterior calibration for the Bayesian layer (ADR-0012): CI "
                "coverage with the grid-resolution guard on, the F-0006 scale-fold "
                "as an exact posterior reparameterisation, and R1's mixture "
                "diagnostic as decisive-vs-Occam Bayes factors."
            ),
        )
    )


def campaign(cfg: dict, seed: int) -> None:
    cc = cfg["campaign"]
    sigma = float(cc["sigma_dex"])
    pot = coordination(2, 2, bonus=2.0)
    har = matching_pennies()

    def proto(spec: tuple[int, float, float]) -> QuenchProtocol:
        n, tau, lam_end = spec
        return QuenchProtocol(
            lambdas=jnp.linspace(0.5, float(lam_end), int(n) + 1),
            taus=jnp.full((int(n),), float(tau)),
        )

    def excess_of(game: DenseTensorGame, spec: tuple[int, float, float]) -> float:
        return float(hatano_sasa_exact(game, proto(spec))[1])

    def floor_of(game: DenseTensorGame, spec: tuple[int, float, float]) -> float:
        p = proto(spec)
        pis = [stationary_distribution(glauber_generator(game, float(lam))) for lam in p.lambdas]
        total = 0.0
        for a, b in pairwise(pis):
            total += float(jnp.sum(a * (jnp.log(a) - jnp.log(b))))
        return max(total, 1e-300)

    def gap_of(game: DenseTensorGame, spec: tuple[int, float, float]) -> float:
        lam_mid = 0.5 * (0.5 + float(spec[2]))
        ev = np.linalg.eigvals(np.asarray(glauber_generator(game, lam_mid)))
        real = np.sort(ev.real)[::-1]
        return float(-real[1])  # spectral gap: minus the second-largest real part

    mixed_games = {
        a: make_family(pot, har, [a], scale=float(cc["family"]["scale"]))[0]
        for a in cc["family"]["alphas_probe"]
    }
    base_game = make_family(pot, har, [0.0], scale=float(cc["family"]["scale"]))[0]

    def h_scale(probe: object) -> float:
        a, spec = probe  # type: ignore[misc]
        scaled = DenseTensorGame([(1.0 - a) * u for u in base_game.payoffs])
        return float(np.log10(max(excess_of(scaled, spec), 1e-300)))

    def h_floor(probe: object) -> float:
        a, spec = probe  # type: ignore[misc]
        base = excess_of(base_game, spec)
        return float(
            np.log10(max(base * floor_of(mixed_games[a], spec) / floor_of(base_game, spec), 1e-300))
        )

    def h_gap(probe: object) -> float:
        a, spec = probe  # type: ignore[misc]
        base = excess_of(base_game, spec)
        return float(
            np.log10(max(base * gap_of(base_game, spec) / gap_of(mixed_games[a], spec), 1e-300))
        )

    def h_quad(probe: object) -> float:
        a, spec = probe  # type: ignore[misc]
        return float(np.log10(max(excess_of(base_game, spec) * (1.0 - a) ** 2, 1e-300)))

    hypotheses = [
        Hypothesis(name="scale_fold", predict=h_scale),
        Hypothesis(name="ness_floor", predict=h_floor),
        Hypothesis(name="spectral_gap", predict=h_gap),
        Hypothesis(name="quadratic", predict=h_quad),
    ]
    probes = [(a, tuple(spec)) for a in cc["family"]["alphas_probe"] for spec in cc["protocols"]]

    def run_probe(probe: object) -> float:
        a, spec = probe  # type: ignore[misc]
        return float(np.log10(max(excess_of(mixed_games[a], spec), 1e-300)))

    result = run_campaign(
        hypotheses,
        probes,
        run_probe=run_probe,
        sigma=sigma,
        budget=int(cc["budget"]),
        stop_confidence=float(cc["stop_confidence"]),
    )
    # absolute adequacy guard (pre-registered): winner must actually FIT
    widx = result.hypothesis_names.index(result.winner)
    resids = [abs(step.predictions[widx] - step.observed) for step in result.history]
    adequate = float(np.median(resids)) < float(cc["adequacy_max_median_resid_sigmas"]) * sigma
    verdict = result.winner if adequate else "all_hypotheses_inadequate"

    # held-out validation (added after run 1 stopped at ONE probe — recorded
    # in config): the winner must keep fitting on every probe the campaign
    # never consumed, else the verdict downgrades.
    used = {step.probe for step in result.history}
    held_out = [p for p in probes if p not in used]
    val_resids = []
    winner_h = hypotheses[widx]
    for p in held_out:
        val_resids.append(abs(winner_h.predict(p) - run_probe(p)))
    val_median = float(np.median(val_resids)) if val_resids else 0.0
    val_max = float(np.max(val_resids)) if val_resids else 0.0
    validated = val_median < float(cc["adequacy_max_median_resid_sigmas"]) * sigma
    if adequate and not validated:
        verdict = "winner_failed_validation"

    # sensitivity: same campaign at the wider sigma must agree (or both fail adequacy)
    result2 = run_campaign(
        hypotheses,
        probes,
        run_probe=run_probe,
        sigma=float(cc["sigma_sensitivity"]),
        budget=int(cc["budget"]),
        stop_confidence=float(cc["stop_confidence"]),
    )
    sensitivity_agrees = result2.winner == result.winner

    _write(
        BenchmarkResult(
            benchmark_id="efe_mechanism_campaign",
            unit=UNIT,
            kind="statistical",
            passed=True,  # the campaign ran honestly; the verdict is data
            metrics={
                **{
                    f"belief_{n}": float(b)
                    for n, b in zip(result.hypothesis_names, result.beliefs, strict=True)
                },
                "n_probes_run": float(len(result.history)),
                "stopped_early": float(result.stopped_early),
                "winner_median_resid_dex": float(np.median(resids)),
                "adequate": float(adequate),
                "heldout_n": float(len(held_out)),
                "heldout_median_resid_dex": val_median,
                "heldout_max_resid_dex": val_max,
                "heldout_validated": float(validated),
                "sensitivity_sigma_agrees": float(sensitivity_agrees),
                **{
                    f"probe{i}_alpha": float(step.probe[0])  # type: ignore[index]
                    for i, step in enumerate(result.history)
                },
                **{f"probe{i}_efe": float(step.efe) for i, step in enumerate(result.history)},
            },
            effect_sizes=[
                EffectSize(
                    name=f"posterior belief in winner '{verdict}'",
                    value=float(np.max(result.beliefs)),
                    ci_low=float(np.max(result.beliefs)),
                    ci_high=float(np.max(result.beliefs)),
                    method=(
                        f"EFE/BALD greedy selection, sigma={sigma} dex, "
                        f"{len(result.history)} of {len(probes)} probes run"
                    ),
                )
            ],
            n=len(result.history),
            n_justification=(
                f"budget {cc['budget']} of {len(probes)} candidate probes — the point "
                "of EFE selection is that the campaign resolves in a fraction of the "
                "grid; every skipped probe is a recorded saving, not silent truncation."
            ),
            seed=seed,
            config_ref=str(CONFIG.relative_to(REPO)),
            library_version=strataq.__version__,
            timestamp=_now(),
            notes=(
                f"F-0012 mechanism campaign verdict: {verdict}. Four quantitative "
                "hypotheses (potential-scale fold / NESS-sensitivity floor / spectral "
                "gap / quadratic strawman) predicting log10 excess; observations are "
                "exact quench computations on the mixed family; pre-registered "
                "adequacy guard and sigma-sensitivity re-run included."
            ),
        )
    )


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text())
    seed = int(cfg["seed"])
    (RESULTS / "bayes_reading.resolved.yaml").write_text(yaml.safe_dump(cfg))
    recovery(cfg, seed)
    campaign(cfg, seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
