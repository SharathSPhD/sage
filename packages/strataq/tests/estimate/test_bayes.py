"""Bayesian layer + EFE experiment selection (unit estimate.bayes).

Three surfaces: (1) grid posterior over λ from choice counts, with the
scale-fold non-identifiability (F-0006's identity) demonstrated IN the
posterior; (2) marginal likelihood / Bayes factors, re-expressing the R1
mixture-misspecification flag as model comparison; (3) the EFE
(expected-information-gain) probe-selection loop that powers auto-research
campaigns.
"""

import jax
import jax.numpy as jnp
import numpy as np
from strataq.estimate.bayes import (
    Hypothesis,
    bayes_factor,
    efe_scores,
    grid_posterior,
    log_evidence,
    log_evidence_mixture,
    refined_posterior,
    run_campaign,
    update_beliefs,
)
from strataq.estimate.lam import sample_choices
from strataq.finite.games.tensor import DenseTensorGame

KEY = jax.random.PRNGKey(20260812)
# the R1 asymmetric anchor: informative choices at every λ (a symmetric
# coordination game's principal branch is uniform — flat likelihood — the
# exact trap the R1 guard exists for)
GAME = DenseTensorGame(
    (
        jnp.array([[3.0, 0.0, 1.5], [1.0, 2.0, 0.5], [0.0, 1.0, 2.5]]),
        jnp.array([[2.0, 1.0, 0.0], [0.5, 3.0, 1.0], [1.5, 0.0, 2.0]]),
    )
)
GRID = np.geomspace(0.1, 12.0, 120)


def _scaled(game: DenseTensorGame, s: float) -> DenseTensorGame:
    return DenseTensorGame([s * u for u in game.payoffs])


class TestGridPosterior:
    def test_recovers_true_lambda(self):
        counts = sample_choices(GAME, 1.8, 4000, KEY)
        post = grid_posterior(GAME, counts, GRID)
        assert abs(float(post.mean) - 1.8) / 1.8 < 0.15
        assert abs(float(np.sum(post.weights)) - 1.0) < 1e-10

    def test_credible_interval_coverage(self):
        """95% CI covers the truth on ≥ 8/10 seeds via refined_posterior —
        the entry point that enforces the resolution guard. (The guard's
        PR ≥ 6 bar exists because a PR ≈ 3 posterior quantises its interval
        to ~2 grid steps and measurably undercovers: 78% in the calibration
        run that caught it.)"""
        hits = 0
        for s in range(10):
            counts = sample_choices(GAME, 1.8, 2000, jax.random.PRNGKey(100 + s))
            post = refined_posterior(GAME, counts, GRID)
            assert post.grid_resolved
            lo, hi = post.credible_interval(0.95)
            hits += int(lo <= 1.8 <= hi)
        assert hits >= 8

    def test_coarse_grid_flags_itself(self):
        """A grid coarser than the data's information self-reports: the CI
        would be resolution-limited, and grid_resolved says so."""
        counts = sample_choices(GAME, 1.8, 20000, KEY)
        post = grid_posterior(GAME, counts, np.geomspace(0.1, 12.0, 60))
        assert not post.grid_resolved

    def test_interval_narrows_with_data(self):
        k1, k2 = jax.random.split(KEY)
        small = grid_posterior(GAME, sample_choices(GAME, 1.8, 200, k1), GRID)
        large = grid_posterior(GAME, sample_choices(GAME, 1.8, 20000, k2), GRID)
        w_small = np.diff(small.credible_interval(0.95))[0]
        w_large = np.diff(large.credible_interval(0.95))[0]
        assert w_large < w_small / 2

    def test_scale_fold_ridge(self):
        """(λ, s) is unidentified from choices: only sλ is — the posterior
        over λ under payoffs scaled by s is the λ/s posterior exactly."""
        counts = sample_choices(GAME, 1.8, 4000, KEY)
        post = grid_posterior(GAME, counts, GRID)
        post_scaled = grid_posterior(_scaled(GAME, 2.0), counts, GRID / 2.0)
        assert np.max(np.abs(post.weights - post_scaled.weights)) < 1e-8


class TestModelComparison:
    def test_bayes_factor_prefers_truth_on_clean_data(self):
        counts = sample_choices(GAME, 1.8, 4000, KEY)
        lz_right = log_evidence(GAME, counts, GRID)
        # a wrong game model: same shape, payoffs shuffled hard
        wrong = DenseTensorGame([jnp.flip(u, axis=0) for u in GAME.payoffs])
        lz_wrong = log_evidence(wrong, counts, GRID)
        assert bayes_factor(lz_right, lz_wrong) > 100.0  # decisive evidence ratio

    def test_mixture_model_wins_on_mixture_data_only(self):
        """R1's ×91 spread diagnostic, re-expressed as matched model
        comparison: the explicit two-λ mixture model decisively beats the
        single-λ model on mixture data, and does NOT beat it on clean data
        (Occam via marginalisation)."""
        k1, k2, k3 = jax.random.split(KEY, 3)
        n = 3000
        clean = sample_choices(GAME, 1.2, n, k1)
        lo = sample_choices(GAME, 0.4, n // 2, k2)
        hi = sample_choices(GAME, 4.0, n // 2, k3)
        mixed = tuple(a + b for a, b in zip(lo, hi, strict=True))
        bf_mixed = bayes_factor(
            log_evidence_mixture(GAME, mixed, GRID), log_evidence(GAME, mixed, GRID)
        )
        bf_clean = bayes_factor(
            log_evidence_mixture(GAME, clean, GRID), log_evidence(GAME, clean, GRID)
        )
        assert bf_mixed > 100.0
        assert bf_clean < 10.0


def _toy_hypotheses() -> list[Hypothesis]:
    return [
        Hypothesis(name="linear", predict=lambda x: float(x)),
        Hypothesis(name="quadratic", predict=lambda x: float(x) ** 2),
        Hypothesis(name="constant", predict=lambda x: 1.0),
    ]


class TestEFE:
    def test_scores_prefer_discriminating_probes(self):
        """At x=1 all three hypotheses agree (EFE ≈ 0); at x=3 they split."""
        hyps = _toy_hypotheses()
        beliefs = np.full(3, 1.0 / 3.0)
        scores = efe_scores(hyps, beliefs, probes=[1.0, 3.0], sigma=0.1)
        assert scores[1] > scores[0] + 0.5
        assert scores[0] < 0.05

    def test_update_concentrates_on_truth(self):
        hyps = _toy_hypotheses()
        beliefs = np.full(3, 1.0 / 3.0)
        for x in (2.0, 3.0):
            beliefs = update_beliefs(hyps, beliefs, probe=x, observed=float(x) ** 2, sigma=0.1)
        assert beliefs[1] > 0.95  # quadratic is the generator

    def test_campaign_selects_and_resolves(self):
        """The greedy loop finds the truth in few probes and never picks a
        non-discriminating probe while informative ones remain."""
        hyps = _toy_hypotheses()
        probes = [1.0, 1.5, 2.0, 3.0, 4.0]
        result = run_campaign(
            hyps,
            probes,
            run_probe=lambda x: float(x) ** 2,
            sigma=0.1,
            budget=4,
            stop_confidence=0.95,
        )
        assert result.winner == "quadratic"
        assert result.beliefs[1] > 0.95
        assert len(result.history) <= 4
        assert result.history[0].probe != 1.0  # the agreement point carries no information
        # audit trail is complete: probe, scores, observation, posterior per round
        for step in result.history:
            assert step.efe > -1e-12 and abs(float(np.sum(step.beliefs)) - 1.0) < 1e-10
