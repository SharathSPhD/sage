"""λ-estimator family (PROGRAMME v3 §4): four routes to the same parameter.

Ground truth by construction: draw synthetic choice data from σ*(λ*) at known
λ*, then require every estimator to recover it — and require the agreement
protocol to notice when the model is misspecified (mixture data).
"""

import jax
import jax.numpy as jnp
from strataq.core.defaults import base_config
from strataq.core.solve.fixedpoint import logit_qre
from strataq.estimate.lam import (
    LambdaAgreement,
    agreement_protocol,
    lambda_dispersion,
    lambda_mle,
    lambda_mle_implicit,
    lambda_moment_chi,
    sample_choices,
)
from strataq.finite.games.library import coordination, rock_paper_scissors
from strataq.finite.games.tensor import DenseTensorGame
from strataq.finite.response.susceptibility import chi_equilibrium

TOL = base_config().tolerances
SEED = 20260810
ASYM = DenseTensorGame(
    (
        jnp.array([[3.0, 0.0, 1.5], [1.0, 2.0, 0.5], [0.0, 1.0, 2.5]]),
        jnp.array([[2.0, 1.0, 0.0], [0.5, 3.0, 1.0], [1.5, 0.0, 2.0]]),
    )
)


def _counts(game, lam_star, n, seed=SEED):
    return sample_choices(game, lam_star, n, jax.random.key(seed))


class TestSampling:
    def test_counts_shape_and_total(self):
        counts = _counts(ASYM, 1.2, 4000)
        assert len(counts) == 2
        assert all(int(jnp.sum(c)) == 4000 for c in counts)

    def test_frequencies_approach_sigma(self):
        lam = 1.2
        counts = _counts(ASYM, lam, 200_000)
        sigma = logit_qre(ASYM, lam).sigma
        for c, s in zip(counts, sigma, strict=True):
            assert float(jnp.max(jnp.abs(c / jnp.sum(c) - s))) < 0.01


class TestRecovery:
    """Each estimator recovers λ* from well-specified synthetic data."""

    def test_mle_grid(self):
        for lam_star in (0.5, 1.2, 3.0):
            counts = _counts(ASYM, lam_star, 20_000)
            est = lambda_mle(ASYM, counts)
            assert abs(est.lam - lam_star) / lam_star < 0.10

    def test_mle_implicit_agrees_with_grid(self):
        counts = _counts(ASYM, 1.2, 20_000)
        grid = lambda_mle(ASYM, counts)
        impl = lambda_mle_implicit(ASYM, counts)
        assert abs(impl.lam - grid.lam) / grid.lam < 0.02

    def test_moment_chi(self):
        lam_star = 1.2
        point = logit_qre(ASYM, lam_star)
        chi_obs = chi_equilibrium(ASYM, point).chi_full  # exact χ as 'observed'
        est = lambda_moment_chi(ASYM, chi_obs)
        assert abs(est.lam - lam_star) / lam_star < 0.02

    def test_dispersion(self):
        for lam_star in (0.5, 1.2, 3.0):
            counts = _counts(ASYM, lam_star, 20_000, seed=SEED + 1)
            est = lambda_dispersion(ASYM, counts)
            assert abs(est.lam - lam_star) / lam_star < 0.15

    def test_dispersion_flags_flat_entropy(self):
        """Below the pitchfork, coordination's principal-branch QRE is uniform
        at every λ — entropy inversion is unidentified and must say so."""
        game = coordination(2, 3, bonus=2.0)
        counts = _counts(game, 0.5, 20_000, seed=SEED + 4)
        est = lambda_dispersion(game, counts)
        assert any("unidentified" in w or "flat" in w for w in est.warnings)

    def test_estimates_carry_uncertainty(self):
        counts = _counts(ASYM, 1.2, 20_000)
        est = lambda_mle(ASYM, counts)
        assert est.ci_low < est.lam < est.ci_high


class TestAgreementProtocol:
    def test_well_specified_data_agrees(self):
        counts = _counts(ASYM, 1.2, 40_000)
        report: LambdaAgreement = agreement_protocol(ASYM, counts)
        assert report.agreement_gap < 0.15
        assert not report.disagreement_flag
        assert set(report.estimates) >= {"mle", "mle_implicit", "dispersion"}

    def test_misspecified_mixture_flags(self):
        """Data from a λ-mixture (half λ=0.4, half λ=4.0) is NOT a QRE at any
        single λ; the estimators should disagree more than on clean data."""
        k1, k2 = jax.random.split(jax.random.key(SEED + 2))
        a = sample_choices(ASYM, 0.4, 20_000, k1)
        b = sample_choices(ASYM, 4.0, 20_000, k2)
        mixed = tuple(x + y for x, y in zip(a, b, strict=True))
        clean = agreement_protocol(ASYM, _counts(ASYM, 1.2, 40_000))
        messy = agreement_protocol(ASYM, mixed)
        assert messy.agreement_gap > clean.agreement_gap

    def test_rps_symmetric_data_is_uninformative_and_says_so(self):
        """On RPS every λ gives the uniform mix — λ is unidentified from
        frequencies alone. The protocol must WARN, not hallucinate a number."""
        counts = _counts(rock_paper_scissors(), 1.5, 40_000, seed=SEED + 3)
        report = agreement_protocol(rock_paper_scissors(), counts)
        assert any("unidentified" in w or "flat" in w for w in report.warnings)

    def test_dispersion_no_bootstrap_point_matches(self):
        counts = _counts(ASYM, 1.2, 20_000, seed=SEED + 1)
        full = lambda_dispersion(ASYM, counts)
        fast = lambda_dispersion(ASYM, counts, bootstrap=False)
        assert fast.lam == full.lam
        assert fast.ci_low == fast.lam == fast.ci_high
