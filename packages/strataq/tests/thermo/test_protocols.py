"""Hatano–Sasa split + λ-quench protocols (unit thermo.protocols).

The split identities are exact statements about the Glauber chain; the
quench integral fluctuation theorems are verified two ways — by exact
weighted-vector propagation (machine precision) and from sampled
trajectories (finite-sample, CI) — because the data-side estimator is
what real systems will get.
"""

import jax.numpy as jnp
from strataq.core.defaults import base_config
from strataq.core.dynamics.markov import glauber_generator, stationary_distribution
from strataq.finite.games.library import (
    congestion,
    congestion_potential,
    coordination,
    matching_pennies,
    rock_paper_scissors,
)
from strataq.thermo.exact import gibbs_distribution
from strataq.thermo.protocols import (
    QuenchProtocol,
    epr_split,
    hatano_sasa_exact,
    hatano_sasa_sampled,
    jarzynski_exact,
    relax,
)

TOL = base_config().tolerances
COSTS = jnp.array([[1.0, 2.5, 4.0], [1.5, 2.0, 3.5], [0.5, 3.0, 5.0]])


def _skewed(n: int) -> jnp.ndarray:
    p = jnp.arange(1.0, n + 1.0)
    return p / jnp.sum(p)


class TestEprSplit:
    def test_sum_identity_off_stationarity(self):
        """total = housekeeping + excess for an arbitrary distribution."""
        gen = glauber_generator(rock_paper_scissors(), 1.5)
        pi = stationary_distribution(gen)
        p = _skewed(pi.shape[0])
        total, hk, ex = epr_split(gen, pi, p)
        assert abs(float(total - hk - ex)) < 1e-10
        assert float(hk) >= -1e-12 and float(ex) >= -1e-12  # both parts nonneg

    def test_potential_game_has_zero_housekeeping_at_any_p(self):
        """Detailed balance ⟹ adiabatic part vanishes identically."""
        gen = glauber_generator(congestion(2, COSTS), 1.2)
        pi = stationary_distribution(gen)
        for p in (pi, _skewed(pi.shape[0])):
            _, hk, _ = epr_split(gen, pi, p)
            assert abs(float(hk)) < 1e-10

    def test_harmonic_game_at_stationarity_is_pure_housekeeping(self):
        """At p = π the excess vanishes and housekeeping = Schnakenberg EPR."""
        from strataq.core.dynamics.entropy import entropy_production_rate

        gen = glauber_generator(matching_pennies(), 2.0)
        pi = stationary_distribution(gen)
        total, hk, ex = epr_split(gen, pi, pi)
        assert abs(float(ex)) < 1e-10
        assert abs(float(hk - entropy_production_rate(gen, pi))) < 1e-10
        assert float(total) > 1e-3  # genuinely dissipative NESS

    def test_excess_is_minus_ddt_kl_to_stationary(self):
        """σ_ex(p_t) = −d/dt D(p_t ‖ π) along fixed-generator relaxation."""
        gen = glauber_generator(coordination(2, 3, bonus=2.0), 1.0)
        pi = stationary_distribution(gen)
        p0 = _skewed(pi.shape[0])
        dt = 1e-5
        p_t = relax(p0, gen, 0.3)
        p_next = relax(p_t, gen, dt)

        def kl(p):
            return float(jnp.sum(p * (jnp.log(p) - jnp.log(pi))))

        _, _, ex = epr_split(gen, pi, p_t)
        assert abs(float(ex) - (-(kl(p_next) - kl(p_t)) / dt)) < 1e-4


class TestRelax:
    def test_relaxation_reaches_stationary(self):
        gen = glauber_generator(matching_pennies(), 1.5)
        pi = stationary_distribution(gen)
        p = relax(_skewed(pi.shape[0]), gen, 50.0)
        assert jnp.max(jnp.abs(p - pi)) < 1e-8

    def test_probability_conserved(self):
        gen = glauber_generator(rock_paper_scissors(), 1.0)
        p = relax(_skewed(gen.shape[0]), gen, 0.7)
        assert abs(float(jnp.sum(p)) - 1.0) < 1e-10


def _mixed_game(alpha: float = 0.6):
    """A genuinely non-potential game whose NESS moves with λ.

    Symmetric matching pennies / RPS have λ-independent uniform stationary
    states (Y ≡ 0 — a vacuous IFT check); the mixed-α family game has both
    a circulating component and a λ-dependent π.
    """
    from strataq.finite.decompose.generate import make_family

    return make_family(coordination(2, 2, bonus=2.0), matching_pennies(), [alpha])[0]


class TestHatanoSasaExact:
    def test_ift_holds_potential_and_harmonic(self):
        """⟨e^{−Y}⟩ = 1 exactly — including OFF detailed balance."""
        proto = QuenchProtocol(
            lambdas=jnp.array([0.5, 1.0, 2.0, 3.0]), taus=jnp.array([0.4, 0.4, 0.4])
        )
        for game, nontrivial in (
            (congestion(2, COSTS), True),
            (_mixed_game(), True),
            (rock_paper_scissors(), False),  # uniform NESS at all λ: Y ≡ 0
        ):
            expectation, mean_y = hatano_sasa_exact(game, proto)
            assert abs(float(expectation) - 1.0) < 1e-9
            assert float(mean_y) >= -1e-12  # Jensen: ⟨Y⟩ ≥ 0
            if nontrivial:
                assert float(mean_y) > 1e-4  # the check has teeth: Y is not ≡ 0

    def test_mean_dissipation_decreases_when_slower(self):
        """At fixed steps, longer holds dissipate less (p lags π less)."""
        lams = jnp.array([0.5, 1.5, 3.0])
        game = coordination(2, 3, bonus=2.0)
        fast = hatano_sasa_exact(game, QuenchProtocol(lams, jnp.array([0.05, 0.05])))[1]
        slow = hatano_sasa_exact(game, QuenchProtocol(lams, jnp.array([5.0, 5.0])))[1]
        assert float(slow) < float(fast)

    def test_quasi_static_limit_needs_fine_steps(self):
        """Long holds alone are NOT quasi-static: ⟨Y⟩ floors at Σ_k D(π_{k−1}‖π_k).

        Refining the same λ ramp into many small switches (each fully
        relaxed) sends the dissipation toward zero — the discrete
        Hatano–Sasa version of 'slowly' means small parameter steps.
        """
        game = coordination(2, 3, bonus=2.0)

        def ramp(n_steps: int) -> QuenchProtocol:
            return QuenchProtocol(
                lambdas=jnp.linspace(0.5, 3.0, n_steps + 1),
                taus=jnp.full((n_steps,), 8.0),
            )

        coarse = float(hatano_sasa_exact(game, ramp(2))[1])
        fine = float(hatano_sasa_exact(game, ramp(16))[1])
        assert fine < coarse / 4  # measured: ~1/K decay (0.66 → 0.067)
        assert fine < 0.1


class TestJarzynskiExact:
    def test_free_energy_recovered_on_potential_game(self):
        """⟨e^{ΣΔλΦ}⟩ = Z_K/Z_0 — the game-theoretic Jarzynski equality."""
        phi = congestion_potential(2, COSTS).reshape(-1)
        game = congestion(2, COSTS)
        proto = QuenchProtocol(lambdas=jnp.array([0.3, 0.9, 1.8]), taus=jnp.array([0.5, 0.5]))
        expectation, z_ratio = jarzynski_exact(game, phi, proto)
        assert abs(float(expectation / z_ratio) - 1.0) < 1e-9

    def test_jarzynski_equals_hatano_sasa_on_potential_games(self):
        """On potential games the two IFTs are the same statement."""
        phi = congestion_potential(2, COSTS).reshape(-1)
        game = congestion(2, COSTS)
        proto = QuenchProtocol(lambdas=jnp.array([0.4, 1.2, 2.5]), taus=jnp.array([0.3, 0.3]))
        hs_exp, _ = hatano_sasa_exact(game, proto)
        jz_exp, z_ratio = jarzynski_exact(game, phi, proto)
        assert abs(float(jz_exp / z_ratio) - float(hs_exp)) < 1e-9

    def test_gibbs_consistency(self):
        """The Z-ratio matches the Gibbs distributions the split relies on."""
        phi = congestion_potential(2, COSTS).reshape(-1)
        lam0, lam1 = 0.3, 1.8
        z0 = float(jnp.sum(jnp.exp(lam0 * phi)))
        z1 = float(jnp.sum(jnp.exp(lam1 * phi)))
        _, z_ratio = jarzynski_exact(
            congestion(2, COSTS),
            phi,
            QuenchProtocol(lambdas=jnp.array([lam0, lam1]), taus=jnp.array([0.5])),
        )
        assert abs(float(z_ratio) - z1 / z0) < 1e-9
        # and the endpoint Gibbs measures normalise consistently
        g = gibbs_distribution(phi, lam1)
        assert abs(float(jnp.sum(g)) - 1.0) < 1e-12


class TestHatanoSasaSampled:
    def test_finite_sample_ift_within_ci(self):
        """The data-side estimator: sampled ⟨e^{−Y}⟩ brackets 1."""
        proto = QuenchProtocol(lambdas=jnp.array([0.5, 1.5, 3.0]), taus=jnp.array([0.5, 0.5]))
        for game in (_mixed_game(), congestion(2, COSTS)):
            est, ci_lo, ci_hi, mean_y = hatano_sasa_sampled(
                game, proto, n_trajectories=4000, steps_per_unit_time=40, seed=7
            )
            assert ci_lo <= 1.0 <= ci_hi
            assert abs(est - 1.0) < 0.2
            assert mean_y > 1e-4  # Jensen visible in the sample too
