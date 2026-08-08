"""Exact non-equilibrium layer (PROGRAMME v3 §8.6 tests 2 and 3, dynamics half)."""

import itertools

import jax.numpy as jnp
from strataq.core.defaults import base_config
from strataq.core.dynamics.markov import glauber_generator, stationary_distribution
from strataq.finite.decompose.generate import make_family
from strataq.finite.games.library import (
    congestion,
    congestion_potential,
    coordination,
    matching_pennies,
    rock_paper_scissors,
)
from strataq.thermo.exact import gibbs_distribution, thermo_read

TOL = base_config().tolerances
COSTS = jnp.array([[1.0, 2.5, 4.0], [1.5, 2.0, 3.5], [0.5, 3.0, 5.0]])


class TestGenerator:
    def test_rows_sum_to_zero_and_offdiag_nonnegative(self):
        gen = glauber_generator(rock_paper_scissors(), 1.5)
        assert jnp.max(jnp.abs(jnp.sum(gen, axis=1))) < TOL.identity
        offdiag = gen - jnp.diag(jnp.diag(gen))
        assert float(jnp.min(offdiag)) >= 0.0

    def test_stationary_is_stationary(self):
        gen = glauber_generator(matching_pennies(), 2.0)
        pi = stationary_distribution(gen)
        assert abs(float(jnp.sum(pi)) - 1.0) < TOL.identity
        assert jnp.max(jnp.abs(pi @ gen)) < 1e-11


class TestPotentialGamesAreEquilibrium:
    """K3: potential ⟹ Gibbs stationary measure, zero current, zero dissipation."""

    def test_gibbs_measure_exact(self):
        lam = 0.9
        game = congestion(2, COSTS)
        phi = congestion_potential(2, COSTS)
        reading = thermo_read(game, lam)
        gibbs = gibbs_distribution(phi, lam)
        assert jnp.max(jnp.abs(reading.pi - gibbs)) < 1e-10

    def test_zero_current_zero_epr(self):
        for game in (congestion(2, COSTS), coordination(2, 3, bonus=2.0)):
            reading = thermo_read(game, 1.2)
            assert float(reading.max_current) < 1e-12
            assert float(reading.epr) < 1e-12
            assert bool(reading.detailed_balance)

    def test_three_player_congestion_gibbs(self):
        lam = 0.7
        game = congestion(3, COSTS)
        phi = congestion_potential(3, COSTS)
        reading = thermo_read(game, lam)
        gibbs = gibbs_distribution(phi, lam)
        assert jnp.max(jnp.abs(reading.pi - gibbs)) < 1e-10
        assert float(reading.epr) < 1e-12


class TestHarmonicGamesAreNESS:
    def test_rps_carries_current_and_dissipates(self):
        reading = thermo_read(rock_paper_scissors(), 2.0)
        assert float(reading.max_current) > 1e-3
        assert float(reading.epr) > 1e-3
        assert not bool(reading.detailed_balance)

    def test_matching_pennies_ness(self):
        reading = thermo_read(matching_pennies(), 1.5)
        assert float(reading.epr) > 1e-3


class TestChainAlongAlpha:
    def test_epr_rises_with_alpha_on_a_family(self):
        """First C1 data: dissipation co-moves with harmonic fraction."""
        family = make_family(
            congestion(2, COSTS), rock_paper_scissors(), [0.0, 0.25, 0.5, 0.75, 1.0], scale=2.0
        )
        eprs = [float(thermo_read(g, 1.2).epr) for g in family]
        assert eprs[0] < 1e-12
        assert all(b > a - 1e-12 for a, b in itertools.pairwise(eprs))
        assert eprs[-1] > 1e-3
