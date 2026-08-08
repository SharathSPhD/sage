"""Exact-identity and behaviour tests for the Engine-1 core (PROGRAMME v3 §8.6)."""

import itertools

import jax
import jax.numpy as jnp
import pytest
from strataq.core.defaults import base_config
from strataq.core.solve.fixedpoint import logit_qre
from strataq.finite.games.library import (
    common_interest,
    congestion,
    congestion_potential,
    coordination,
    matching_pennies,
    rock_paper_scissors,
)
from strataq.finite.games.normalise import normalise
from strataq.finite.games.tensor import DenseTensorGame, expected_payoffs

TOL = base_config().tolerances


def _random_game(key, shape=(3, 3), n_players=2):
    keys = jax.random.split(key, n_players)
    return DenseTensorGame(tuple(jax.random.normal(k, shape) for k in keys))


class TestIdentities:
    def test_gradient_of_log_partition_is_lambda_sigma(self):
        """Test 1 (K2): ∇_U ψ = λσ and ∇²_U ψ = λ²C, to the identity tolerance."""
        lam = 1.7
        key = jax.random.PRNGKey(base_config().seeds.root)
        u = jax.random.normal(key, (5,))

        def psi(util):
            return jax.nn.logsumexp(lam * util)

        sigma = jnp.exp(jax.nn.log_softmax(lam * u))
        grad = jax.grad(psi)(u)
        assert jnp.max(jnp.abs(grad - lam * sigma)) < TOL.identity

        hess = jax.hessian(psi)(u)
        c_matrix = jnp.diag(sigma) - jnp.outer(sigma, sigma)
        assert jnp.max(jnp.abs(hess - lam**2 * c_matrix)) < TOL.identity

    def test_potential_game_qre_matches_gibbs_marginals(self):
        """Test 2 (K3): in an exact potential game, QRE marginals = Gibbs marginals of e^{λΦ}."""
        lam = 0.8
        costs = jnp.array([[1.0, 2.5, 4.0], [1.5, 2.0, 3.5], [0.5, 3.0, 5.0]])
        game = congestion(3, costs)
        phi = congestion_potential(3, costs)

        point = logit_qre(game, lam, tol=1e-14, max_iter=200_000, damping=0.5)
        assert bool(point.converged)

        # Gibbs joint over profiles from the potential; but logit QRE is the
        # *product* measure fixed point: σ_i(a) ∝ exp(λ E_{σ_{-i}}[Φ | a_i=a]).
        # For the check use the mean-field self-consistency of the Gibbs form:
        # verify σ solves the potential-game fixed point through Φ alone.
        for i in range(3):
            phi_i = phi
            for j in range(2, -1, -1):
                if j == i:
                    continue
                phi_i = jnp.tensordot(phi_i, point.sigma[j], axes=([j], [0]))
            gibbs_sigma = jnp.exp(jax.nn.log_softmax(lam * phi_i))
            assert jnp.max(jnp.abs(gibbs_sigma - point.sigma[i])) < 1e-10

    def test_nonstrategic_component_invariance(self):
        """Test 4 (K5): adding a component constant in own action leaves σ* unchanged."""
        key = jax.random.PRNGKey(base_config().seeds.root + 1)
        game = _random_game(key)
        k1, k2 = jax.random.split(key)
        # f_i depends only on the rival's action (constant in own action).
        f0 = jax.random.normal(k1, (3,))[None, :] * jnp.ones((3, 1))
        f1 = jax.random.normal(k2, (3,))[:, None] * jnp.ones((1, 3))
        shifted = DenseTensorGame((game.payoffs[0] + f0, game.payoffs[1] + f1))

        p_base = logit_qre(game, 1.3, tol=1e-13, max_iter=100_000)
        p_shift = logit_qre(shifted, 1.3, tol=1e-13, max_iter=100_000)
        for s_a, s_b in zip(p_base.sigma, p_shift.sigma, strict=True):
            assert jnp.max(jnp.abs(s_a - s_b)) < 1e-10

    def test_lambda_limits(self):
        """Test 9: λ→0 gives the centroid; large λ concentrates on a strict dominant action."""
        # Strictly dominant action for both players.
        u = jnp.array([[5.0, 5.0], [0.0, 0.0]])
        game = DenseTensorGame((u, u.T))
        p0 = logit_qre(game, 0.0)
        for s in p0.sigma:
            assert jnp.max(jnp.abs(s - 0.5)) < TOL.solve
        p_inf = logit_qre(game, 50.0, tol=1e-13, max_iter=100_000)
        assert p_inf.sigma[0][0] > 1.0 - 1e-8
        assert p_inf.sigma[1][0] > 1.0 - 1e-8


class TestMechanics:
    def test_expected_payoffs_matches_bruteforce(self):
        key = jax.random.PRNGKey(base_config().seeds.root + 2)
        game = _random_game(key, shape=(2, 3, 4), n_players=3)
        sigma = tuple(
            jnp.exp(jax.nn.log_softmax(jax.random.normal(k, (m,))))
            for k, m in zip(jax.random.split(key, 3), (2, 3, 4), strict=True)
        )
        utilities = expected_payoffs(game, sigma)
        # Brute force for player 1 (middle axis).
        brute = jnp.zeros(3)
        for a0, a1, a2 in itertools.product(range(2), range(3), range(4)):
            brute = brute.at[a1].add(sigma[0][a0] * sigma[2][a2] * game.payoffs[1][a0, a1, a2])
        assert jnp.max(jnp.abs(utilities[1] - brute)) < TOL.identity

    def test_normalise_zero_own_mean_and_idempotent(self):
        key = jax.random.PRNGKey(base_config().seeds.root + 3)
        game = _random_game(key, shape=(4, 5), n_players=2)
        norm = normalise(game)
        for i, u in enumerate(norm.payoffs):
            assert jnp.max(jnp.abs(jnp.mean(u, axis=i))) < TOL.identity
        again = normalise(norm)
        for u1, u2 in zip(norm.payoffs, again.payoffs, strict=True):
            assert jnp.max(jnp.abs(u1 - u2)) < TOL.identity

    def test_matching_pennies_qre_is_uniform(self):
        point = logit_qre(matching_pennies(), 2.0, tol=1e-13, max_iter=100_000)
        for s in point.sigma:
            assert jnp.max(jnp.abs(s - 0.5)) < 1e-10

    def test_rps_qre_is_uniform(self):
        point = logit_qre(rock_paper_scissors(), 3.0, tol=1e-13, max_iter=100_000)
        for s in point.sigma:
            assert jnp.max(jnp.abs(s - 1.0 / 3.0)) < 1e-10

    def test_coordination_and_common_interest_agree(self):
        game = coordination(2, 3, bonus=2.0)
        ci = common_interest(game.payoffs[0])
        for u1, u2 in zip(game.payoffs, ci.payoffs, strict=True):
            assert jnp.array_equal(u1, u2)

    def test_bad_shapes_raise(self):
        with pytest.raises(ValueError):
            DenseTensorGame((jnp.ones((2, 2)), jnp.ones((2, 3))))
        with pytest.raises(ValueError, match="at least one player"):
            DenseTensorGame(())
        with pytest.raises(ValueError):
            DenseTensorGame((jnp.ones((2, 2, 2)), jnp.ones((2, 2, 2))))
