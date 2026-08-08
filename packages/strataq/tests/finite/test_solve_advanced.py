"""Mirror descent, implicit diff, and pygambit agreement (PROGRAMME §8.6 tests 5/6)."""

import jax
import jax.numpy as jnp
import pytest
from strataq.core.defaults import base_config
from strataq.core.solve.fixedpoint import logit_qre
from strataq.core.solve.implicit import qre_sigma
from strataq.core.solve.mirror import logit_qre_mirror, solve
from strataq.finite.games.library import congestion, matching_pennies, rock_paper_scissors
from strataq.finite.games.tensor import DenseTensorGame
from strataq.finite.response.susceptibility import chi_equilibrium

TOL = base_config().tolerances
COSTS = jnp.array([[1.0, 2.5, 4.0], [1.5, 2.0, 3.5], [0.5, 3.0, 5.0]])


def _random_game(seed, shape=(3, 3)):
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    return DenseTensorGame((jax.random.normal(k1, shape), jax.random.normal(k2, shape)))


class TestMirrorDescent:
    def test_agrees_with_damped_on_random_games(self):
        for seed in range(5):
            game = _random_game(seed)
            a = logit_qre(game, 1.1, tol=1e-13, max_iter=200_000)
            b = logit_qre_mirror(game, 1.1, tol=1e-13, max_iter=200_000)
            gap = max(float(jnp.max(jnp.abs(x - y))) for x, y in zip(a.sigma, b.sigma, strict=True))
            assert gap < 1e-9, f"seed {seed}: gap {gap}"

    def test_last_iterate_on_harmonic_game(self):
        """Where undamped best-response would cycle, MMD's last iterate lands."""
        point = logit_qre_mirror(rock_paper_scissors(), 4.0, tol=1e-13, max_iter=200_000)
        assert bool(point.converged)
        for s in point.sigma:
            assert jnp.max(jnp.abs(s - 1.0 / 3.0)) < 1e-10

    def test_strategy_dispatch(self):
        game = matching_pennies()
        a = solve(game, 1.0, method="damped")
        b = solve(game, 1.0, method="mirror")
        assert (
            max(float(jnp.max(jnp.abs(x - y))) for x, y in zip(a.sigma, b.sigma, strict=True))
            < 1e-8
        )
        with pytest.raises(ValueError):
            solve(game, 1.0, method="nope")


class TestImplicitDiff:
    def test_jacobian_matches_chi_equilibrium(self):
        """The custom VJP reconstructs exactly the resolvent (Result 1)."""
        game = _random_game(42)
        lam = 0.9
        total = sum(game.num_actions)
        jac = jax.jacrev(lambda h: qre_sigma(game, h, lam, 1e-13, 200_000))(jnp.zeros(total))
        point = logit_qre(game, lam, tol=1e-13, max_iter=200_000)
        chi = chi_equilibrium(game, point).chi_full
        assert jnp.max(jnp.abs(jac - chi)) < 1e-9

    def test_scalar_objective_gradient_matches_fd(self):
        game = congestion(2, COSTS)
        lam = 1.2
        total = sum(game.num_actions)
        weights = jnp.linspace(0.5, 1.5, total)

        def objective(h):
            return weights @ qre_sigma(game, h, lam, 1e-13, 200_000)

        grad = jax.grad(objective)(jnp.zeros(total))
        eps = 1e-6
        for col in (0, 3, 5):
            e = jnp.zeros(total).at[col].set(eps)
            fd = (objective(e) - objective(-e)) / (2 * eps)
            assert abs(float(grad[col]) - float(fd)) < TOL.fd


gambit = pytest.importorskip("pygambit", reason="gambit extra not installed")
from strataq.core.solve.validate import gambit_qre_sigma, max_profile_gap  # noqa: E402


class TestGambitAgreement:
    def test_fixed_lambda_agreement_small_games(self):
        """Test 6: our fixed-λ QRE matches Gambit's homotopy to 1e-8."""
        games = [
            matching_pennies(),
            rock_paper_scissors(),
            _random_game(7, (2, 2)),
            _random_game(8, (3, 3)),
        ]
        for lam in (0.5, 1.5):
            for game in games:
                ours = logit_qre(game, lam, tol=1e-14, max_iter=400_000)
                theirs = gambit_qre_sigma(game, lam)
                assert max_profile_gap(ours.sigma, theirs) < 1e-8
