"""Hodge/α contract (PROGRAMME v3 §8.6 test 8, plus the §1.1 equivalence rule)."""

import itertools

import jax
import jax.numpy as jnp
from strataq.core.defaults import base_config
from strataq.core.solve.fixedpoint import logit_qre
from strataq.finite.decompose.generate import make_family
from strataq.finite.decompose.hodge import alpha, exact_potential_of, hodge_decompose
from strataq.finite.decompose.kron import subset_component, subset_decompose
from strataq.finite.games.library import (
    congestion,
    coordination,
    matching_pennies,
    rock_paper_scissors,
)
from strataq.finite.games.tensor import DenseTensorGame
from strataq.finite.response.reciprocity import reciprocity_defect

TOL = base_config().tolerances
COSTS = jnp.array([[1.0, 2.5, 4.0], [1.5, 2.0, 3.5], [0.5, 3.0, 5.0]])


def _random_game(seed, shape, n_players):
    keys = jax.random.split(jax.random.PRNGKey(seed), n_players)
    return DenseTensorGame(tuple(jax.random.normal(k, shape) for k in keys))


class TestSubsetTransform:
    def test_reconstruction_and_orthogonality(self):
        key = jax.random.PRNGKey(base_config().seeds.root + 20)
        tensor = jax.random.normal(key, (3, 4, 2))
        parts = subset_decompose(tensor)
        recon = sum(parts.values())
        assert jnp.max(jnp.abs(recon - tensor)) < TOL.decompose
        flat = {t: p.ravel() for t, p in parts.items()}
        keys_list = list(flat)
        for a in range(len(keys_list)):
            for b in range(a + 1, len(keys_list)):
                assert abs(float(flat[keys_list[a]] @ flat[keys_list[b]])) < 1e-9

    def test_idempotence(self):
        key = jax.random.PRNGKey(base_config().seeds.root + 21)
        tensor = jax.random.normal(key, (3, 3))
        comp = subset_component(tensor, frozenset({0}))
        again = subset_component(comp, frozenset({0}))
        assert jnp.max(jnp.abs(comp - again)) < TOL.decompose


class TestHodge:
    def test_reconstruction(self):
        game = _random_game(1, (3, 4), 2)
        dec = hodge_decompose(game)
        for u, p, h, ns in zip(
            game.payoffs,
            dec.potential.payoffs,
            dec.harmonic.payoffs,
            dec.nonstrategic.payoffs,
            strict=True,
        ):
            assert jnp.max(jnp.abs(u - (p + h + ns))) < TOL.decompose

    def test_weighted_orthogonality(self):
        """P ⊥ H under the m-weighted (Candogan response-graph) inner product."""
        game = _random_game(2, (3, 5), 2)
        dec = hodge_decompose(game)
        m = game.num_actions
        ip = sum(
            m[i] * float(dec.potential.payoffs[i].ravel() @ dec.harmonic.payoffs[i].ravel())
            for i in range(2)
        )
        norms = sum(
            m[i] * float(jnp.sum(dec.potential.payoffs[i] ** 2 + dec.harmonic.payoffs[i] ** 2))
            for i in range(2)
        )
        assert abs(ip) / max(norms, 1e-30) < 1e-10

    def test_alpha_zero_on_exact_potential_games(self):
        for game in (congestion(2, COSTS), congestion(3, COSTS), coordination(3, 2, bonus=1.5)):
            assert alpha(game) < 1e-10

    def test_alpha_one_on_pure_harmonic_games(self):
        for game in (rock_paper_scissors(), rock_paper_scissors(5), matching_pennies()):
            assert alpha(game) > 1.0 - 1e-10

    def test_idempotence_of_components(self):
        game = _random_game(3, (4, 4), 2)
        dec = hodge_decompose(game)
        pot_again = hodge_decompose(dec.potential)
        harm_again = hodge_decompose(dec.harmonic)
        assert float(pot_again.harmonic_norm) < 1e-10 * max(float(dec.potential_norm), 1e-30)
        assert float(harm_again.potential_norm) < 1e-10 * max(float(dec.harmonic_norm), 1e-30)

    def test_strategic_equivalence_gives_identical_alpha(self):
        """§1.1 rule: adding own-action-constant terms must not move α."""
        game = _random_game(4, (3, 3), 2)
        key = jax.random.PRNGKey(base_config().seeds.root + 22)
        k1, k2 = jax.random.split(key)
        shift0 = jax.random.normal(k1, (3,))[None, :] * jnp.ones((3, 1))
        shift1 = jax.random.normal(k2, (3,))[:, None] * jnp.ones((1, 3))
        shifted = DenseTensorGame(
            (game.payoffs[0] + 10.0 * shift0, game.payoffs[1] + 10.0 * shift1)
        )
        assert abs(alpha(game) - alpha(shifted)) < 1e-10

    def test_recovered_potential_is_a_potential_for_the_component(self):
        """Φ from the decomposition satisfies u_i^P(a_i, a_-i) − u_i^P(b_i, a_-i)
        = Φ(a_i, a_-i) − Φ(b_i, a_-i) for every player and deviation."""
        game = _random_game(5, (3, 3, 2), 3)
        dec = hodge_decompose(game)
        phi = exact_potential_of(dec)
        for i in range(3):
            u = dec.potential.payoffs[i]
            du = u - jnp.take(u, jnp.array([0]), axis=i)
            dphi = phi - jnp.take(phi, jnp.array([0]), axis=i)
            assert jnp.max(jnp.abs(du - dphi)) < 1e-9


class TestFamilies:
    def test_family_hits_target_alpha_exactly(self):
        pot = congestion(2, COSTS)
        harm = rock_paper_scissors()
        targets = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        family = make_family(pot, harm, targets)
        for target, game in zip(targets, family, strict=True):
            assert abs(alpha(game) - target) < 1e-9

    def test_reciprocity_increases_along_family(self):
        """The milestone chain: ℛ(α) rises monotonically on a clean family."""
        pot = congestion(2, COSTS)
        harm = rock_paper_scissors()
        targets = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        family = make_family(pot, harm, targets, scale=2.0)
        values = []
        for game in family:
            point = logit_qre(game, 1.2, tol=1e-13, max_iter=200_000)
            values.append(float(reciprocity_defect(game, point)))
        assert values[0] < 1e-10
        assert all(b > a - 1e-12 for a, b in itertools.pairwise(values))
        assert values[-1] > 0.1
