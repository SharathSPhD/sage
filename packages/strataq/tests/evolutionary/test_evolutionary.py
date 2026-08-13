"""Evolutionary dynamics against analytic ground truth, and the beta = lambda bridge."""

import jax
import jax.numpy as jnp
import pytest
from strataq.core.defaults import base_config
from strataq.core.solve.fixedpoint import logit_qre
from strataq.evolutionary.moran import (
    compare_intensity,
    constant_selection_fixation,
    fermi,
    fixation_probability,
    moran_chain,
    pairwise_comparison_ratios,
    payoff_difference,
    small_mutation_stationary,
)
from strataq.evolutionary.replicator import (
    discrete_replicator,
    logit_dynamic_field,
    logit_rest_point,
    replicator_field,
    replicator_flow,
    rest_points,
    stability,
)
from strataq.finite.games.tensor import DenseTensorGame
from strataq.problems.evolutionary import EvolutionaryProblem

TOL = base_config().tolerances

RPS = jnp.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
COORDINATION = jnp.array([[2.0, 0.0], [0.0, 1.0]])
# Stag hunt where hare (type 1) is risk dominant: 3+0 < 2+2.
STAG_HUNT = jnp.array([[3.0, 0.0], [2.0, 2.0]])


class TestReplicatorRestPoints:
    def test_rock_paper_scissors_has_three_vertices_and_a_centre(self):
        points = rest_points(RPS)
        assert len(points) == 4
        interior = [p for p in points if int(jnp.sum(p.support)) == 3]
        assert len(interior) == 1
        assert jnp.allclose(interior[0].x, 1.0 / 3.0, atol=1e-12)

    def test_the_rps_centre_is_a_centre_and_the_vertices_are_saddles(self):
        """Every RPS vertex is invaded by exactly one rival, so it is a saddle."""
        points = {int(jnp.sum(p.support)): p for p in rest_points(RPS)}
        assert points[3].kind == "centre"
        assert jnp.allclose(jnp.real(points[3].eigenvalues), 0.0, atol=1e-10)
        assert jnp.max(jnp.abs(jnp.imag(points[3].eigenvalues))) > 0.1
        assert points[1].kind == "saddle"
        # At e_0 the eigenvalues are A[1,0] - A[0,0] = 1 and A[2,0] - A[0,0] = -1.
        vertex = next(p for p in rest_points(RPS) if float(p.x[0]) > 0.5)
        real = sorted(float(v) for v in jnp.real(vertex.eigenvalues))
        assert real == pytest.approx([-1.0, 1.0], abs=1e-10)

    def test_the_rps_centre_is_the_unique_nash_rest_point(self):
        points = rest_points(RPS)
        nash = [p for p in points if p.is_nash]
        assert len(nash) == 1
        assert jnp.allclose(nash[0].x, 1.0 / 3.0, atol=1e-12)

    def test_coordination_game_has_two_stable_vertices_and_an_unstable_interior(self):
        points = rest_points(COORDINATION)
        assert len(points) == 3
        interior = next(p for p in points if int(jnp.sum(p.support)) == 2)
        # (Ax)_0 = (Ax)_1 gives 2 x0 = x1, so x = (1/3, 2/3).
        assert jnp.allclose(interior.x, jnp.array([1.0 / 3.0, 2.0 / 3.0]), atol=1e-12)
        assert interior.kind == "unstable"
        vertices = [p for p in points if int(jnp.sum(p.support)) == 1]
        assert {v.kind for v in vertices} == {"stable"}

    def test_vertex_eigenvalues_are_the_invasion_fitnesses(self):
        """At e_k the tangent eigenvalues are A[i,k] - A[k,k] for i != k."""
        eigenvalues, kind = stability(COORDINATION, jnp.array([1.0, 0.0]))
        assert kind == "stable"
        assert float(jnp.real(eigenvalues[0])) == pytest.approx(
            float(COORDINATION[1, 0] - COORDINATION[0, 0])
        )

    def test_the_interior_eigenvalue_matches_the_two_by_two_formula(self):
        """For 2x2, the interior eigenvalue is x0 x1 (a - c - b + d)."""
        interior = jnp.array([1.0 / 3.0, 2.0 / 3.0])
        eigenvalues, _ = stability(COORDINATION, interior)
        expected = (1.0 / 3.0) * (2.0 / 3.0) * (2.0 - 0.0 - 0.0 + 1.0)
        assert float(jnp.real(eigenvalues[0])) == pytest.approx(expected, abs=1e-10)


class TestReplicatorFlow:
    def test_the_field_vanishes_at_a_rest_point(self):
        for point in rest_points(RPS):
            assert float(jnp.max(jnp.abs(replicator_field(RPS, point.x)))) < 1e-12

    def test_the_flow_stays_on_the_simplex(self):
        path = replicator_flow(COORDINATION, [0.4, 0.6], step=0.01, steps=500)
        assert jnp.allclose(jnp.sum(path, axis=1), 1.0, atol=1e-12)
        assert bool(jnp.all(path >= 0.0))

    def test_the_flow_leaves_the_unstable_interior_for_a_stable_vertex(self):
        """The basin boundary is the interior rest point x0 = 1/3."""
        above = replicator_flow(COORDINATION, [0.4, 0.6], step=0.01, steps=4000)
        below = replicator_flow(COORDINATION, [0.3, 0.7], step=0.01, steps=4000)
        assert float(above[-1, 0]) > 0.999
        assert float(below[-1, 1]) > 0.999

    def test_rps_orbits_conserve_the_product_of_shares(self):
        """The cyclic game has the constant of motion x0 x1 x2 along interior orbits."""
        path = replicator_flow(RPS, [0.5, 0.3, 0.2], step=0.001, steps=4000)
        invariant = jnp.prod(path, axis=1)
        drift = float(jnp.max(jnp.abs(invariant - invariant[0])) / invariant[0])
        assert drift < 1e-4

    def test_discrete_replicator_agrees_with_the_continuous_flow_for_small_steps(self):
        discrete = discrete_replicator(COORDINATION, [0.4, 0.6], steps=200, background=10.0)
        assert jnp.allclose(jnp.sum(discrete, axis=1), 1.0, atol=1e-12)
        assert float(discrete[-1, 0]) > float(discrete[0, 0])
        assert float(discrete[-1, 0]) > 0.99

    def test_discrete_replicator_refuses_negative_fitness(self):
        with pytest.raises(ValueError, match="non-negative fitness"):
            discrete_replicator(RPS, [0.5, 0.3, 0.2], steps=10)


class TestMoranProcess:
    def test_constant_selection_matches_the_closed_form(self):
        """rho = (1 - 1/r) / (1 - 1/r^N) when the payoff difference does not move."""
        for population in (4, 10, 50):
            for r in (0.5, 1.2, 3.0):
                ratios = jnp.full((population - 1,), 1.0 / r)
                assert float(fixation_probability(ratios)) == pytest.approx(
                    float(constant_selection_fixation(population, r)), abs=1e-12
                )

    def test_neutral_drift_fixes_with_probability_one_over_n(self):
        for population in (5, 20, 100):
            ratios = jnp.ones((population - 1,))
            assert float(fixation_probability(ratios)) == pytest.approx(1.0 / population)

    def test_a_constant_payoff_advantage_reproduces_the_closed_form_through_the_game(self):
        """A game whose payoff difference is a constant s gives r = e^{beta s} exactly."""
        advantage = 0.3
        beta = 1.5
        matrix = jnp.array([[advantage, advantage], [0.0, 0.0]])
        population = 20
        assert jnp.allclose(payoff_difference(matrix, population)[1:-1], advantage, atol=1e-12)
        ratios = pairwise_comparison_ratios(matrix, population, beta)
        expected = constant_selection_fixation(population, float(jnp.exp(beta * advantage)))
        assert float(fixation_probability(ratios)) == pytest.approx(float(expected), abs=1e-12)

    def test_risk_dominance_wins_the_small_mutation_limit(self):
        """In the stag hunt the risk-dominant hare fixes more easily than the stag."""
        ratios = pairwise_comparison_ratios(STAG_HUNT, 50, 1.0)
        rho_stag = fixation_probability(ratios)
        rho_hare = rho_stag * jnp.exp(jnp.sum(jnp.log(ratios)))
        weights = small_mutation_stationary(rho_stag, rho_hare)
        assert float(rho_hare) > float(rho_stag)
        assert float(weights[0]) > float(weights[1])  # all-hare carries more weight

    def test_the_stationary_distribution_is_normalised_and_reversible(self):
        chain = moran_chain(STAG_HUNT, 30, 1.0, mutation=1e-2)
        stationary = chain.stationary
        assert float(jnp.sum(stationary)) == pytest.approx(1.0, abs=1e-12)
        flux = stationary[:-1] * chain.up[:-1] - stationary[1:] * chain.down[1:]
        assert float(jnp.max(jnp.abs(flux))) < 1e-14

    def test_neutral_selection_gives_a_symmetric_stationary_distribution(self):
        chain = moran_chain(jnp.zeros((2, 2)), 20, 0.0, mutation=1e-2)
        stationary = chain.stationary
        assert jnp.allclose(stationary, stationary[::-1], atol=1e-12)


class TestIntensityIsPrecision:
    def test_the_fermi_rule_is_the_two_action_logit_exactly(self):
        differences = jnp.linspace(-5.0, 5.0, 41)
        for beta in (0.0, 0.7, 3.0, 12.0):
            stacked = jnp.stack([differences, jnp.zeros_like(differences)], axis=-1)
            logit = jnp.exp(jax.nn.log_softmax(beta * stacked, axis=-1))[:, 0]
            assert float(jnp.max(jnp.abs(fermi(differences, beta) - logit))) < TOL.identity

    def test_the_logit_dynamic_rest_point_is_the_symmetric_qre(self):
        """x = softmax(beta A x) is the symmetric logit QRE of (A, A^T) at lambda = beta."""
        for beta in (0.3, 1.0, 2.5):
            rest = logit_rest_point(COORDINATION, beta, tol=1e-13, max_iter=200_000)
            point = logit_qre(
                DenseTensorGame((COORDINATION, COORDINATION.T)),
                beta,
                tol=1e-13,
                max_iter=200_000,
            )
            assert float(jnp.max(jnp.abs(rest - point.sigma[0]))) < 1e-9

    def test_the_logit_dynamic_field_vanishes_at_its_rest_point(self):
        rest = logit_rest_point(RPS, 2.0, tol=1e-13, max_iter=200_000)
        assert float(jnp.max(jnp.abs(logit_dynamic_field(RPS, rest, 2.0)))) < 1e-9

    def test_compare_intensity_reports_zero_gaps(self):
        comparison = compare_intensity(COORDINATION, 1.0, 40, tol=1e-13)
        assert float(comparison.fermi_gap) < TOL.identity
        assert float(comparison.qre_gap) < 1e-9

    def test_zero_intensity_is_uniform_on_both_readings(self):
        comparison = compare_intensity(STAG_HUNT, 0.0, 30)
        assert jnp.allclose(comparison.logit_rest_point, 0.5, atol=1e-9)
        assert float(comparison.moran_share) == pytest.approx(0.5, abs=1e-9)

    def test_the_two_readings_select_the_same_convention_in_the_stag_hunt(self):
        """Both the finite population and the logit reading favour the risk-dominant hare."""
        comparison = compare_intensity(STAG_HUNT, 2.0, 60)
        assert comparison.selected == 1
        assert float(comparison.moran_share) < 0.5
        assert float(comparison.logit_rest_point[1]) > 0.5


class TestEvolutionaryProblem:
    def test_solves_rock_paper_scissors(self):
        res = EvolutionaryProblem(payoff=RPS, intensity=1.0).solve()
        assert res.rest_points.shape == (4, 3)
        assert res.stable.shape[0] == 0
        assert "centre" in res.kinds

    def test_solves_a_two_type_game_with_a_population(self):
        res = EvolutionaryProblem(payoff=STAG_HUNT, intensity=1.5, population=40).solve()
        assert res.fixation_a is not None and res.fixation_b is not None
        assert res.moran_stationary is not None
        assert len(res.moran_stationary) == 41
        assert res.qre_gap is not None and res.qre_gap < 1e-8
        assert res.fermi_gap is not None and res.fermi_gap < 1e-12

    def test_summary_and_dict(self):
        res = EvolutionaryProblem(payoff=COORDINATION, intensity=1.0, population=20).solve()
        assert "EvolutionaryProblem" in str(res.summary())
        body = res.as_dict()
        assert len(body["rest_points"]) == 3
        assert body["population"] == 20

    def test_bad_input_is_rejected(self):
        with pytest.raises(ValueError, match="square matrix"):
            EvolutionaryProblem(payoff=[[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError, match="two-type"):
            EvolutionaryProblem(payoff=RPS, population=20)
        with pytest.raises(ValueError, match="intensity must be"):
            EvolutionaryProblem(payoff=COORDINATION, intensity=-1.0)
