"""Engine 2 exact identities: Beckmann gradient, Fisk KKT, DF symmetry, toll response."""

import jax
import jax.numpy as jnp
from strataq.core.defaults import base_config
from strataq.population.games.routing import (
    RoutingNetwork,
    payoff_field_jacobian,
    solve_sue,
)
from strataq.population.response.susceptibility import (
    df_symmetry_defect,
    population_reciprocity_defect,
    population_tangent_basis,
    toll_susceptibility,
    toll_susceptibility_fd,
)

TOL = base_config().tolerances


def parallel_network() -> RoutingNetwork:
    """Two parallel routes, one link each, linear costs (BPR p=1). Analytic UE."""
    return RoutingNetwork(
        incidence=jnp.eye(2),
        od_index=jnp.zeros(2, dtype=jnp.int32),
        demand=jnp.array([3.0]),
        free_flow=jnp.array([1.0, 2.0]),
        b_coeff=jnp.array([1.0, 0.25]),  # slopes: 1·v, 2·0.25/1·v = 0.5·v
        capacity=jnp.array([1.0, 1.0]),
        power=jnp.array([1.0, 1.0]),
    )


def braess_network() -> RoutingNetwork:
    """Classic Braess diamond with the crossing link: 3 routes over 5 links.

    Links: 0 = A→B (steep), 1 = B→D (flat), 2 = A→C (flat), 3 = C→D (steep),
    4 = B→C (crossing, cheap). Routes: up (0,1), down (2,3), cross (0,4,3).
    Shared links couple the routes — off-diagonal DF terms, still symmetric.
    """
    incidence = jnp.array(
        [
            [1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [1, 0, 0, 1, 1],
        ],
        dtype=jnp.float64,
    )
    return RoutingNetwork(
        incidence=incidence,
        od_index=jnp.zeros(3, dtype=jnp.int32),
        demand=jnp.array([6.0]),
        free_flow=jnp.array([1.0, 45.0, 45.0, 1.0, 1.0]),
        b_coeff=jnp.array([10.0, 0.0, 0.0, 10.0, 0.0]),
        capacity=jnp.ones(5),
        power=jnp.ones(5),
    )


class TestExactIdentities:
    def test_beckmann_gradient_is_route_cost(self):
        """THE Engine-2 identity: ∇ₓ Beckmann(x) = c(x), by autodiff, to 1e-12."""
        for net in (parallel_network(), braess_network()):
            x = jnp.linspace(1.0, 2.0, net.n_routes)
            grad = jax.grad(net.beckmann)(x)
            assert jnp.max(jnp.abs(grad - net.route_costs(x))) < TOL.identity

    def test_sue_satisfies_fisk_kkt(self):
        """At x*, c_a + θ⁻¹ log x_a is constant within each OD (Fisk 1980)."""
        theta = 2.0
        for net in (parallel_network(), braess_network()):
            x, residual, _ = solve_sue(net, theta, tol=1e-14)
            assert float(residual) < 1e-13
            g = net.route_costs(x) + jnp.log(x) / theta
            assert float(jnp.max(g) - jnp.min(g)) < 1e-10

    def test_sue_minimises_fisk_objective(self):
        theta = 1.5
        net = braess_network()
        x, _, _ = solve_sue(net, theta, tol=1e-14)
        base = float(net.fisk_objective(x, theta))
        q = population_tangent_basis(net)
        key = jax.random.PRNGKey(base_config().seeds.root + 30)
        for k in jax.random.split(key, 5):
            direction = q @ jax.random.normal(k, (q.shape[1],))
            perturbed = x + 1e-3 * direction
            assert float(net.fisk_objective(perturbed, theta)) > base

    def test_theta_limit_reaches_analytic_ue(self):
        """θ → ∞: SUE approaches the Wardrop UE (x₁ = 5/3 on the parallel net)."""
        net = parallel_network()
        x, _, _ = solve_sue(net, 200.0, tol=1e-14)
        assert abs(float(x[0]) - 5.0 / 3.0) < 5e-3
        costs = net.route_costs(x)
        assert abs(float(costs[0] - costs[1])) < 5e-3


class TestPotentiality:
    def test_df_is_symmetric(self):
        for net in (parallel_network(), braess_network()):
            x = jnp.linspace(1.0, 2.0, net.n_routes)
            assert df_symmetry_defect(net, x) < 1e-14
            df = payoff_field_jacobian(net, x)
            # Braess: shared links produce genuine off-diagonal coupling.
            if net.n_routes == 3:
                assert float(jnp.abs(df[0, 2])) > 1e-6

    def test_reciprocity_reads_zero(self):
        """The α = 0 anchor: toll response is symmetric on real route structure."""
        for net in (parallel_network(), braess_network()):
            assert population_reciprocity_defect(net, 1.5) < 1e-10


class TestSusceptibility:
    def test_matches_finite_differences(self):
        net = braess_network()
        theta = 1.2
        chi, _ = toll_susceptibility(net, theta)
        fd = toll_susceptibility_fd(net, theta)
        assert jnp.max(jnp.abs(chi - fd)) < 1e-5

    def test_rows_sum_zero(self):
        """Responses conserve demand: columns of χ live on the tangent space."""
        net = braess_network()
        chi, _ = toll_susceptibility(net, 1.2)
        assert jnp.max(jnp.abs(jnp.sum(chi, axis=0))) < 1e-10
