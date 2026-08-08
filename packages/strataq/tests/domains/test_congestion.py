"""Congestion plugin on real Sioux Falls data: the calibration standard.

Network access: the loader fetches (and caches) from the TransportationNetworks
repository; tests skip cleanly when offline.
"""

import jax.numpy as jnp
import pytest
from strataq.core.protocols import DatasetLoader, DomainPlugin, PayoffOracle
from strataq.domains.congestion import (
    PLUGIN,
    BPROracle,
    TNTPLoader,
    routing_network_from_tntp,
)
from strataq.population.games.routing import solve_sue
from strataq.population.response.susceptibility import (
    df_symmetry_defect,
    population_reciprocity_defect,
)


def _sioux_falls():
    try:
        return TNTPLoader().load()
    except OSError:
        pytest.skip("TNTP fetch unavailable (offline)")


class TestContract:
    def test_plugin_shape(self):
        assert isinstance(PLUGIN, DomainPlugin)
        assert PLUGIN.engine == "population"
        assert PLUGIN.response_instruments_available
        assert PLUGIN.field_spec.linearity == "exact"  # tolls: the cleanest field

    def test_loader_validates_loudly(self):
        loader = TNTPLoader()
        assert isinstance(loader, DatasetLoader)
        report = loader.validate() if _sioux_falls() else ""
        assert "SiouxFalls" in report
        assert "REFUSING" not in report
        assert "76 links" in report


@pytest.fixture(scope="module")
def network():
    net = _sioux_falls()
    od_pairs = sorted(net.demand, key=net.demand.get, reverse=True)[:8]
    return routing_network_from_tntp(net, od_pairs, k_routes=3)


class TestSiouxFallsCalibration:
    def test_oracle_conforms(self, network):
        oracle = BPROracle(network)
        assert isinstance(oracle, PayoffOracle)
        x = jnp.ones(network.n_routes)
        assert oracle.profit(x).shape == (network.n_routes,)

    def test_sue_solves_and_satisfies_kkt(self, network):
        theta = 0.5
        x, residual, _ = solve_sue(network, theta, tol=1e-12)
        assert float(residual) < 1e-10
        costs = network.route_costs(x)
        for od in range(network.n_od):
            idx = jnp.where(network.od_index == od)[0]
            g = costs[idx] + jnp.log(x[idx]) / theta
            assert float(jnp.max(g) - jnp.min(g)) < 1e-8

    def test_df_symmetric_on_real_network(self, network):
        x, _, _ = solve_sue(network, 0.5, tol=1e-12)
        assert df_symmetry_defect(network, x) < 1e-14

    def test_reciprocity_reads_zero_on_real_network(self, network):
        """The programme's calibration standard: ℛ = 0 on real network data.
        A nonzero here is a bug in the code, not the world."""
        assert population_reciprocity_defect(network, 0.5) < 1e-10

    def test_tolls_shift_flows_the_right_way(self, network):
        """The conjugate field is live and signed correctly: tolling every link
        of the most-used route reduces its flow."""
        theta = 0.5
        x0, _, _ = solve_sue(network, theta, tol=1e-12)
        busiest = int(jnp.argmax(x0))
        tolls = 5.0 * network.incidence[busiest]
        x1, _, _ = solve_sue(network, theta, tolls=tolls, tol=1e-12)
        assert float(x1[busiest]) < float(x0[busiest])


class TestParsersOffline:
    """Parser/validator branches on inline text — no network required."""

    NET_TEXT = """<NUMBER OF NODES> 3
<END OF METADATA>
~ from to cap len fft b power speed toll type ;
1 2 100.0 1.0 2.0 0.15 4.0 0 0 1 ;
2 3 200.0 1.0 3.0 0.15 4.0 0 0 1 ;
"""
    TRIPS_TEXT = """<END OF METADATA>
Origin 1
 2 : 50.0; 3 : 25.0;
Origin 2
 3 : 10.0;
"""

    def test_parse_net(self):
        from strataq.domains.congestion.tntp import parse_net

        net = parse_net(self.NET_TEXT, "toy")
        assert net.n_links == 2
        assert net.free_flow_time == (2.0, 3.0)
        assert net.capacity == (100.0, 200.0)

    def test_parse_trips_skips_self_pairs(self):
        from strataq.domains.congestion.tntp import parse_trips

        demand = parse_trips(self.TRIPS_TEXT + "Origin 3\n 3 : 5.0;\n")
        assert demand == {(1, 2): 50.0, (1, 3): 25.0, (2, 3): 10.0}

    def test_validate_refuses_empty_od(self):
        from strataq.domains.congestion.tntp import parse_net, validate

        net = parse_net(self.NET_TEXT, "toy")
        report = validate(net)
        assert "REFUSING" in report and "empty OD matrix" in report

    def test_validate_refuses_bad_capacity(self):
        from strataq.domains.congestion.tntp import TNTPNetwork, validate

        net = TNTPNetwork(
            name="bad",
            n_nodes=2,
            init_node=(1,),
            term_node=(2,),
            capacity=(0.0,),
            free_flow_time=(1.0,),
            b=(0.15,),
            power=(4.0,),
            demand={(1, 2): 1.0},
        )
        assert "non-positive capacities" in validate(net)

    def test_loader_rejects_unknown_network(self):
        import pytest as _pytest
        from strataq.domains.congestion import TNTPLoader

        with _pytest.raises(ValueError, match="SiouxFalls only"):
            TNTPLoader("Anaheim")

    def test_k_shortest_on_toy_graph(self):
        from strataq.domains.congestion.tntp import k_shortest_routes, parse_net

        text = """<END OF METADATA>
1 2 1 1 1.0 0.15 4 0 0 1 ;
2 3 1 1 1.0 0.15 4 0 0 1 ;
1 3 1 1 5.0 0.15 4 0 0 1 ;
"""
        net = parse_net(text, "toy")
        routes = k_shortest_routes(net, 1, 3, 2)
        assert routes[0] == (0, 1)  # 1->2->3 (cost 2) beats direct (cost 5)
        assert routes[1] == (2,)
