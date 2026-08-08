"""Blotto plugin: contract compliance + the instruments read what they must."""

import jax.numpy as jnp
from strataq.core.protocols import ActionGridBuilder, DomainPlugin, PayoffOracle
from strataq.core.solve.fixedpoint import logit_qre
from strataq.domains.blotto import PLUGIN, BlottoGridBuilder, BlottoOracle, blotto_game_tensors
from strataq.domains.blotto.oracle import allocations
from strataq.finite.decompose.hodge import alpha
from strataq.finite.games.tensor import DenseTensorGame
from strataq.finite.response.reciprocity import reciprocity_defect
from strataq.thermo.exact import thermo_read

VALUES = jnp.array([1.0, 1.0, 1.0])


class TestContract:
    def test_plugin_shape(self):
        assert isinstance(PLUGIN, DomainPlugin)
        assert PLUGIN.engine == "finite"
        assert PLUGIN.response_instruments_available  # budgets are a real field
        assert PLUGIN.loader_factory is None  # synthetic-only, honestly

    def test_oracle_conforms_structurally(self):
        oracle = BlottoOracle(VALUES)
        assert isinstance(oracle, PayoffOracle)
        assert isinstance(BlottoGridBuilder([3, 3], 3), ActionGridBuilder)

    def test_allocations_enumerate_the_simplex(self):
        grid = allocations(3, 3)
        assert len(grid) == 10  # C(3+3-1, 3-1)
        assert all(sum(a) == 3 for a in grid)
        assert len(set(grid)) == len(grid)

    def test_zero_sum_by_construction(self):
        oracle = BlottoOracle(VALUES)
        pay = oracle.profit(jnp.asarray([[2.0, 1.0, 0.0], [1.0, 1.0, 1.0]]))
        assert abs(float(pay[0] + pay[1]) - float(jnp.sum(VALUES))) < 1e-12
        # a beats field 1 (2>1), ties field 2, loses field 3 -> 1.5
        assert abs(float(pay[0]) - 1.5) < 1e-12


class TestInstrumentsOnBlotto:
    def test_alpha_is_high(self):
        """Zero-sum with symmetric budgets: strongly non-potential."""
        u_a, u_b, _, _ = blotto_game_tensors(BlottoOracle(VALUES), (3, 3))
        game = DenseTensorGame((u_a, u_b))
        assert alpha(game) > 0.6

    def test_reciprocity_reads_positive(self):
        u_a, u_b, _, _ = blotto_game_tensors(BlottoOracle(VALUES), (3, 3))
        game = DenseTensorGame((u_a, u_b))
        point = logit_qre(game, 1.5, tol=1e-12, max_iter=100_000)
        assert float(reciprocity_defect(game, point)) > 0.1

    def test_equal_values_budget2_is_degenerate_and_reads_zero(self):
        """2 equal fields, budget 2: every profile ties at 1–1 — constant
        payoffs, so the meters must read exactly zero. (Found while writing
        the positive test: a correct null reading, kept as a regression.)"""
        u_a, u_b, _, _ = blotto_game_tensors(BlottoOracle(jnp.array([1.0, 1.0])), (2, 2))
        assert float(jnp.max(u_a) - jnp.min(u_a)) < 1e-12
        reading = thermo_read(DenseTensorGame((u_a, u_b)), 2.0)
        assert float(reading.epr) < 1e-12

    def test_dissipation_positive_and_currents_circulate(self):
        """The DOMAINS v1 §4.2 question: non-potential allocation competition
        carries measurable probability current (asymmetric field values)."""
        u_a, u_b, _, _ = blotto_game_tensors(BlottoOracle(jnp.array([2.0, 1.0])), (2, 2))
        game = DenseTensorGame((u_a, u_b))
        reading = thermo_read(game, 2.0)
        assert float(reading.epr) > 1e-4
        assert float(reading.max_current) > 1e-5

    def test_budget_field_moves_payoffs(self):
        """The conjugate field is live: a budget increment changes the game."""
        oracle = BlottoOracle(VALUES)
        u3, _, grid3, _ = blotto_game_tensors(oracle, (3, 3))
        u4, _, grid4, _ = blotto_game_tensors(oracle, (4, 3))
        assert len(grid4) > len(grid3)
        # richer grid weakly improves the best response value at any rival mix
        assert float(jnp.max(u4)) >= float(jnp.max(u3))
