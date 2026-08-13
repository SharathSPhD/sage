"""Repeated games: the folk-theorem arithmetic against its closed form."""

import jax.numpy as jnp
import pytest
from strataq.core.defaults import base_config
from strataq.finite.games.tensor import DenseTensorGame
from strataq.problems.repeated import RepeatedProblem
from strataq.repeated.automata import Automaton, always, grim_trigger, tit_for_tat
from strataq.repeated.cycles import (
    alternating_logit_path,
    bertrand_ladder,
    detect_cycle,
    edgeworth_cycle,
    linear_market_demand,
)
from strataq.repeated.folk import (
    best_deviation,
    critical_discount,
    deviation_gains,
    grim_critical_discount,
    is_sustainable,
    logit_trigger_equilibrium,
    machine_values,
    minmax_payoffs,
    sustainable_payoff_set,
)

TOL = base_config().tolerances

# Classic Prisoner's Dilemma: action 0 = cooperate, 1 = defect.
# (C,C) = 3, (C,D) = 0/5, (D,D) = 1.  Grim critical delta = (5-3)/(5-1) = 1/2.
PD = DenseTensorGame(
    (
        jnp.array([[3.0, 0.0], [5.0, 1.0]]),
        jnp.array([[3.0, 5.0], [0.0, 1.0]]),
    )
)
COOPERATE = (0, 0)
DEFECT = (1, 1)

# Edgeworth setting: two firms, capacity below market demand, so the dearer firm
# still sells the residual and always has a reason to relent.
CYCLE_COSTS = [1.0, 1.0]
CYCLE_LADDER = [1.0 + 0.2 * i for i in range(11)]
CYCLE_DEMAND = linear_market_demand(10.0, 1.0)
CYCLE_CAPACITY = [5.0, 5.0]


class TestFolkTheoremArithmetic:
    def test_minmax_and_deviation_payoffs_on_the_prisoners_dilemma(self):
        assert jnp.allclose(minmax_payoffs(PD), jnp.array([1.0, 1.0]), atol=TOL.identity)
        assert jnp.allclose(best_deviation(PD, COOPERATE), jnp.array([5.0, 5.0]), atol=TOL.identity)

    def test_critical_delta_is_one_half(self):
        """The textbook answer: grim trigger sustains cooperation iff delta >= 1/2."""
        critical = grim_critical_discount(PD, COOPERATE)
        assert jnp.allclose(critical, jnp.array([0.5, 0.5]), atol=TOL.identity)

    def test_critical_delta_scales_with_the_temptation(self):
        """delta* = (d - u) / (d - p), so doubling the temptation gap moves it."""
        game = DenseTensorGame(
            (
                jnp.array([[3.0, 0.0], [9.0, 1.0]]),
                jnp.array([[3.0, 9.0], [0.0, 1.0]]),
            )
        )
        expected = (9.0 - 3.0) / (9.0 - 1.0)
        assert float(jnp.max(grim_critical_discount(game, COOPERATE))) == pytest.approx(expected)

    def test_a_stage_equilibrium_needs_no_patience(self):
        assert float(jnp.max(grim_critical_discount(PD, DEFECT))) == 0.0

    def test_sustainable_set_grows_with_patience(self):
        impatient = sustainable_payoff_set(PD, 0.4)
        patient = sustainable_payoff_set(PD, 0.6)
        assert int(jnp.sum(impatient.sustainable)) < int(jnp.sum(patient.sustainable))
        assert bool(patient.sustainable[0])  # (C, C) is index 0
        assert not bool(impatient.sustainable[0])

    def test_frontier_of_the_sustainable_set_is_cooperation(self):
        found = sustainable_payoff_set(PD, 0.9)
        frontier = found.frontier
        assert frontier.shape[0] >= 1
        assert any(bool(jnp.allclose(row, jnp.array([3.0, 3.0]))) for row in frontier)


class TestAutomata:
    def test_grim_trigger_values_are_the_stage_payoffs_when_nobody_deviates(self):
        machines = tuple(grim_trigger(PD.num_actions, i, COOPERATE, DEFECT) for i in range(2))
        solved = machine_values(PD, machines, 0.9)
        assert jnp.allclose(solved.path_value, jnp.array([3.0, 3.0]), atol=1e-12)

    def test_grim_trigger_punishment_state_is_absorbing(self):
        machine = grim_trigger(PD.num_actions, 0, COOPERATE, DEFECT)
        assert int(machine.transitions[(0, *COOPERATE)]) == 0
        assert int(machine.transitions[(0, 1, 0)]) == 1
        assert all(int(machine.transitions[(1, a, b)]) == 1 for a in range(2) for b in range(2))

    def test_one_shot_deviation_agrees_with_the_closed_form(self):
        """Bisection on the generic criterion must land on the analytic 1/2."""
        machines = tuple(grim_trigger(PD.num_actions, i, COOPERATE, DEFECT) for i in range(2))
        assert critical_discount(PD, machines) == pytest.approx(0.5, abs=1e-9)
        assert is_sustainable(PD, machines, 0.51)
        assert not is_sustainable(PD, machines, 0.49)

    def test_deviation_gains_vanish_exactly_at_the_critical_delta(self):
        machines = tuple(grim_trigger(PD.num_actions, i, COOPERATE, DEFECT) for i in range(2))
        gains = deviation_gains(PD, machines, 0.5)
        assert float(jnp.max(gains)) == pytest.approx(0.0, abs=1e-12)

    def test_tit_for_tat_cooperates_on_path_but_is_not_subgame_perfect(self):
        """The textbook result: after a defection the punisher would rather forgive.

        On the equilibrium path tit-for-tat sustains cooperation, but at the
        history where one side has defected the other is in an alternating
        punishment worth less than simply returning to cooperation, so the
        one-shot deviation criterion fails however patient the players are.
        """
        machines = tuple(tit_for_tat(PD.num_actions, i) for i in range(2))
        solved = machine_values(PD, machines, 0.9)
        assert jnp.allclose(solved.path_value, jnp.array([3.0, 3.0]), atol=1e-12)
        assert not is_sustainable(PD, machines, 0.9)
        assert not is_sustainable(PD, machines, 0.1)
        assert jnp.isnan(jnp.asarray(critical_discount(PD, machines)))

    def test_the_alternating_punishment_is_what_tit_for_tat_produces(self):
        """After a lone defection the profile echoes (D,C), (C,D), (D,C), ..."""
        machines = tuple(tit_for_tat(PD.num_actions, i) for i in range(2))
        solved = machine_values(PD, machines, 0.9)
        states = {tuple(int(v) for v in row): k for k, row in enumerate(solved.states)}
        echo = solved.actions[states[(1, 0)]]
        assert [int(a) for a in echo] == [1, 0]
        assert int(solved.successor[states[(1, 0)]]) == states[(0, 1)]

    def test_always_defect_is_sustainable_at_any_patience(self):
        machines = tuple(always(PD.num_actions, i, 1) for i in range(2))
        assert is_sustainable(PD, machines, 0.0)
        assert is_sustainable(PD, machines, 0.95)

    def test_bad_automata_are_rejected(self):
        with pytest.raises(ValueError, match="transitions must land"):
            Automaton([0], jnp.array([[[5, 0], [0, 0]]]))
        with pytest.raises(ValueError, match="at least one state"):
            Automaton([], jnp.zeros((0, 2, 2)))


class TestLogitTrigger:
    def test_sharp_precision_recovers_the_incentive_condition(self):
        """As lambda grows, cooperation goes to 1 above delta* and to 0 below it."""
        machines = tuple(grim_trigger(PD.num_actions, i, COOPERATE, DEFECT) for i in range(2))
        patient = logit_trigger_equilibrium(PD, machines, 0.8, 40.0)
        impatient = logit_trigger_equilibrium(PD, machines, 0.2, 40.0)
        start = int(patient.initial)
        assert float(patient.sigma[0][start, 0]) > 0.99
        assert float(impatient.sigma[0][start, 0]) < 0.01

    def test_zero_precision_is_uniform(self):
        machines = tuple(grim_trigger(PD.num_actions, i, COOPERATE, DEFECT) for i in range(2))
        point = logit_trigger_equilibrium(PD, machines, 0.5, 0.0)
        assert jnp.allclose(point.sigma[0], 0.5, atol=1e-10)

    def test_noise_demands_more_patience_than_the_folk_theorem_does(self):
        """At delta just above the critical value, moderate noise kills cooperation.

        Grim trigger sustains cooperation at delta = 0.6 in the sharp sense
        (delta* = 0.5), but a trigger that fires on slips is fragile: at moderate
        precision the cooperative fixed point is gone and only defection remains,
        and it comes back as precision rises. That gap between the sharp folk
        theorem and its quantal analogue is the point of the construction.
        """
        machines = tuple(grim_trigger(PD.num_actions, i, COOPERATE, DEFECT) for i in range(2))
        assert float(jnp.max(grim_critical_discount(PD, COOPERATE))) < 0.6
        moderate = logit_trigger_equilibrium(PD, machines, 0.6, 5.0)
        sharp = logit_trigger_equilibrium(PD, machines, 0.6, 20.0)
        assert float(moderate.sigma[0][0, 0]) < 0.05
        assert float(sharp.sigma[0][0, 0]) > 0.95

    def test_cooperation_probability_rises_with_patience(self):
        machines = tuple(grim_trigger(PD.num_actions, i, COOPERATE, DEFECT) for i in range(2))
        probabilities = [
            float(logit_trigger_equilibrium(PD, machines, delta, 3.0).sigma[0][0, 0])
            for delta in (0.1, 0.5, 0.9)
        ]
        assert probabilities[0] < probabilities[1] < probabilities[2]


class TestEdgeworthCycles:
    def test_the_ladder_game_is_winner_takes_all(self):
        demand = linear_market_demand(10.0, 1.0)
        game = bertrand_ladder([1.0, 1.0], [2.0, 3.0], demand)
        # (low, high): firm 0 undercuts and takes the whole market at price 2.
        assert float(game.payoffs[0][0, 1]) == pytest.approx((2.0 - 1.0) * 8.0)
        assert float(game.payoffs[1][0, 1]) == 0.0
        # A tie splits it.
        assert float(game.payoffs[0][1, 1]) == pytest.approx((3.0 - 1.0) * 7.0 / 2.0)

    def test_capacity_leaves_a_residual_for_the_expensive_firm(self):
        """Efficient rationing: the cheap firm sells its capacity, the dear one the rest."""
        demand = linear_market_demand(10.0, 1.0)
        game = bertrand_ladder([1.0, 1.0], [2.0, 3.0], demand, capacities=[5.0, 5.0])
        # (2.0, 3.0): firm 0 sells 5 at 2; firm 1 sells max(0, 7 - 5) = 2 at 3.
        assert float(game.payoffs[0][0, 1]) == pytest.approx((2.0 - 1.0) * 5.0)
        assert float(game.payoffs[1][0, 1]) == pytest.approx((3.0 - 1.0) * 2.0)

    def test_undercutting_dynamics_cycle_rather_than_settle(self):
        """With capacity, alternating logit response never reaches a fixed point."""
        cycle = edgeworth_cycle(
            CYCLE_COSTS,
            CYCLE_LADDER,
            CYCLE_DEMAND,
            60.0,
            capacities=CYCLE_CAPACITY,
            n_steps=200,
        )
        assert not bool(cycle.is_fixed_point)
        assert int(cycle.period) > 1
        assert float(cycle.amplitude) > 0.0
        assert float(cycle.trough) <= float(cycle.mean_price) <= float(cycle.peak)

    def test_low_precision_smooths_the_cycle_away(self):
        sharp = edgeworth_cycle(
            CYCLE_COSTS,
            CYCLE_LADDER,
            CYCLE_DEMAND,
            60.0,
            capacities=CYCLE_CAPACITY,
            n_steps=200,
        )
        blunt = edgeworth_cycle(
            CYCLE_COSTS,
            CYCLE_LADDER,
            CYCLE_DEMAND,
            0.02,
            capacities=CYCLE_CAPACITY,
            n_steps=200,
        )
        assert float(blunt.amplitude) < float(sharp.amplitude)

    def test_unlimited_capacity_settles_just_above_cost(self):
        """Textbook Bertrand has no cycle: matching beats undercutting to zero margin."""
        cycle = edgeworth_cycle(
            [1.0, 1.0],
            [1.0 + 0.1 * i for i in range(11)],
            linear_market_demand(10.0, 1.0),
            60.0,
            n_steps=200,
        )
        assert bool(cycle.is_fixed_point)
        assert float(cycle.mean_price) < 1.5

    def test_detect_cycle_finds_an_exact_period(self):
        series = jnp.asarray([[0.0], [1.0], [2.0], [0.0], [1.0], [2.0]])
        period, start = detect_cycle(series, tol=1e-12)
        assert period == 3
        assert start == 3

    def test_detect_cycle_reports_zero_on_a_transient(self):
        series = jnp.asarray([[float(i)] for i in range(10)])
        period, _ = detect_cycle(series, tol=1e-12)
        assert period == 0

    def test_path_shape_and_normalisation(self):
        game = bertrand_ladder([1.0, 1.0], [1.5, 2.0, 2.5], linear_market_demand(10.0, 1.0))
        path = alternating_logit_path(game, 5.0, n_steps=6)
        assert path.shape == (7, 2, 3)
        assert jnp.allclose(jnp.sum(path, axis=-1), 1.0, atol=1e-12)


class TestRepeatedProblem:
    def test_solves_the_prisoners_dilemma_and_names_the_critical_delta(self):
        res = RepeatedProblem(payoffs=PD.payoffs, discount=0.6).solve()
        assert res.critical_discount == pytest.approx(0.5)
        assert res.sustainable
        assert [int(a) for a in res.target] == [0, 0]
        assert [float(u) for u in res.target_payoffs] == [3.0, 3.0]
        assert "0.5" in res.message

    def test_impatient_players_cannot_sustain_cooperation(self):
        res = RepeatedProblem(payoffs=PD.payoffs, discount=0.3).solve()
        assert not res.sustainable
        assert res.sustainable_profiles.shape[0] >= 1  # (D, D) always survives

    def test_nash_reversion_punishment_matches_minmax_here(self):
        minmax = RepeatedProblem(payoffs=PD.payoffs, discount=0.6, punishment="minmax").solve()
        nash = RepeatedProblem(payoffs=PD.payoffs, discount=0.6, punishment="nash").solve()
        assert minmax.critical_discount == pytest.approx(nash.critical_discount)

    def test_precision_adds_a_cooperation_probability(self):
        res = RepeatedProblem(payoffs=PD.payoffs, discount=0.8, precision=20.0).solve()
        assert res.cooperation_probability is not None
        assert res.cooperation_probability > 0.9

    def test_summary_and_dict_round_trip(self):
        res = RepeatedProblem(payoffs=PD.payoffs, discount=0.6).solve()
        text = str(res.summary())
        assert "RepeatedProblem" in text
        body = res.as_dict()
        assert body["critical_discount"] == pytest.approx(0.5)
        assert body["sustainable"] is True

    def test_bad_input_is_rejected_before_the_solver(self):
        with pytest.raises(ValueError, match=r"discount must be in \[0, 1\)"):
            RepeatedProblem(payoffs=PD.payoffs, discount=1.0)
        with pytest.raises(ValueError, match="target must have"):
            RepeatedProblem(payoffs=PD.payoffs, discount=0.5, target=(0,))
        with pytest.raises(ValueError, match="punishment must be"):
            RepeatedProblem(payoffs=PD.payoffs, discount=0.5, punishment="revenge")
