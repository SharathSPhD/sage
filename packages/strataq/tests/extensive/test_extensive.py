"""Game trees: representation, Kuhn's theorem, backward induction and AQRE."""

import jax.numpy as jnp
import pytest
from strataq.core.defaults import base_config
from strataq.extensive.aqre import agent_qre, agent_qre_branch
from strataq.extensive.backward import backward_induction, verify_subgame_perfection
from strataq.extensive.behaviour import (
    behaviour_to_mixed,
    expected_payoffs,
    mixed_to_behaviour,
    pure_strategies,
    reach_probabilities,
    realisation_gap,
    reduced_normal_form,
    uniform_behaviour,
)
from strataq.extensive.catalogue import (
    bargaining,
    build,
    centipede,
    entry_deterrence,
    kuhn_poker,
    seltens_horse,
)
from strataq.extensive.tree import ExtensiveGame, perfect_recall_violations
from strataq.extensive.validate import HAVE_GAMBIT, gambit_agent_qre, max_behaviour_gap
from strataq.problems.extensive import ExtensiveProblem

TOL = base_config().tolerances


class TestTree:
    def test_entry_deterrence_shape(self):
        game = entry_deterrence()
        assert game.n_nodes == 5
        assert game.n_infosets == 2
        assert game.n_players == 2
        assert game.is_perfect_information
        assert not game.has_chance

    def test_round_trips_through_the_dict_form(self):
        for game in (entry_deterrence(), centipede(4), seltens_horse(), kuhn_poker()):
            rebuilt = ExtensiveGame.from_dict(game.to_dict())
            assert rebuilt.n_nodes == game.n_nodes
            assert rebuilt.n_infosets == game.n_infosets
            assert jnp.allclose(rebuilt.payoffs, game.payoffs)
            assert jnp.array_equal(rebuilt.player, game.player)
            assert jnp.array_equal(rebuilt.infoset, game.infoset)

    def test_shared_information_sets_are_recognised(self):
        game = seltens_horse()
        assert not game.is_perfect_information
        shared = [h for h in range(game.n_infosets) if len(game.infoset_members(h)) > 1]
        assert len(shared) == 1
        assert len(game.infoset_members(shared[0])) == 2

    def test_kuhn_poker_size_and_chance(self):
        game = kuhn_poker()
        assert game.has_chance
        assert game.n_infosets == 12
        assert game.n_nodes == 55
        assert len(pure_strategies(game, 0)) == 64
        assert len(pure_strategies(game, 1)) == 64

    def test_all_catalogue_trees_have_perfect_recall(self):
        for name in ("entry_deterrence", "centipede", "bargaining", "seltens_horse", "kuhn_poker"):
            assert perfect_recall_violations(build(name)) == ()

    def test_reach_probabilities_sum_to_one_over_the_leaves(self):
        game = kuhn_poker()
        from strataq.extensive.behaviour import policy_from_behaviour

        policy = policy_from_behaviour(game, uniform_behaviour(game))
        reach = reach_probabilities(game, policy)
        assert float(jnp.sum(reach[game.is_terminal])) == pytest.approx(1.0, abs=1e-12)

    def test_malformed_specs_are_rejected(self):
        with pytest.raises(ValueError, match="'players' and 'root'"):
            ExtensiveGame.from_dict({"root": {"payoffs": [1.0]}})
        with pytest.raises(ValueError, match="sum to"):
            ExtensiveGame.from_dict(
                {
                    "players": ["A"],
                    "root": {
                        "player": "chance",
                        "probs": [0.3, 0.3],
                        "children": [{"payoffs": [1.0]}, {"payoffs": [2.0]}],
                    },
                }
            )
        with pytest.raises(ValueError, match="payoffs' nor 'children'"):
            ExtensiveGame.from_dict({"players": ["A"], "root": {"player": "A"}})


class TestKuhnsTheorem:
    def test_behaviour_and_mixed_are_realisation_equivalent(self):
        for game in (entry_deterrence(), centipede(4), kuhn_poker()):
            point = agent_qre(game, 1.5)
            for player in range(game.n_players):
                gap = realisation_gap(game, point.behaviour, player, n_probes=4)
                assert gap < 1e-10

    def test_the_two_maps_are_inverse_on_behaviour_strategies(self):
        game = centipede(4)
        point = agent_qre(game, 2.0)
        for player in range(game.n_players):
            mixed = behaviour_to_mixed(game, point.behaviour, player)
            back = mixed_to_behaviour(game, mixed, player)
            for h in game.player_infosets(player):
                assert jnp.allclose(back[h], point.behaviour[h], atol=1e-10)

    def test_the_reduced_normal_form_reproduces_tree_payoffs(self):
        game = entry_deterrence()
        normal = reduced_normal_form(game)
        assert normal.num_actions == (2, 2)
        # Pure plan (in, fight) -> (-1, -1); (out, accommodate) -> (0, 2).
        assert float(normal.payoffs[0][0, 0]) == pytest.approx(-1.0)
        assert float(normal.payoffs[1][1, 1]) == pytest.approx(2.0)

    def test_mixed_to_behaviour_needs_perfect_recall(self):
        game = ExtensiveGame.from_dict(
            {
                "players": ["A"],
                "root": {
                    "player": "A",
                    "infoset": "first",
                    "actions": ["l", "r"],
                    "children": [
                        {
                            "player": "A",
                            "infoset": "second",
                            "actions": ["u", "d"],
                            "children": [{"payoffs": [1.0]}, {"payoffs": [2.0]}],
                        },
                        {
                            "player": "A",
                            "infoset": "second",
                            "actions": ["u", "d"],
                            "children": [{"payoffs": [3.0]}, {"payoffs": [4.0]}],
                        },
                    ],
                },
            }
        )
        assert perfect_recall_violations(game) == ("second",)
        with pytest.raises(ValueError, match="perfect recall"):
            mixed_to_behaviour(game, jnp.ones((4,)) / 4.0, 0)


class TestBackwardInduction:
    def test_entry_deterrence_accommodates_and_the_entrant_enters(self):
        game = entry_deterrence()
        solution = backward_induction(game)
        assert jnp.allclose(solution.value, jnp.array([1.0, 1.0]))
        labels = {
            game.infoset_labels[h]: game.action_labels[h][int(jnp.argmax(solution.behaviour[h]))]
            for h in range(game.n_infosets)
        }
        assert labels == {"enter?": "in", "respond": "accommodate"}
        assert not solution.has_ties
        assert verify_subgame_perfection(game, solution) == pytest.approx(0.0, abs=1e-12)

    def test_a_credible_threat_deters_entry(self):
        """Raise the cost of accommodation and the incumbent's threat becomes real."""
        game = entry_deterrence(duopoly=-2.0, fight=-3.0, monopoly=5.0)
        solution = backward_induction(game)
        assert float(solution.value[0]) == 0.0
        assert float(solution.value[1]) == 5.0

    def test_centipede_backward_induction_takes_immediately(self):
        game = centipede(6)
        solution = backward_induction(game)
        assert int(solution.action[0]) == 0  # "take"
        assert jnp.allclose(solution.value, jnp.array([0.4, 0.1]))

    def test_the_four_move_centipede_is_the_mckelvey_palfrey_experiment(self):
        game = centipede(4)
        leaves = sorted(
            (
                [round(float(v), 4) for v in game.payoffs[n]]
                for n in range(game.n_nodes)
                if bool(game.is_terminal[n])
            ),
        )
        assert leaves == sorted([[0.4, 0.1], [0.2, 0.8], [1.6, 0.4], [0.8, 3.2], [6.4, 1.6]])

    def test_two_stage_bargaining_splits_one_minus_delta(self):
        """The discrete Rubinstein solution: (1 - delta, delta) when it is on the grid."""
        game = bargaining(discount=0.75, n_offers=5)
        solution = backward_induction(game)
        assert float(solution.value[0]) == pytest.approx(0.25, abs=1e-12)
        assert float(solution.value[1]) == pytest.approx(0.75, abs=1e-12)

    def test_bargaining_is_a_perfect_information_tree_with_chance_free_payoffs(self):
        game = bargaining()
        assert game.is_perfect_information
        assert verify_subgame_perfection(game, backward_induction(game)) < 1e-12

    def test_backward_induction_refuses_imperfect_information(self):
        with pytest.raises(ValueError, match="perfect information"):
            backward_induction(seltens_horse())


class TestAgentQRE:
    def test_zero_precision_is_uniform_behaviour(self):
        game = kuhn_poker()
        point = agent_qre(game, 0.0)
        assert jnp.allclose(point.behaviour[game.action_mask()], 0.5, atol=1e-12)

    def test_entry_deterrence_aqre_is_the_hand_computation(self):
        """Incumbent: softmax(lam * [-1, 1]); entrant: softmax(lam * [E, 0])."""
        game = entry_deterrence()
        lam = 0.5
        point = agent_qre(game, lam, tol=1e-14, max_iter=500_000)
        fight = float(jnp.exp(-lam) / (jnp.exp(-lam) + jnp.exp(lam)))
        assert float(point.behaviour[1, 0]) == pytest.approx(fight, abs=1e-10)
        entry_value = fight * (-1.0) + (1.0 - fight) * 1.0
        enter = float(jnp.exp(lam * entry_value) / (jnp.exp(lam * entry_value) + 1.0))
        assert float(point.behaviour[0, 0]) == pytest.approx(enter, abs=1e-10)

    def test_high_precision_approaches_backward_induction(self):
        game = entry_deterrence()
        point = agent_qre(game, 60.0, tol=1e-13, max_iter=500_000)
        induction = backward_induction(game)
        assert max_behaviour_gap(point.behaviour, induction.behaviour) < 1e-8

    def test_centipede_play_does_not_stop_immediately(self):
        """The known qualitative result: AQRE keeps the centipede alive for many moves."""
        game = centipede(6)
        point = agent_qre(game, 2.0, tol=1e-13, max_iter=500_000)
        first_pass = float(point.behaviour[0, 1])
        assert first_pass > 0.5
        # Play reaches the third decision node, which backward induction says is
        # unreachable, and the root payoff beats the backward-induction 0.40.
        reach = point.reach
        third = [
            n for n in range(game.n_nodes) if int(game.depth[n]) == 2 and int(game.player[n]) >= 0
        ]
        assert third and float(jnp.max(reach[jnp.asarray(third)])) > 0.05
        assert float(point.expected_payoffs[0]) > 0.4

    def test_centipede_passing_falls_as_precision_rises(self):
        game = centipede(6)
        probabilities = [
            float(agent_qre(game, lam, tol=1e-12, max_iter=500_000).behaviour[0, 1])
            for lam in (1.0, 4.0, 16.0)
        ]
        assert probabilities[0] > probabilities[1] > probabilities[2]

    def test_the_branch_is_continuous_and_starts_uniform(self):
        game = entry_deterrence()
        grid, trace = agent_qre_branch(game, 8.0, n_points=20)
        assert grid.shape == (20,)
        assert jnp.allclose(trace[0][game.action_mask()], 0.5, atol=1e-10)
        steps = jnp.max(jnp.abs(jnp.diff(trace, axis=0)))
        assert float(steps) < 0.5

    def test_seltens_horse_has_an_interior_aqre(self):
        game = seltens_horse()
        point = agent_qre(game, 1.0, tol=1e-13, max_iter=500_000)
        assert bool(point.converged)
        mask = game.action_mask()
        assert float(jnp.min(point.behaviour[mask])) > 0.0
        assert float(jnp.max(point.behaviour[mask])) < 1.0

    def test_kuhn_poker_is_zero_sum_at_every_precision(self):
        game = kuhn_poker()
        for lam in (0.5, 3.0, 10.0):
            point = agent_qre(game, lam, tol=1e-12, max_iter=500_000)
            assert float(jnp.sum(point.expected_payoffs)) == pytest.approx(0.0, abs=1e-10)

    def test_kuhn_poker_value_approaches_minus_one_eighteenth(self):
        """The game value to player One is -1/18; the AQRE branch runs toward it."""
        game = kuhn_poker()
        point = agent_qre(game, 30.0, tol=1e-12, max_iter=500_000)
        value = float(point.expected_payoffs[0])
        assert -0.12 < value < 0.0
        assert abs(value - (-1.0 / 18.0)) < 0.06

    def test_behaviour_rows_are_distributions(self):
        game = kuhn_poker()
        point = agent_qre(game, 2.0)
        assert jnp.allclose(jnp.sum(point.behaviour, axis=1), 1.0, atol=1e-12)
        assert float(jnp.sum(expected_payoffs(game, point.behaviour))) == pytest.approx(
            0.0, abs=1e-10
        )


@pytest.mark.skipif(not HAVE_GAMBIT, reason="pygambit is not installed")
class TestAgainstGambit:
    @pytest.mark.parametrize("lam", [0.5, 2.0, 5.0])
    def test_entry_deterrence_matches_gambit(self, lam):
        game = entry_deterrence()
        ours = agent_qre(game, lam, tol=1e-13, max_iter=500_000).behaviour
        theirs = gambit_agent_qre(game, lam)
        assert max_behaviour_gap(ours, theirs) < TOL.oracle

    def test_centipede_matches_gambit(self):
        game = centipede(4)
        ours = agent_qre(game, 1.5, tol=1e-13, max_iter=500_000).behaviour
        theirs = gambit_agent_qre(game, 1.5)
        assert max_behaviour_gap(ours, theirs) < TOL.oracle

    def test_seltens_horse_matches_gambit(self):
        game = seltens_horse()
        ours = agent_qre(game, 1.0, tol=1e-13, max_iter=500_000).behaviour
        theirs = gambit_agent_qre(game, 1.0)
        assert max_behaviour_gap(ours, theirs) < TOL.oracle

    def test_kuhn_poker_matches_gambit(self):
        game = kuhn_poker()
        ours = agent_qre(game, 2.0, tol=1e-13, max_iter=500_000).behaviour
        theirs = gambit_agent_qre(game, 2.0)
        assert max_behaviour_gap(ours, theirs) < TOL.oracle


class TestExtensiveProblem:
    def test_catalogue_by_name(self):
        res = ExtensiveProblem(tree="entry_deterrence", precision=3.0).solve()
        assert res.success
        assert res.perfect_information
        assert res.subgame_perfect_actions == ("in", "accommodate")
        assert res.n_nodes == 5

    def test_imperfect_information_has_no_backward_induction(self):
        res = ExtensiveProblem(tree="seltens_horse", precision=1.0).solve()
        assert not res.perfect_information
        assert res.subgame_perfect is None
        assert res.divergence is None

    def test_centipede_divergence_is_large_at_low_precision(self):
        res = ExtensiveProblem(tree="centipede", precision=1.0, options={"n_moves": 6}).solve()
        assert res.divergence is not None and res.divergence > 0.5

    def test_summary_and_dict(self):
        res = ExtensiveProblem(tree="entry_deterrence", precision=2.0).solve()
        assert "ExtensiveProblem" in str(res.summary())
        body = res.as_dict()
        assert body["n_infosets"] == 2
        assert len(body["behaviour"]) == 2
        assert body["recommended"] in (["in", "accommodate"], ["out", "accommodate"])

    def test_accepts_a_raw_spec(self):
        spec = entry_deterrence().to_dict()
        res = ExtensiveProblem(tree=spec, precision=1.0).solve()
        assert res.n_nodes == 5

    def test_rejects_an_unknown_tree(self):
        with pytest.raises(ValueError, match="unknown tree"):
            ExtensiveProblem(tree="poker_night")
