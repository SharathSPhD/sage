"""solve_situation(): the whole recommendation, assembled once."""

import jax.numpy as jnp
import pytest
import strataq as sq
from strataq.problems.situation import Situation, solve_situation

# Prisoner's Dilemma, action 0 = cooperate, 1 = defect.
PD = [
    [[3.0, 0.0], [5.0, 1.0]],
    [[3.0, 5.0], [0.0, 1.0]],
]
# Safe (row 0) pays 2 whatever happens; risky (row 1) pays 5 against the rival's
# first action and 0 against its second — and the rival's second action is
# strictly dominant. Sharpen the precision and the recommendation flips.
RISK = [
    [[2.0, 2.0], [5.0, 0.0]],
    [[0.0, 1.0], [0.0, 1.0]],
]


class TestRecommendation:
    def test_the_dominant_action_is_recommended(self):
        res = solve_situation(PD, actions=["cooperate", "defect"], precision=2.0)
        assert res.action_label == "defect"
        assert res.action == 1
        assert res.success

    def test_alternatives_are_ranked_with_zero_regret_at_the_top(self):
        res = solve_situation(PD, actions=["cooperate", "defect"], precision=2.0)
        assert [a.label for a in res.alternatives] == ["defect", "cooperate"]
        assert res.alternatives[0].regret == 0.0
        assert res.alternatives[1].regret > 0.0
        assert res.runner_up is not None and res.runner_up.label == "cooperate"

    def test_alternative_payoffs_match_the_equilibrium_utilities(self):
        res = solve_situation(PD, precision=1.5)
        curve = res.point.expected_payoffs[res.you]
        for alternative in res.alternatives:
            assert alternative.expected_payoff == pytest.approx(float(curve[alternative.action]))

    def test_the_rival_distribution_is_a_distribution_with_names(self):
        res = solve_situation(
            PD,
            actions=["cooperate", "defect"],
            rival_actions=[["cooperate", "defect"]],
            players=["you", "them"],
            precision=2.0,
        )
        assert len(res.rivals) == 1
        rival = res.rivals[0]
        assert rival.label == "them"
        assert sum(rival.distribution) == pytest.approx(1.0)
        assert rival.most_likely == "defect"
        assert 0.0 <= rival.entropy <= float(jnp.log(2.0)) + 1e-12

    def test_confidence_is_the_normalised_gap_to_the_runner_up(self):
        res = solve_situation(PD, precision=2.0)
        gap = res.expected_payoff - res.alternatives[1].expected_payoff
        assert res.confidence == pytest.approx(gap / float(res.game.payoff_range))
        assert res.confidence > 0.0

    def test_three_players_report_two_rivals(self):
        payoffs = [jnp.zeros((2, 2, 2)) + i for i in range(3)]
        res = solve_situation(payoffs, you=1, precision=1.0)
        assert len(res.rivals) == 2
        assert [r.player for r in res.rivals] == [0, 2]


class TestSensitivity:
    def test_a_dominant_action_is_stable_across_the_whole_ladder(self):
        res = solve_situation(PD, precision=2.0)
        assert res.sensitivity.stable
        assert res.sensitivity.robustness == 1.0
        assert res.sensitivity.switch_precision is None
        assert len(res.sensitivity.precisions) == len(res.sensitivity.recommended)

    def test_the_ladder_brackets_the_stated_precision(self):
        res = solve_situation(PD, precision=2.0)
        low, high = res.sensitivity.precisions[0], res.sensitivity.precisions[-1]
        assert low < 2.0 < high

    def test_an_explicit_ladder_is_honoured(self):
        res = solve_situation(PD, precision=1.0, ladder=[0.1, 1.0, 10.0])
        assert res.sensitivity.precisions == (0.1, 1.0, 10.0)

    def test_a_recommendation_that_flips_reports_where(self):
        """The gamble is worth taking against a coin flip and not against a sure thing."""
        res = solve_situation(
            RISK, actions=["safe", "risky"], precision=8.0, ladder=[0.02, 0.2, 2.0, 20.0]
        )
        assert res.action_label == "safe"
        assert not res.sensitivity.stable
        assert res.sensitivity.switch_precision == 0.02
        assert res.sensitivity.robustness < 1.0


class TestSurface:
    def test_reachable_from_the_top_level(self):
        assert sq.solve_situation is solve_situation
        assert issubclass(sq.Situation, sq.Problem)

    def test_summary_and_dict(self):
        res = solve_situation(PD, actions=["cooperate", "defect"], precision=2.0)
        text = str(res.summary())
        assert "Situation" in text and "defect" in text
        body = res.as_dict()
        assert body["action_label"] == "defect"
        assert len(body["alternatives"]) == 2
        assert len(body["rivals"]) == 1
        assert body["sensitivity"]["stable"] is True

    def test_diagnostics_are_off_the_answer(self):
        res = solve_situation(PD, precision=2.0)
        assert res.diagnostics.residual < 1e-8
        assert res.diagnostics.alpha is not None

    def test_problem_form_matches_the_function_form(self):
        a = Situation(payoffs=PD, precision=1.7).solve()
        b = solve_situation(PD, precision=1.7)
        assert a.action == b.action
        assert a.expected_payoff == pytest.approx(b.expected_payoff)

    def test_bad_input_is_rejected(self):
        with pytest.raises(ValueError, match="you must be in"):
            solve_situation(PD, you=5)
        with pytest.raises(ValueError, match="precision must be"):
            solve_situation(PD, precision=0.0)
        with pytest.raises(ValueError, match="expected 2 names"):
            solve_situation(PD, actions=["only one"])
