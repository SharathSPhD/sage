"""Data-facing Hatano–Sasa estimator (unit thermo.hs_estimator).

The estimator only ever sees state sequences; its validity meter is the
IFT itself — ⟨e^{−Ŷ}⟩ must bracket 1, and when hold windows are too short
to estimate the stationary distributions the meter must flag itself
unusable rather than quote a number.
"""

import jax.numpy as jnp
import numpy as np
from strataq.finite.decompose.generate import make_family
from strataq.finite.games.library import coordination, matching_pennies
from strataq.thermo.hs_estimator import hs_y_estimate, sample_quench_states
from strataq.thermo.protocols import QuenchProtocol, hatano_sasa_exact

# alpha=0.25 with a steep ramp: exact <Y> ~ 0.25 at fast holds — enough
# signal for the diagnostic to have power (alpha=0.5 gentle ramps sit at
# <Y> ~ 0.01, below plug-in noise — measured during design)
GAME = make_family(coordination(2, 2, bonus=2.0), matching_pennies(), [0.25])[0]
LAMBDAS = jnp.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])


def _proto(tau: float) -> QuenchProtocol:
    return QuenchProtocol(lambdas=LAMBDAS, taus=jnp.full((len(LAMBDAS) - 1,), tau))


class TestLongHoldRecovery:
    def test_recovers_exact_mean_y_and_gates_pass(self):
        """tau=24: every hold clears the relaxation gate (slowest window's
        estimated relaxation ~5, x3 safety) — only then is coverage owed."""
        proto = _proto(24.0)
        # n=800: the IFT companion is an equivalence test — its CI half-width
        # must fit inside the tolerance band, which n=400 barely misses
        windows = sample_quench_states(
            GAME, proto, n_trajectories=800, steps_per_unit_time=25, seed=1
        )
        est = hs_y_estimate(windows, n_states=4, hold_durations=[24.0] * len(windows))
        exact = float(hatano_sasa_exact(GAME, proto)[1])
        assert est.usable
        assert est.mean_y_ci_low <= exact <= est.mean_y_ci_high


class TestShortHoldSelfFlags:
    def test_short_windows_flag_unusable(self):
        proto = _proto(0.25)
        windows = sample_quench_states(
            GAME, proto, n_trajectories=400, steps_per_unit_time=25, seed=2
        )
        est = hs_y_estimate(windows, n_states=4, hold_durations=[0.25] * len(windows))
        assert not est.usable
        assert any("relaxation gate" in w for w in est.warnings)

    def test_ift_false_pass_regime_caught_by_relaxation_gate(self):
        """The F-0016 case: tau=1.0 fooled the IFT (45% bias behind IFT~1.01);
        the relaxation gate must catch it."""
        proto = _proto(1.0)
        windows = sample_quench_states(
            GAME, proto, n_trajectories=400, steps_per_unit_time=25, seed=2
        )
        est = hs_y_estimate(windows, n_states=4, hold_durations=[1.0] * len(windows))
        assert not est.usable

    def test_missing_hold_durations_unusable_with_warning(self):
        proto = _proto(8.0)
        windows = sample_quench_states(
            GAME, proto, n_trajectories=100, steps_per_unit_time=25, seed=2
        )
        est = hs_y_estimate(windows, n_states=4)
        assert not est.usable
        assert any("hold_durations" in w for w in est.warnings)


class TestValidation:
    def test_mismatched_windows_raise(self):
        try:
            hs_y_estimate([np.zeros((5, 3), dtype=int), np.zeros((4, 3), dtype=int)], n_states=4)
        except ValueError as e:
            assert "trajector" in str(e)
        else:
            raise AssertionError("mismatched trajectory counts must raise")

    def test_state_out_of_range_raises(self):
        w = [np.zeros((5, 3), dtype=int), np.full((5, 3), 9)]
        try:
            hs_y_estimate(w, n_states=4)
        except ValueError as e:
            assert "n_states" in str(e)
        else:
            raise AssertionError("out-of-range states must raise")
