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
    def test_recovers_exact_mean_y_and_ift_passes(self):
        proto = _proto(8.0)
        windows = sample_quench_states(
            GAME, proto, n_trajectories=400, steps_per_unit_time=25, seed=1
        )
        est = hs_y_estimate(windows, n_states=4)
        exact = float(hatano_sasa_exact(GAME, proto)[1])
        assert est.usable
        assert est.ift_ci_low <= 1.0 <= est.ift_ci_high
        assert est.mean_y_ci_low <= exact <= est.mean_y_ci_high


class TestShortHoldSelfFlags:
    def test_short_windows_flag_unusable(self):
        proto = _proto(0.25)
        windows = sample_quench_states(
            GAME, proto, n_trajectories=400, steps_per_unit_time=25, seed=2
        )
        est = hs_y_estimate(windows, n_states=4)
        assert not est.usable
        assert any("window" in w or "IFT" in w for w in est.warnings)


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
