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
    def test_unbiased_across_seeds_after_lambda0_fix(self):
        """The missing-lambda_0-window bug made every estimate ~35% low; with
        the pre-quench window the estimator is UNBIASED (mean of means within
        the seed spread of exact). Per-seed CI coverage is NOT asserted —
        the bootstrap still under-covers ~1.7x (open in F-0016) and the
        module is banner-marked experimental."""
        import numpy as _np

        proto = _proto(24.0)
        exact = float(hatano_sasa_exact(GAME, proto)[1])
        means = []
        for seed in range(4):
            windows = sample_quench_states(
                GAME, proto, n_trajectories=400, steps_per_unit_time=25, seed=100 + seed
            )
            est = hs_y_estimate(windows, n_states=4, hold_durations=[24.0] * len(windows))
            means.append(est.mean_y)
        assert abs(float(_np.mean(means)) - exact) < 0.03


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


class TestAnomalyDetector:
    def test_ift_fires_on_continuous_ramp(self):
        """F1 (fresh red-team): the excludes-1 companion must CATCH a
        genuinely non-stepwise system — a continuous λ ramp chopped into
        fake holds — or it is decorative. Measured: CI [1.03, 1.10]."""
        import jax
        from jax.scipy.linalg import expm
        from strataq.core.dynamics.markov import glauber_generator, stationary_distribution

        key = jax.random.PRNGKey(3)
        n_traj, dt, total = 400, 0.1, 160.0
        n_steps = int(total / dt)
        pi0 = stationary_distribution(glauber_generator(GAME, 0.5))
        key, k0 = jax.random.split(key)
        states = jax.random.categorical(k0, jnp.log(pi0), shape=(n_traj,))
        seq = np.empty((n_traj, n_steps), dtype=np.int64)
        for t in range(n_steps):
            lam = 0.5 + 5.0 * (t / n_steps)
            kern = expm(dt * glauber_generator(GAME, lam))
            key, ks = jax.random.split(key)
            states = jax.random.categorical(ks, jnp.log(jnp.maximum(kern, 1e-300))[states])
            seq[:, t] = np.asarray(states)
        chunk = n_steps // 5
        windows = [seq[:, i * chunk : (i + 1) * chunk] for i in range(5)]
        est = hs_y_estimate(windows, n_states=4, hold_durations=[total / 5] * 5)
        assert not est.usable
        assert not (est.ift_ci_low <= 1.0 <= est.ift_ci_high)  # the CI EXCLUDES 1
        assert any("ANOMALY" in w for w in est.warnings)


class TestIntervalMethods:
    """R8: the interval method becomes selectable so small-n coverage can be
    certified (or refused) on evidence rather than by preference. The
    percentile bootstrap is the incumbent; bootstrap-t studentises each
    resample; t_widened scales the percentile half-width by the Student-t
    quantile for n-1 df (the small-n correction)."""

    def test_all_methods_agree_at_large_n(self):
        """At n=400 the three intervals must be close — a method that moves
        the answer where the CLT already holds is suspect."""
        proto = _proto(32.0)
        windows = sample_quench_states(
            GAME, proto, n_trajectories=400, steps_per_unit_time=25, seed=11
        )
        widths = {}
        for method in ("percentile", "bootstrap_t", "t_widened"):
            est = hs_y_estimate(
                windows,
                n_states=4,
                hold_durations=[32.0] * len(windows),
                interval_method=method,
            )
            widths[method] = est.mean_y_ci_high - est.mean_y_ci_low
        base = widths["percentile"]
        for method, w in widths.items():
            assert 0.5 * base <= w <= 2.0 * base, (method, widths)

    def test_t_widened_is_wider_at_small_n(self):
        """The correction must actually widen where it is meant to."""
        proto = _proto(32.0)
        windows = sample_quench_states(
            GAME, proto, n_trajectories=25, steps_per_unit_time=25, seed=12
        )
        kw = {"n_states": 4, "hold_durations": [32.0] * len(windows)}
        pct = hs_y_estimate(windows, interval_method="percentile", **kw)
        wide = hs_y_estimate(windows, interval_method="t_widened", **kw)
        assert (wide.mean_y_ci_high - wide.mean_y_ci_low) > (pct.mean_y_ci_high - pct.mean_y_ci_low)

    def test_unknown_method_raises(self):
        proto = _proto(32.0)
        windows = sample_quench_states(
            GAME, proto, n_trajectories=25, steps_per_unit_time=25, seed=13
        )
        try:
            hs_y_estimate(
                windows,
                n_states=4,
                hold_durations=[32.0] * len(windows),
                interval_method="wishful",
            )
        except ValueError as e:
            assert "interval_method" in str(e)
        else:
            raise AssertionError("unknown interval method must raise")

    def test_boot_se_exposed_and_width_comparable(self):
        """R8 objection 2: callers must be able to guard interval
        informativeness, so the bootstrap SE is part of the read."""
        proto = _proto(32.0)
        windows = sample_quench_states(
            GAME, proto, n_trajectories=60, steps_per_unit_time=25, seed=14
        )
        est = hs_y_estimate(windows, n_states=4, hold_durations=[32.0] * len(windows))
        assert est.boot_se > 0.0
        half = 0.5 * (est.mean_y_ci_high - est.mean_y_ci_low)
        assert half <= 4.0 * est.boot_se  # sane percentile interval

    def test_t_widened_warns_about_being_heuristic(self):
        proto = _proto(32.0)
        windows = sample_quench_states(
            GAME, proto, n_trajectories=25, steps_per_unit_time=25, seed=15
        )
        est = hs_y_estimate(
            windows,
            n_states=4,
            hold_durations=[32.0] * len(windows),
            interval_method="t_widened",
        )
        assert any("heuristic" in w for w in est.warnings)
