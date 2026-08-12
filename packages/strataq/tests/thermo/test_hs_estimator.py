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
from strataq.thermo.hs_estimator import (
    hs_y_estimate,
    relaxation_gate,
    sample_quench_states,
)
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


class TestRelaxationGateSE:
    """R9 (unit thermo.hs_estimator.gate_se): the relaxation gate's SE must
    not depend on the ARBITRARY ORDER of the trajectories.

    R8/F-0019 localised the small-n flag instability to the incumbent 4-way
    ``i::4`` trajectory split: a permutation that cannot change any physical
    property reshuffles which trajectories share a split group, so the SE —
    and with it the ``tau_hat + 2 SE`` threshold — moves. Criteria G1-G5 are
    registered in config/experiments/gate_se.yaml.
    """

    METHODS = ("split", "jackknife", "delta", "bootstrap")

    @staticmethod
    def _windows(n: int, tau: float = 32.0, seed: int = 7) -> list[np.ndarray]:
        return sample_quench_states(
            GAME, _proto(tau), n_trajectories=n, steps_per_unit_time=25, seed=seed
        )

    def _gate(self, w, method, tau=32.0, **kw):
        return relaxation_gate(
            w,
            n_states=4,
            hold_durations=[tau] * len(w),
            se_method=method,
            **kw,
        )

    def test_point_estimate_identical_across_methods(self):
        """The SE method must change ONLY the SE. If a candidate moves
        tau_hat itself it is a different estimator, not a different error
        bar, and every downstream comparison would be confounded."""
        w = self._windows(60)
        ref = self._gate(w, "split").tau_hats
        for method in self.METHODS:
            got = self._gate(w, method).tau_hats
            assert got == ref, (method, got, ref)

    def test_jackknife_and_delta_are_exactly_order_invariant(self):
        """G1 at its strongest: not 'approximately stable across seeds' but
        INVARIANT to the permutation, to floating-point tolerance."""
        w = self._windows(50)
        perm = np.random.default_rng(0).permutation(50)
        for method in ("jackknife", "delta"):
            base = self._gate(w, method)
            shuf = self._gate([x[perm] for x in w], method)
            for t0, t1 in zip(base.ses, shuf.ses, strict=True):
                assert abs(t1 - t0) <= 1e-9 * max(abs(t0), 1e-12), (method, t0, t1)
            assert base.ok == shuf.ok

    def test_incumbent_split_se_is_order_dependent(self):
        """The R8 diagnosis as an executable characterisation: the incumbent
        SE MOVES under the physically-null permutation. This test documents
        the defect R9 exists to fix — if it ever starts passing trivially
        (identical SEs), the split implementation changed underneath us."""
        w = self._windows(50)
        perm = np.random.default_rng(1).permutation(50)
        base = self._gate(w, "split")
        shuf = self._gate([x[perm] for x in w], "split")
        rel = [abs(b - s) / max(abs(b), 1e-12) for b, s in zip(base.ses, shuf.ses, strict=True)]
        assert max(rel) > 0.01, rel  # measured: O(10-100%) movement

    def test_jackknife_closed_form_equals_brute_force_leave_one_out(self):
        """The O(nT) sufficient-statistic jackknife must equal the O(n^2 T)
        definition. This is the algebra's only real check: a wrong
        leave-one-out would still be order-invariant and would still look
        plausible, so G1 could not catch it."""
        n = 24
        w = self._windows(n, seed=9)
        fast = self._gate(w, "jackknife").ses
        for wi, se_fast in zip(w, fast, strict=True):
            dt = 32.0 / wi.shape[1]
            reps = [
                relaxation_gate(
                    [np.delete(wi, j, axis=0)],
                    n_states=4,
                    hold_durations=[32.0],
                    se_method="delta",  # any method: only tau_hat is read
                ).tau_hats[0]
                for j in range(n)
            ]
            arr = np.asarray(reps)
            se_slow = float(np.sqrt((n - 1) / n * np.sum((arr - arr.mean()) ** 2)))
            assert abs(se_fast - se_slow) <= 1e-8 * max(se_slow, 1e-12), (
                se_fast,
                se_slow,
                dt,
            )

    def test_every_method_still_refuses_unsettled_holds(self):
        """G4 in miniature, as a unit test rather than only a campaign
        metric: a candidate that passes G1-G3 by shrinking the SE toward
        zero must still REFUSE a hold that never settled."""
        for method in self.METHODS:
            w = self._windows(50, tau=1.0, seed=3)
            gate = self._gate(w, method, tau=1.0)
            assert not gate.ok, method
            assert gate.offenders, method

    def test_unknown_se_method_raises(self):
        w = self._windows(20)
        try:
            self._gate(w, "vibes")
        except ValueError as e:
            assert "se_method" in str(e)
        else:
            raise AssertionError("unknown se_method must raise")

    def test_tiny_n_falls_back_to_maximally_cautious_se(self):
        """With too few trajectories to estimate a spread at all, the SE must
        equal the estimate itself (the incumbent's rule, kept): the gate then
        demands 3x the hold and refuses rather than guessing."""
        w = self._windows(3)
        for method in self.METHODS:
            gate = self._gate(w, method)
            for tau_hat, se in zip(gate.tau_hats, gate.ses, strict=True):
                assert se == tau_hat, (method, tau_hat, se)

    def test_hs_y_estimate_threads_the_se_method(self):
        w = self._windows(60)
        kw = {"n_states": 4, "hold_durations": [32.0] * len(w)}
        base = hs_y_estimate(w, **kw)
        jack = hs_y_estimate(w, relax_se_method="jackknife", **kw)
        assert base.mean_y == jack.mean_y  # the SE method cannot move <Y>
        assert isinstance(jack.usable, bool)


class TestIntervalOrderDependence:
    """R9 found a SECOND order-dependence, outside everything it fixed.

    `hs_y_estimate`'s CI/IFT bootstrap draws indices from a fixed-seed RNG
    (`default_rng(0)`), so permuting the trajectories changes which ones land
    in each resample. The point estimate is unaffected — it is a mean — but
    the interval moves, and the ANOMALY flag is a hard boolean thresholded on
    that interval ("does the CI exclude 1?"), so a physically-null permutation
    can flip it. Measured on real CAISO month-05 data: mean_y = 6.152647 to
    six decimals under every permutation while the IFT upper bound swung
    0.970..1.246 and the flag flipped with it (F-0020).

    This CHARACTERISES the open defect rather than asserting the fix: a
    thresholded Monte-Carlo statistic cannot be stable when the bound sits at
    the threshold, so reseeding alone is not the answer (that is R10). When
    R10 lands, this test must fail and be updated deliberately.
    """

    def test_mean_is_order_invariant_but_the_interval_is_not(self):
        proto = _proto(32.0)
        windows = sample_quench_states(
            GAME, proto, n_trajectories=40, steps_per_unit_time=25, seed=21
        )
        kw = {"n_states": 4, "hold_durations": [32.0] * len(windows)}
        base = hs_y_estimate(windows, **kw)
        rng = np.random.default_rng(5)
        widths = []
        for _ in range(4):
            perm = rng.permutation(windows[0].shape[0])
            shuf = hs_y_estimate([w[perm] for w in windows], **kw)
            # the ESTIMATE is a mean over trajectories: exactly invariant
            assert abs(shuf.mean_y - base.mean_y) < 1e-12
            widths.append(shuf.ift_ci_high - shuf.ift_ci_low)
        spread = (max(widths) - min(widths)) / max(min(widths), 1e-12)
        assert spread > 1e-6, widths  # the INTERVAL moves — the open defect
