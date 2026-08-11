"""The plain-data facade (unit product.toolkit): a stranger's five minutes.

Every test here uses ONLY plain lists/numpy — the way a user who has never
seen DenseTensorGame would call the library — and checks both the numbers
and the honesty warnings.
"""

import jax
import jax.numpy as jnp
import numpy as np
import strataq.toolkit as tk
from strataq.core.dynamics.markov import glauber_generator
from strataq.core.dynamics.sample import sample_trajectories
from strataq.estimate.lam import sample_choices
from strataq.finite.games.library import matching_pennies
from strataq.finite.games.tensor import DenseTensorGame

RPS_U1 = [[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]]
RPS_U2 = [[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]]
ASYM_U1 = [[3.0, 0.0, 1.5], [1.0, 2.0, 0.5], [0.0, 1.0, 2.5]]
ASYM_U2 = [[2.0, 1.0, 0.0], [0.5, 3.0, 1.0], [1.5, 0.0, 2.0]]


class TestGameThermo:
    def test_rps_reads_whirlpool(self):
        read = tk.game_thermo([RPS_U1, RPS_U2])
        assert read.alpha > 0.9
        assert read.epr > 1e-3
        assert "whirlpool" in read.verdict

    def test_coordination_reads_landscape(self):
        bonus = [[2.0, 0.0], [0.0, 2.0]]
        read = tk.game_thermo([bonus, bonus])
        assert read.alpha < 0.05
        assert read.epr < 1e-9
        assert "landscape" in read.verdict


class TestReciprocityRead:
    def test_symmetric_matrix_is_reciprocal(self):
        read = tk.reciprocity_read([[1.0, 0.2], [0.2, 0.9]])
        assert read.r < 1e-12
        assert "reciprocal" in read.verdict
        assert read.warnings  # honesty text always present
        assert any("point read" in w for w in read.warnings)  # no chi_se -> flagged

    def test_empirical_f0011_matrix(self):
        """The actual F-0011 pass-through matrix reads as measured."""
        read = tk.reciprocity_read([[1.0697, 0.0028], [0.0005, 0.9685]])
        assert abs(read.r - 0.0011) < 3e-4
        assert "reciprocal" in read.verdict

    def test_antisymmetric_matrix_is_whirlpool(self):
        read = tk.reciprocity_read([[0.0, 1.0], [-1.0, 0.0]])
        assert read.r > 10.0  # denominator ~ 0: strongly non-reciprocal
        assert "non-reciprocal" in read.verdict

    def test_chi_se_gives_interval_and_uncertain_verdict(self):
        """Red-team B2: a noisy near-threshold matrix must NOT be classified."""
        chi = [[1.0, 0.05], [0.01, 1.0]]  # R ~ 0.02, on the threshold
        se = [[0.02, 0.02], [0.02, 0.02]]
        read = tk.reciprocity_read(chi, chi_se=se, seed=1)
        assert read.ci_low is not None and read.ci_high is not None
        assert read.ci_low < read.r < read.ci_high
        assert "uncertain" in read.verdict or "CI" in read.verdict

    def test_borderline_point_read_says_so(self):
        read = tk.reciprocity_read([[1.0, 0.05], [0.01, 1.0]])
        assert "borderline" in read.verdict

    def test_nan_chi_raises(self):
        try:
            tk.reciprocity_read([[1.0, float("nan")], [0.2, 0.9]])
        except ValueError as e:
            assert "NaN" in str(e)
        else:
            raise AssertionError("NaN chi must raise")


class TestEstimateRationality:
    def test_recovers_lambda_from_plain_lists(self):
        counts = sample_choices(
            DenseTensorGame((jnp.array(ASYM_U1), jnp.array(ASYM_U2))),
            1.8,
            3000,
            jax.random.PRNGKey(3),
        )
        est = tk.estimate_rationality([ASYM_U1, ASYM_U2], [np.asarray(c).tolist() for c in counts])
        assert est.ci_low <= 1.8 <= est.ci_high or abs(est.mean - 1.8) / 1.8 < 0.1
        assert any("payoff unit" in w for w in est.warnings)  # scale-fold warning always on

    def test_flat_likelihood_warns(self):
        """Uniform counts on symmetric matching pennies: lambda unidentified."""
        est = tk.estimate_rationality(
            [[[1.0, -1.0], [-1.0, 1.0]], [[-1.0, 1.0], [1.0, -1.0]]],
            [[500, 500], [500, 500]],
        )
        assert any("flat likelihood" in w for w in est.warnings)


class TestIrreversibilityTest:
    def test_driven_series_detected(self):
        """A sampled trajectory from a dissipative chain, read back as values."""
        gen = glauber_generator(matching_pennies(), 2.0)
        batch = sample_trajectories(gen, jax.random.PRNGKey(0), n_steps=1200, n_trajectories=1)
        # map the 4 joint states to a scalar 'price' that exposes the cycle
        level = np.array([0.0, 1.0, 3.0, 2.0])
        series = level[np.asarray(batch.states[0])]
        verdict = tk.irreversibility_test(series, n_bins=2, n_surrogates=100, seed=1)
        assert verdict.detected
        assert verdict.p_value < 0.02

    def test_reversible_series_at_null(self):
        rng = np.random.default_rng(0)
        series = np.cumsum(rng.normal(size=800))  # random walk: reversible
        verdict = tk.irreversibility_test(series, n_bins=3, n_surrogates=100, seed=1)
        assert not verdict.detected
        assert verdict.p_value > 0.05

    def test_too_short_raises(self):
        try:
            tk.irreversibility_test([1.0, 2.0, 3.0])
        except ValueError as e:
            assert "50" in str(e)
        else:
            raise AssertionError("short series must raise")

    def test_nan_series_raises(self):
        series = [1.0, 2.0] * 50
        series[10] = float("nan")
        try:
            tk.irreversibility_test(series)
        except ValueError as e:
            assert "NaN" in str(e)
        else:
            raise AssertionError("NaN series must raise")

    def test_constant_series_raises(self):
        try:
            tk.irreversibility_test([5.0] * 200)
        except ValueError as e:
            assert "constant" in str(e)
        else:
            raise AssertionError("constant series must raise, not return a vacuous verdict")


class TestInputValidation:
    def test_nan_payoff_raises(self):
        try:
            tk.estimate_rationality(
                [[[1.0, float("nan")], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
                [[10, 20], [15, 15]],
            )
        except ValueError as e:
            assert "NaN" in str(e)
        else:
            raise AssertionError("NaN payoffs must raise")

    def test_wrong_length_counts_raise(self):
        try:
            tk.estimate_rationality([ASYM_U1, ASYM_U2], [[10, 20], [15, 15, 12]])
        except ValueError as e:
            assert "length-3" in str(e)
        else:
            raise AssertionError("mis-shaped counts must raise")
