"""Electricity domain: CAISO loader + irreversibility plumbing (unit domains.electricity)."""

from datetime import date

from strataq.core.dynamics.sample import trajectory_from_series
from strataq.domains.electricity import discretize_quantile, fetch_dam_lmp
from strataq.thermo.estimators import kld_epr


class TestDiscretize:
    def test_quantile_bins_cover_all_states(self):
        prices = [float(x) for x in range(100)]
        states, edges = discretize_quantile(prices, 5)
        assert set(states) == {0, 1, 2, 3, 4}
        assert len(edges) == 4

    def test_deterministic(self):
        prices = [3.0, 1.0, 4.0, 1.5, 9.0, 2.6, 5.0, 3.5]
        a, _ = discretize_quantile(prices, 3)
        b, _ = discretize_quantile(prices, 3)
        assert a == b


class TestSeriesTrajectory:
    def test_wraps_shapes(self):
        batch = trajectory_from_series([0, 1, 2, 1, 0, 2, 1], 3, dt=1.0)
        assert batch.states.shape == (1, 7)
        assert batch.dt.shape == (1, 6)
        assert batch.n_states == 3

    def test_iid_series_reads_near_zero(self):
        import numpy as np

        rng = np.random.default_rng(0)
        seq = rng.integers(0, 4, 20_000)
        batch = trajectory_from_series(seq, 4)
        assert float(kld_epr(batch, k=1)) < 5e-3

    def test_noisy_cycle_reads_loud(self):
        """A driven (mostly-forward) cycle carries measurable irreversibility.

        A PURE deterministic cycle reads 0 — its reversed blocks have zero
        probability and the plug-in truncates them (documented lower-bound
        behaviour) — so the drive here is 80% forward / 20% backward, whose
        per-step EP is 0.6·ln4 ≈ 0.83 nats."""
        import numpy as np

        rng = np.random.default_rng(1)
        s, seq = 0, []
        for _ in range(20_000):
            seq.append(s)
            s = (s + 1) % 4 if rng.random() < 0.8 else (s - 1) % 4
        batch = trajectory_from_series(seq, 4)
        est = float(kld_epr(batch, k=1))
        assert abs(est - 0.6 * np.log(4)) < 0.05  # matches the analytic EP


class TestLiveCAISO:
    def test_fetch_two_days(self):
        ts, prices = fetch_dam_lmp(date(2026, 8, 1), 2)
        assert len(prices) >= 44  # >= 90% of 48 hours
        assert all(-500 < p < 5000 for p in prices)
        assert ts == sorted(ts)


class TestPhaseEmbed:
    def test_blindness_and_sight(self):
        """A clean periodic loop: value-space KLD reads ~0 (blind), the
        phase-space embedding reads loud (the loop no longer retraces)."""
        import math

        import numpy as np
        from strataq.domains.electricity import phase_embed

        rng = np.random.default_rng(3)
        prices = [
            math.sin(2 * math.pi * t / 24) + 0.25 * float(rng.standard_normal())
            for t in range(24 * 400)
        ]
        vstates, _ = discretize_quantile(prices, 4)
        v = float(kld_epr(trajectory_from_series(vstates, 4), k=1))
        estates, n = phase_embed(prices, 4)
        e = float(kld_epr(trajectory_from_series(estates, n), k=1))
        assert v < 5e-3  # value space: blind to the loop
        assert e > 0.02  # phase space: sees it
        assert e > 100 * v
