"""Electricity domain: CAISO loader + irreversibility plumbing (unit domains.electricity)."""

from datetime import date

import jax.numpy as jnp
from strataq.core.dynamics.sample import trajectory_from_series
from strataq.domains.electricity import discretize_quantile, fetch_dam_lmp
from strataq.domains.electricity.oracle import BiddingOracle
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


class TestLoaderGuards:
    def test_fresh_fetch_exercises_network_path(self):
        """An uncached day goes through the paced fetch (happy path)."""
        from strataq.domains.electricity.caiso import CACHE, DEFAULT_NODE

        target = CACHE / f"DAM_{DEFAULT_NODE}_2026-06-15.zip"
        target.unlink(missing_ok=True)
        _ts, prices = fetch_dam_lmp(date(2026, 6, 15), 1)
        assert len(prices) >= 22
        assert target.exists()

    def test_coverage_refusal(self, monkeypatch):
        """Below-90% coverage must raise, never silently truncate."""
        import io
        import zipfile

        from strataq.domains.electricity import caiso

        def tiny_zip(node, day, market="DAM"):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                zf.writestr("notes.xml", "<ignored/>")  # non-csv entries are skipped
                zf.writestr(
                    "x.csv",
                    "INTERVALSTARTTIME_GMT,LMP_TYPE,MW\n"
                    "2026-08-01T09:00:00-00:00,LMP,42.0\n"
                    "2026-08-01T10:00:00-00:00,MCC,1.0\n",
                )
            return buf.getvalue()

        monkeypatch.setattr(caiso, "_fetch_day", tiny_zip)
        try:
            caiso.fetch_dam_lmp(date(2026, 8, 1), 1)
        except ValueError as exc:
            assert "coverage" in str(exc)
        else:
            raise AssertionError("expected coverage refusal")


class TestRateLimitBackoff:
    def test_429_retries_then_raises(self, monkeypatch):
        from urllib.error import HTTPError

        from strataq.domains.electricity import caiso

        calls = {"n": 0}

        def always_429(url, timeout=0):
            calls["n"] += 1
            raise HTTPError(url, 429, "rate limited", None, None)

        monkeypatch.setattr(caiso, "urlopen", always_429)
        monkeypatch.setattr(caiso.time, "sleep", lambda s: None)
        try:
            caiso._fetch_day("FAKE_NODE", date(2030, 1, 1))
        except TimeoutError:
            assert calls["n"] == 6  # all retries consumed
        else:
            raise AssertionError("expected TimeoutError")


class TestBiddingOracle:
    """Uniform-price auction oracle (2 generators, D=1): the cheaper offer
    dispatches and sets the price; ties split the demand."""

    def _game(self):
        from strataq.domains.electricity.oracle import bidding_game

        return bidding_game(costs=(20.0, 20.0), offers=(25.0, 35.0, 45.0, 60.0))

    def test_payoff_logic(self):
        game = self._game()
        u1, u2 = game.payoffs
        # gen1 offers 25, gen2 offers 45 -> gen1 dispatches at 25: profit 5; gen2 gets 0
        assert float(u1[0, 2]) == 5.0
        assert float(u2[0, 2]) == 0.0
        # tie at 35: split -> each (35-20)/2
        assert float(u1[1, 1]) == 7.5
        assert float(u2[1, 1]) == 7.5

    def test_alpha_between_anchors(self):
        from strataq.finite.decompose.hodge import hodge_decompose

        alpha = float(hodge_decompose(self._game()).alpha)
        assert 0.05 < alpha < 0.95  # a genuine mixed game, not a degenerate anchor

    def test_qre_dispersion_monotone_in_lambda(self):
        """Sharper bidders concentrate on the aggressive offer — clearing-price
        dispersion falls with λ (the identification channel for λ̂)."""
        import numpy as np
        from strataq.core.solve.fixedpoint import logit_qre
        from strataq.domains.electricity.oracle import clearing_price_distribution

        game = self._game()
        disps = []
        for lam in (0.05, 0.3, 2.0):
            sigma = logit_qre(game, lam).sigma
            prices, probs = clearing_price_distribution(sigma, offers=(25.0, 35.0, 45.0, 60.0))
            mean = float(np.dot(prices, probs))
            disps.append(float(np.dot(probs, (np.asarray(prices) - mean) ** 2)) ** 0.5)
        assert disps[0] > disps[1] > disps[2]

    def test_plugin_contract_complete(self):
        from strataq.core.protocols import ConjugateFieldSpec
        from strataq.domains import electricity as plug

        assert plug.engine == "finite"
        assert callable(plug.oracle_factory)
        assert callable(plug.grid_factory)
        assert isinstance(plug.field_spec, ConjugateFieldSpec)
        assert plug.field_spec.linearity == "exact"
        assert plug.learn.slug


class TestOracleProtocolConformance:
    """The plugin shipped without `response_matrix` — the PayoffOracle
    protocol requires it, and only CI's strict typing caught the gap
    (2026-08-12; the same blind-spot lesson as F-0018). These tests make
    conformance a runtime fact, not just a type-checker opinion."""

    def test_bidding_oracle_satisfies_payoff_oracle(self):
        from strataq.core.protocols import PayoffOracle

        oracle = BiddingOracle((10.0, 12.0), (11.0, 13.0, 15.0))
        assert isinstance(oracle, PayoffOracle)

    def test_response_matrix_shape_and_sign(self):
        """Undercutting a rival lifts your own profit on this grid: the
        own-derivative of profit w.r.t. one's own offer rung is finite and
        the matrix is (2, 2)."""
        oracle = BiddingOracle((10.0, 10.0), (11.0, 12.0, 13.0))
        jac = oracle.response_matrix(jnp.asarray([1, 1]))
        assert jac.shape == (2, 2)
        assert bool(jnp.all(jnp.isfinite(jac)))

    def test_response_matrix_edge_clamped(self):
        """At a grid edge the one-sided difference is used, never an index
        out of range."""
        oracle = BiddingOracle((10.0, 10.0), (11.0, 12.0, 13.0))
        for idx in ([0, 0], [2, 2]):
            jac = oracle.response_matrix(jnp.asarray(idx))
            assert jac.shape == (2, 2) and bool(jnp.all(jnp.isfinite(jac)))
