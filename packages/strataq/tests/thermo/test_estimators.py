"""Trajectory irreversibility estimators (PROGRAMME v3 §3.5 estimators 1–2).

Ground truth is the exact Schnakenberg EPR of the same generator. The sharp
anchor: for the uniformised skeleton P = I + L/Λ, per-step entropy production
is exactly EPR/Λ, and the (k+1)-block KLD of a stationary Markov chain equals
k · (per-step EP) — so the KLD estimator converges to the exact meter for
every k, not just qualitatively.
"""

import jax
import jax.numpy as jnp
from strataq.core.defaults import base_config
from strataq.core.dynamics.markov import glauber_generator, stationary_distribution
from strataq.core.dynamics.sample import sample_trajectories, uniformized_chain
from strataq.finite.games.library import congestion, coordination, rock_paper_scissors
from strataq.thermo.estimators import (
    _reversed_codes,
    empirical_flux_weights,
    kld_epr,
    stationary_current_weights,
    tur_epr_bound,
    tur_epr_bound_ci,
    window_currents,
)
from strataq.thermo.exact import thermo_read

TOL = base_config().tolerances
LAM = 1.5
SEED = 20260810
N_STEPS = 60_000
N_TRAJ = 8


def _batch(game, key, *, n_steps=N_STEPS, n_traj=N_TRAJ):
    gen = glauber_generator(game, LAM)
    return gen, sample_trajectories(gen, key, n_steps=n_steps, n_trajectories=n_traj)


class TestSampler:
    def test_skeleton_is_stochastic(self):
        gen = glauber_generator(rock_paper_scissors(), LAM)
        kernel, rate = uniformized_chain(gen)
        assert float(rate) > 0.0
        assert jnp.max(jnp.abs(jnp.sum(kernel, axis=1) - 1.0)) < TOL.identity
        assert float(jnp.min(kernel)) >= 0.0

    def test_deterministic_under_seed(self):
        gen = glauber_generator(rock_paper_scissors(), LAM)
        a = sample_trajectories(gen, jax.random.key(SEED), n_steps=500, n_trajectories=2)
        b = sample_trajectories(gen, jax.random.key(SEED), n_steps=500, n_trajectories=2)
        assert jnp.array_equal(a.states, b.states)
        assert jnp.array_equal(a.dt, b.dt)

    def test_occupancy_matches_stationary(self):
        gen, batch = _batch(rock_paper_scissors(), jax.random.key(SEED))
        pi = stationary_distribution(gen)
        counts = jnp.bincount(batch.states.reshape(-1), length=pi.shape[0])
        occupancy = counts / jnp.sum(counts)
        assert 0.5 * float(jnp.sum(jnp.abs(occupancy - pi))) < 0.01

    def test_holding_times_match_uniformization_rate(self):
        gen, batch = _batch(rock_paper_scissors(), jax.random.key(SEED))
        _, rate = uniformized_chain(gen)
        assert abs(float(jnp.mean(batch.dt)) * float(rate) - 1.0) < 0.02


class TestKLD:
    def test_potential_game_reads_zero(self):
        _, batch = _batch(congestion(2, jnp.array([[1.0, 2.5], [1.5, 2.0]])), jax.random.key(SEED))
        assert float(kld_epr(batch, k=1)) < 5e-3

    def test_harmonic_game_recovers_exact_epr(self):
        game = rock_paper_scissors()
        _gen, batch = _batch(game, jax.random.key(SEED + 1))
        exact = float(thermo_read(game, LAM).epr)
        est = float(kld_epr(batch, k=1))
        assert exact > 1e-3
        assert abs(est - exact) / exact < 0.15

    def test_k2_agrees_with_k1_on_markov_data(self):
        _, batch = _batch(rock_paper_scissors(), jax.random.key(SEED + 2))
        k1, k2 = float(kld_epr(batch, k=1)), float(kld_epr(batch, k=2))
        assert abs(k2 - k1) / max(k1, 1e-12) < 0.25

    def test_data_starved_k_underestimates(self):
        """The documented failure mode: n ≪ n_states^(k+1) reads low, not high."""
        game = rock_paper_scissors()
        _, batch = _batch(game, jax.random.key(SEED + 7))
        exact = float(thermo_read(game, LAM).epr)
        assert float(kld_epr(batch, k=5)) < 0.75 * exact

    def test_reversed_codes_reverses_digits(self):
        n, block = 3, 3
        rev = _reversed_codes(n, block)
        for code in (0, 5, 14, 26):
            digits = [(code // n**j) % n for j in range(block)]
            expected = sum(d * n ** (block - 1 - j) for j, d in enumerate(digits))
            assert int(rev[code]) == expected


class TestTUR:
    def test_bound_is_below_exact_epr(self):
        game = rock_paper_scissors()
        gen, batch = _batch(game, jax.random.key(SEED + 3), n_traj=32)
        exact = float(thermo_read(game, LAM).epr)
        bound = float(tur_epr_bound(batch, stationary_current_weights(gen)))
        assert bound <= exact * 1.10  # certified bound + finite-sample fluctuation margin

    def test_bound_is_positive_on_harmonic(self):
        gen, batch = _batch(rock_paper_scissors(), jax.random.key(SEED + 4), n_traj=32)
        assert float(tur_epr_bound(batch, stationary_current_weights(gen))) > 1e-4

    def test_bound_near_zero_on_potential(self):
        gen, batch = _batch(coordination(2, 3), jax.random.key(SEED + 5), n_traj=32)
        assert float(tur_epr_bound(batch, stationary_current_weights(gen))) < 5e-3

    def test_data_driven_weights_on_held_out_split(self):
        """Weights from one batch, bound evaluated on an independent batch."""
        game = rock_paper_scissors()
        gen, weight_batch = _batch(game, jax.random.key(SEED + 6), n_traj=32)
        _, eval_batch = _batch(game, jax.random.key(SEED + 60), n_traj=32)
        exact = float(thermo_read(game, LAM).epr)
        w = empirical_flux_weights(weight_batch, n_states=gen.shape[0])
        bound = float(tur_epr_bound(eval_batch, w))
        assert bound <= exact * 1.10
        assert bound > 0.0

    def test_certified_quantile_below_exact(self):
        game = rock_paper_scissors()
        gen, batch = _batch(game, jax.random.key(SEED + 8), n_traj=32)
        exact = float(thermo_read(game, LAM).epr)
        w = stationary_current_weights(gen)
        ci_low = float(tur_epr_bound_ci(batch, w, jax.random.key(SEED + 9)))
        assert 0.0 < ci_low <= exact
        assert ci_low <= float(tur_epr_bound(batch, w))

    def test_window_currents_use_common_fixed_horizon(self):
        gen, batch = _batch(rock_paper_scissors(), jax.random.key(SEED + 10), n_traj=4)
        w = stationary_current_weights(gen)
        j, horizon = window_currents(batch, w)
        assert float(horizon) == float(jnp.min(jnp.sum(batch.dt, axis=1)))
        for m in range(4):
            cum = jnp.cumsum(batch.dt[m])
            steps = w[batch.states[m, :-1], batch.states[m, 1:]]
            manual = float(jnp.sum(jnp.where(cum <= horizon, steps, 0.0)))
            assert abs(float(j[m]) - manual) < 1e-12
