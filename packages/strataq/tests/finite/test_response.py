"""The instruments' contract: reads zero where it must, positive where it must,
and agrees with brute-force differentiation (PROGRAMME v3 §8.6 tests 3 and 5)."""

import jax
import jax.numpy as jnp
from strataq.core.defaults import base_config
from strataq.core.solve.fixedpoint import logit_qre
from strataq.core.types import BIFURCATION_NONE
from strataq.finite.games.library import (
    congestion,
    coordination,
    matching_pennies,
    rock_paper_scissors,
)
from strataq.finite.games.tensor import DenseTensorGame
from strataq.finite.response.reciprocity import reciprocity_defect
from strataq.finite.response.spectral import critical_lambda, strategic_spectrum
from strataq.finite.response.susceptibility import (
    build_operators,
    chi_equilibrium,
    chi_fd,
    chi_partial,
    choice_covariance,
)
from strataq.finite.response.tangent import block_basis, helmert_basis

TOL = base_config().tolerances
SOLVE = {"tol": 1e-13, "max_iter": 200_000}

COSTS = jnp.array([[1.0, 2.5, 4.0], [1.5, 2.0, 3.5], [0.5, 3.0, 5.0]])


class TestTangentBasis:
    def test_helmert_orthonormal_and_mean_zero(self):
        for m in (2, 3, 5, 8):
            q = helmert_basis(m)
            assert jnp.max(jnp.abs(q.T @ q - jnp.eye(m - 1))) < TOL.identity
            assert jnp.max(jnp.abs(jnp.ones(m) @ q)) < TOL.identity

    def test_block_basis_shape(self):
        q = block_basis((3, 4))
        assert q.shape == (7, 5)
        assert jnp.max(jnp.abs(q.T @ q - jnp.eye(5))) < TOL.identity


class TestChiPartial:
    def test_chi_partial_is_lambda_c(self):
        """K7: χ^part blocks are exactly λ_i C_i."""
        game = rock_paper_scissors()
        point = logit_qre(game, 1.4, **SOLVE)
        chi = chi_partial(point)
        for i, off in enumerate((0, 3)):
            block = chi[off : off + 3, off : off + 3]
            expected = point.lam[i] * choice_covariance(point.sigma[i])
            assert jnp.max(jnp.abs(block - expected)) < TOL.identity

    def test_c_rows_sum_zero(self):
        sigma = jnp.array([0.2, 0.5, 0.3])
        c = choice_covariance(sigma)
        assert jnp.max(jnp.abs(jnp.sum(c, axis=0))) < TOL.identity
        assert jnp.max(jnp.abs(jnp.sum(c, axis=1))) < TOL.identity


class TestReciprocity:
    def test_reads_zero_on_potential_games(self):
        """Milestone half 1: ℛ < 1e-10 on verified exact potential games."""
        for game in (
            congestion(2, COSTS),
            congestion(3, COSTS),
            coordination(2, 3, bonus=2.0),
            coordination(3, 2, bonus=1.5, mismatch=-0.5),
        ):
            point = logit_qre(game, 1.0, **SOLVE)
            r = reciprocity_defect(game, point)
            assert float(r) < 1e-10, f"R={float(r)} on a potential game"

    def test_reads_positive_on_harmonic_games(self):
        """Milestone half 2: ℛ well above zero on RPS-family games."""
        for game, lam in (
            (rock_paper_scissors(), 1.5),
            (rock_paper_scissors(5), 1.0),
            (matching_pennies(), 1.2),
        ):
            point = logit_qre(game, lam, **SOLVE)
            r = float(reciprocity_defect(game, point))
            assert r > 0.1, f"R={r} on a harmonic game"

    def test_b_asymmetry_iff_positive_r(self):
        """Result 2 mechanism: symmetric B_T ⟺ ℛ = 0 at the same point."""
        game = congestion(2, COSTS)
        point = logit_qre(game, 1.1, **SOLVE)
        ops = build_operators(game, point)
        b_t = ops.b_tangent
        assert jnp.max(jnp.abs(b_t - b_t.T)) < 1e-10

        harm = rock_paper_scissors()
        h_point = logit_qre(harm, 1.1, **SOLVE)
        h_ops = build_operators(harm, h_point)
        h_b = h_ops.b_tangent
        assert jnp.max(jnp.abs(h_b - h_b.T)) > 0.1


class TestChiEquilibrium:
    def test_matches_finite_differences(self):
        """Test 5 / oracle_agreement: χ^eq vs central differences to 1e-6."""
        key = jax.random.PRNGKey(base_config().seeds.root + 10)
        k1, k2 = jax.random.split(key)
        game = DenseTensorGame(
            (0.5 * jax.random.normal(k1, (3, 3)), 0.5 * jax.random.normal(k2, (3, 3)))
        )
        lam = 0.9
        point = logit_qre(game, lam, **SOLVE)
        resp = chi_equilibrium(game, point)
        fd = chi_fd(game, lam)
        assert jnp.max(jnp.abs(resp.chi_full - fd)) < TOL.fd

    def test_symmetric_on_potential_asymmetric_on_harmonic(self):
        game = congestion(2, COSTS)
        point = logit_qre(game, 1.0, **SOLVE)
        chi = chi_equilibrium(game, point).chi_tangent
        assert jnp.max(jnp.abs(chi - chi.T)) < 1e-10

        harm = rock_paper_scissors()
        h_point = logit_qre(harm, 1.0, **SOLVE)
        h_chi = chi_equilibrium(harm, h_point).chi_tangent
        assert jnp.max(jnp.abs(h_chi - h_chi.T)) > 0.01


class TestSpectrum:
    def test_potential_game_spectrum_is_real(self):
        """N3 numerical leg: B = Bᵀ ⟹ S_T B_T similar to symmetric ⟹ real spectrum."""
        for game in (congestion(2, COSTS), coordination(3, 2, bonus=1.5)):
            point = logit_qre(game, 1.3, **SOLVE)
            spec = strategic_spectrum(game, point)
            rel_imag = jnp.max(jnp.abs(jnp.imag(spec.eigenvalues))) / jnp.maximum(
                jnp.max(jnp.abs(spec.eigenvalues)), 1e-30
            )
            assert float(rel_imag) < 1e-8

    def test_harmonic_game_has_complex_spectrum(self):
        game = rock_paper_scissors()
        point = logit_qre(game, 2.0, **SOLVE)
        spec = strategic_spectrum(game, point)
        assert float(jnp.max(jnp.abs(jnp.imag(spec.eigenvalues)))) > 0.01

    def test_small_lambda_is_subcritical_and_untyped(self):
        game = coordination(2, 2, bonus=1.0)
        point = logit_qre(game, 0.1, **SOLVE)
        spec = strategic_spectrum(game, point)
        assert float(spec.rho) < 1.0
        assert int(spec.bifurcation_type) == BIFURCATION_NONE
        assert not bool(spec.near_critical)

    def test_critical_lambda_brackets_coordination_transition(self):
        """2×2 coordination has the Brock–Durlauf pitchfork; critical λ is finite
        and ρ crosses 1 there along the principal branch."""
        game = coordination(2, 2, bonus=1.0)
        lam_c = critical_lambda(game, bracket=(0.1, 50.0), tol=1e-4)
        assert 0.5 < lam_c < 50.0

    def test_near_critical_flag_fires(self):
        game = coordination(2, 2, bonus=1.0)
        lam_c = critical_lambda(game, bracket=(0.1, 50.0), tol=1e-6)
        point = logit_qre(game, lam_c * 0.9999, **SOLVE)
        resp = chi_equilibrium(game, point)
        # At 0.01% below the critical λ the distance is far inside the warn band.
        assert bool(resp.near_critical) or float(resp.distance_to_criticality) < 0.01
