"""Branch tracer: agreement with the fixed-λ solver, fold handling, gambit cross-check."""

import jax.numpy as jnp
import pytest
from strataq.core.solve.fixedpoint import logit_qre
from strataq.core.solve.homotopy import logit_branch
from strataq.finite.games.library import coordination, matching_pennies, rock_paper_scissors
from strataq.finite.games.tensor import DenseTensorGame


class TestBranch:
    def test_matches_fixed_lambda_solver_on_unique_branch(self):
        """Where the QRE is unique, the branch must pass through the fixed-λ points."""
        game = matching_pennies()
        branch = logit_branch(game, 3.0, n_points=400)
        assert float(branch.lambdas[-1]) >= 3.0
        for probe in (0.5, 1.5, 2.5):
            idx = int(jnp.argmin(jnp.abs(branch.lambdas - probe)))
            lam_at = float(branch.lambdas[idx])
            direct = logit_qre(game, lam_at, tol=1e-13, max_iter=200_000)
            gap = float(jnp.max(jnp.abs(branch.sigmas[idx] - jnp.concatenate(direct.sigma))))
            assert gap < 1e-8, f"probe {probe}: gap {gap}"

    def test_starts_at_centroid(self):
        branch = logit_branch(rock_paper_scissors(), 1.0, n_points=100)
        assert float(branch.lambdas[0]) < 0.1
        assert jnp.max(jnp.abs(branch.sigmas[0] - 1.0 / 3.0)) < 0.05

    def test_coordination_branch_passes_through_criticality(self):
        """2x2 coordination: the principal branch crosses rho = 1 (Brock–Durlauf
        pitchfork territory) and the tracer keeps going rather than stalling."""
        game = coordination(2, 2, bonus=1.0)
        branch = logit_branch(game, 12.0, n_points=600)
        assert float(jnp.max(branch.rhos)) > 0.99
        assert float(branch.lambdas[-1]) >= 12.0

    def test_asymmetric_game_traces_smoothly(self):
        u = jnp.array([[3.0, 0.2], [0.1, 2.0]])
        game = DenseTensorGame((u, u.T))
        branch = logit_branch(game, 5.0, n_points=400)
        assert float(branch.lambdas[-1]) >= 5.0
        # sigma stays a valid distribution along the whole branch
        total = branch.sigmas.shape[1]
        assert float(jnp.min(branch.sigmas)) > 0.0
        assert jnp.max(jnp.abs(jnp.sum(branch.sigmas, axis=1) - total / 2.0)) < 1e-6


gambit = pytest.importorskip("pygambit", reason="gambit extra not installed")


class TestGambitBranchAgreement:
    def test_branch_endpoints_match_gambit(self):
        """Test 6 second half: branch profiles at sampled lambdas match Gambit."""
        from strataq.core.solve.validate import gambit_qre_sigma

        game = rock_paper_scissors()
        branch = logit_branch(game, 2.0, n_points=300)
        for probe in (0.8, 1.6):
            idx = int(jnp.argmin(jnp.abs(branch.lambdas - probe)))
            lam_at = float(branch.lambdas[idx])
            theirs = gambit_qre_sigma(game, lam_at)
            ours = branch.sigmas[idx]
            gap = float(jnp.max(jnp.abs(ours - jnp.concatenate(theirs))))
            assert gap < 1e-7
