"""strataq.diagnose — one call, one verdict, and every refusal recoverable.

Every test here is a *user-facing* claim: the quadrant a known game must read, the
refusal a missing coordinate must produce, and the fact that an uncertain reading
degrades to a bound instead of being classified. Games are 2 players x 3 actions
(9 joint states) so the exact route stays sub-second.
"""

import json

import numpy as np
import pytest
import strataq
from strataq.diagnose import R_BANDS, Coordinate, Diagnosis, diagnose

# A chi whose Monte-Carlo interval brackets the 0.02 band edge: R ~ 0.0200,
# 95% CI ~ [0.0013, 0.0477]. Deliberately un-classifiable.
STRADDLING_CHI = [[1.0, 0.05], [0.01, 1.0]]
STRADDLING_SE = [[0.02, 0.02], [0.02, 0.02]]


def _reversible_series(n: int = 400) -> np.ndarray:
    """A random walk: reversible, so the null is not escaped."""
    return np.cumsum(np.random.default_rng(0).normal(size=n))


class TestExactRoute:
    def test_coordination_reads_landscape(self):
        """An exact potential game: R below the band edge, EPR numerically zero."""
        d = diagnose(strataq.games.coordination(2, 3, bonus=2.0), lam=1.5)
        assert d.quadrant == "landscape"
        assert d.live_quadrants == ("landscape",)
        assert d.response.value < R_BANDS[0]
        assert d.dissipation.value == pytest.approx(0.0, abs=1e-9)
        assert d.alpha == pytest.approx(0.0, abs=1e-9)
        assert not d.refusals

    def test_rock_paper_scissors_reads_whirlpool(self):
        d = diagnose(strataq.games.rock_paper_scissors(), lam=1.5)
        assert d.quadrant == "whirlpool"
        assert d.response.value > R_BANDS[1]
        assert d.dissipation.value > 1e-3
        assert d.alpha == pytest.approx(1.0, abs=1e-6)

    def test_plain_lists_and_dense_game_agree(self):
        """payoffs= takes whatever game_thermo takes, plus a DenseTensorGame."""
        game = strataq.games.rock_paper_scissors()
        mats = [np.asarray(u) for u in game.payoffs]
        assert diagnose(game, lam=1.5).response.value == pytest.approx(
            diagnose(mats, lam=1.5).response.value
        )

    def test_lambda_is_required_with_payoffs(self):
        with pytest.raises(ValueError, match="lam"):
            diagnose(strataq.games.matching_pennies())

    def test_lambda_scaling_warning_is_always_present(self):
        d = diagnose(strataq.games.rock_paper_scissors(), lam=1.5)
        assert any("lambda" in w and "magnitude" in w.lower() for w in d.warnings)

    def test_nan_payoffs_raise(self):
        bad = [np.array([[1.0, np.nan], [0.0, 1.0]]), np.array([[1.0, 0.0], [0.0, 1.0]])]
        with pytest.raises(ValueError, match="NaN"):
            diagnose(bad, lam=1.0)


class TestRefusals:
    def test_chi_only_is_undetermined_and_refuses_epr(self):
        d = diagnose(chi=STRADDLING_CHI, chi_se=STRADDLING_SE, seed=1)
        assert d.quadrant == "undetermined"
        assert d.live_quadrants
        assert any("EPR" in r for r in d.refusals)
        assert d.dissipation.kind == "absent"

    def test_series_only_is_undetermined_and_refuses_r(self):
        d = diagnose(series=_reversible_series(), n_surrogates=60, seed=1)
        assert d.quadrant == "undetermined"
        assert d.live_quadrants
        assert any(r.startswith("R not identified") for r in d.refusals)
        assert d.response.kind == "absent"

    def test_no_arguments_raises(self):
        with pytest.raises(ValueError, match="Nothing was supplied"):
            diagnose()

    def test_chi_without_se_refuses_the_band_assignment(self):
        d = diagnose(chi=[[1.0, 0.9], [0.1, 1.0]])
        assert any("chi_se" in r for r in d.refusals)
        assert d.response.kind == "point"


class TestUncertaintyDegrades:
    def test_straddling_chi_is_not_classified(self):
        """R's interval crosses the band edge, so the R half-plane is NOT picked.

        Paired with a reversible series so the EPR side IS determined: the verdict is
        undetermined purely because of the response coordinate. (~1s: the null test
        draws 60 surrogates.)
        """
        d = diagnose(
            chi=STRADDLING_CHI,
            chi_se=STRADDLING_SE,
            series=_reversible_series(),
            n_surrogates=60,
            seed=1,
        )
        assert d.response.kind == "interval"
        assert d.response.lo < R_BANDS[0] < d.response.hi
        assert d.quadrant == "undetermined"
        assert set(d.live_quadrants) == {"landscape", "stalled whirlpool"}
        assert any("straddles" in w for w in d.warnings)

    def test_tight_chi_below_the_edge_is_classified(self):
        """Control: the same machinery does commit when the interval is on one side."""
        d = diagnose(
            chi=[[1.0, 0.2], [0.2, 1.0]],
            chi_se=[[1e-4, 1e-4], [1e-4, 1e-4]],
            series=_reversible_series(),
            n_surrogates=60,
            seed=1,
        )
        assert d.quadrant == "landscape"


class TestPresentation:
    def test_repr_names_the_quadrant(self):
        assert "WHIRLPOOL" in repr(diagnose(strataq.games.rock_paper_scissors(), lam=1.5))

    def test_snippet_is_runnable_looking_python(self):
        for d in (
            diagnose(strataq.games.rock_paper_scissors(), lam=1.5),
            diagnose(chi=STRADDLING_CHI, chi_se=STRADDLING_SE, seed=1),
        ):
            snip = d.snippet()
            assert isinstance(snip, str)
            assert "strataq" in snip
            compile(snip.replace("...,", "None,"), "<snippet>", "exec")

    def test_explain_surfaces_refusals_and_the_lambda_caveat(self):
        d = diagnose(chi=STRADDLING_CHI, chi_se=STRADDLING_SE, seed=1)
        text = d.explain()
        assert "REFUSAL" in text
        assert any(r in text for r in d.refusals)
        assert "MAGNITUDE of R scales with lambda" in text

    def test_explain_is_present_even_with_no_refusals(self):
        text = diagnose(strataq.games.coordination(2, 3), lam=1.5).explain()
        assert "MAGNITUDE of R scales with lambda" in text
        assert "PROVENANCE" in text

    def test_as_dict_round_trips_through_json(self):
        d = diagnose(strataq.games.rock_paper_scissors(), lam=1.5)
        back = json.loads(json.dumps(d.as_dict()))
        assert back["quadrant"] == "whirlpool"
        assert back["response"]["value"] == pytest.approx(d.response.value)
        assert back["provenance"]["n_joint_states"] == 9

    def test_as_dict_round_trips_with_absent_coordinates(self):
        d = diagnose(chi=STRADDLING_CHI, chi_se=STRADDLING_SE, seed=1)
        back = json.loads(json.dumps(d.as_dict()))
        assert back["dissipation"]["kind"] == "absent"
        assert back["live_quadrants"]


class TestCoordinateBands:
    def test_absent_coordinate_says_so(self):
        c = Coordinate("R", None, None, None, "absent", "nothing supplied")
        assert "not identified" in c.band()

    def test_one_sided_bounds_render(self):
        assert Coordinate("e", 0.5, 0.1, None, "lower_bound", "m").band().startswith(">=")
        assert Coordinate("e", 0.01, None, 0.2, "upper_bound", "m").band().startswith("<=")


def test_public_facade_exports_diagnose():
    assert strataq.diagnose is diagnose
    assert strataq.Diagnosis is Diagnosis
    assert "diagnose" in strataq.__all__ and "games" in strataq.__all__
    assert strataq.games.rock_paper_scissors().num_actions == (3, 3)


def test_importing_strataq_does_not_import_matplotlib():
    """viz is an optional extra: the facade must not pull matplotlib in."""
    import subprocess
    import sys

    code = "import sys, strataq; assert 'matplotlib' not in sys.modules"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
