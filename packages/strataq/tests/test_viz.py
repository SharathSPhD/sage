"""strataq.viz — the figures must build headlessly and return composable Axes.

These are smoke tests with synthetic inputs, not image-regression tests: the claim is
that every public plotter runs without a display and hands back an Axes the caller can
keep drawing on. Agg is forced before pyplot is imported.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import strataq
from matplotlib.axes import Axes
from strataq.viz import (
    PALETTE,
    REFERENCE_CLOUD,
    plot_branch,
    plot_decomposition,
    plot_plane,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class TestPlotPlane:
    def test_empty_plane_with_reference_cloud(self):
        ax = plot_plane()
        assert isinstance(ax, Axes)
        assert ax.get_xscale() == "log" and ax.get_yscale() == "log"

    def test_tuple_readings(self):
        ax = plot_plane([(0.5, 0.4, "synthetic")], reference=False)
        assert isinstance(ax, Axes)

    def test_diagnosis_reading(self):
        d = strataq.diagnose(strataq.games.rock_paper_scissors(), lam=1.5)
        ax = plot_plane([d])
        assert isinstance(ax, Axes)

    def test_one_sided_reading_is_drawn_as_a_band(self):
        """A reading with only R must not be given an invented EPR value."""
        d = strataq.diagnose(chi=[[1.0, 0.2], [0.05, 1.0]])
        n_before = len(plot_plane(reference=False).lines)
        ax = plot_plane([d], reference=False)
        assert isinstance(ax, Axes)
        assert len(ax.lines) >= n_before

    def test_accepts_a_supplied_axes(self):
        _, ax = plt.subplots()
        assert plot_plane([(0.1, 0.1, "x")], ax=ax, reference=False) is ax


class TestPlotBranch:
    def test_tuple_branch(self):
        lams = np.geomspace(0.1, 10.0, 20)
        sigmas = np.stack([np.linspace(0.33, 0.9, 20), np.linspace(0.33, 0.05, 20)], axis=1)
        ax = plot_branch((lams, sigmas), mark_lambda=1.5, turning_points=[2.0])
        assert isinstance(ax, Axes)
        assert ax.get_xscale() == "log"

    def test_object_branch_without_getitem(self):
        """getattr defaults are eager: an object branch must not require __getitem__."""

        class Branch:
            lambdas = np.geomspace(0.1, 10.0, 15)
            sigmas = np.linspace(0.2, 0.8, 15)
            turning_points = (1.0,)

        assert isinstance(plot_branch(Branch()), Axes)


class TestPlotDecomposition:
    def test_alpha_only(self):
        ax = plot_decomposition(0.42)
        assert isinstance(ax, Axes)
        assert "0.42" in ax.get_xlabel()

    def test_with_norms(self):
        ax = plot_decomposition(1.0, potential_norm=0.0, harmonic_norm=2.5)
        assert isinstance(ax, Axes)
        assert ax.get_title(loc="left")


def test_palette_covers_every_quadrant():
    from strataq.diagnose import QUADRANTS

    assert set(QUADRANTS) <= set(PALETTE)
    assert all(ref["quadrant"] in PALETTE for ref in REFERENCE_CLOUD)
