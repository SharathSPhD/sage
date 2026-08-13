"""strataq.viz -- publication output, not debug output.

One palette, shared by the library, the app and the papers. Vector by default,
Type-42 fonts, no seaborn dependency. Every function returns the Axes so a caller can
keep composing.

    from strataq.viz import plot_plane, plot_branch, plot_decomposition, PALETTE

Design rules, enforced by convention rather than code:
  * dashed black is ALWAYS a bound, an oracle, or a reference -- never a series;
  * quadrant colours are fixed and identical in the app's CSS custom properties;
  * every figure that shows a magnitude of R also shows the lambda it was read at.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

__all__ = [
    "PALETTE",
    "REFERENCE_CLOUD",
    "plot_branch",
    "plot_decomposition",
    "plot_plane",
    "use_paper_style",
]

PALETTE: dict[str, str] = {
    "landscape": "#2A9D8F",
    "driven landscape": "#4C72B0",
    "stalled whirlpool": "#DD8452",
    "whirlpool": "#C44E52",
    "undetermined": "#8E8E8E",
    "primary": "#2A9D8F",
    "refuted": "#C44E52",
    "mechanism": "#8172B3",
    "bound": "#000000",
    "ink": "#1A1A1A",
}

#: Systems this project has measured, with the artifact each is drawn from. Plotting a new
#: reading against this cloud is what stops a naked number being mistaken for a result.
REFERENCE_CLOUD: tuple[dict[str, Any], ...] = (
    {
        "label": "Sioux Falls road network",
        "r": 5.6e-17,
        "epr": 0.0,
        "quadrant": "landscape",
        "artifact": "sioux_falls_calibration",
        "exact": True,
    },
    {
        "label": "Colonel Blotto (b=3)",
        "r": 0.118,
        "epr": 0.098,
        "quadrant": "whirlpool",
        "artifact": "blotto_readings",
        "exact": True,
    },
    {
        "label": "rock-paper-scissors",
        "r": 0.78,
        "epr": 0.83,
        "quadrant": "whirlpool",
        "artifact": "estimator_alpha_sweep",
        "exact": True,
    },
    {
        "label": "Dominick's retail panel",
        "r": 0.00112,
        "epr": None,
        "quadrant": "landscape",
        "artifact": "pricing_passthrough_R",
        "exact": False,
    },
    {
        "label": "CAISO day-ahead",
        "r": None,
        "epr": 1.07,
        "quadrant": "driven landscape",
        "artifact": "electricity_irreversibility_dam",
        "exact": False,
    },
)

R_BANDS = (0.02, 0.30)


def use_paper_style() -> None:
    """Apply the house style. Idempotent; safe to call from a notebook."""
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "mathtext.fontset": "dejavuserif",
            "font.size": 8.5,
            "axes.titlesize": 9.0,
            "axes.titleweight": "bold",
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.3,
            "figure.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def _plane_axes(ax: Axes, r_lo: float, r_hi: float, e_lo: float, e_hi: float, e_cut: float) -> Axes:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(r_lo, r_hi)
    ax.set_ylim(e_lo, e_hi)
    r_cut = R_BANDS[0]
    for x0, x1, y0, y1, key in [
        (r_lo, r_cut, e_lo, e_cut, "landscape"),
        (r_lo, r_cut, e_cut, e_hi, "driven landscape"),
        (r_cut, r_hi, e_lo, e_cut, "stalled whirlpool"),
        (r_cut, r_hi, e_cut, e_hi, "whirlpool"),
    ]:
        ax.add_patch(
            mpatches.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                facecolor=PALETTE[key],
                alpha=0.085,
                zorder=0,
                linewidth=0,
            )
        )
    ax.axvline(r_cut, color="#999999", lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax.axhline(e_cut, color="#999999", lw=0.7, ls=(0, (4, 3)), zorder=1)
    for fx, fy, ha, key, lab in [
        (0.015, 0.475, "left", "landscape", "I  landscape"),
        (0.015, 0.985, "left", "driven landscape", "II  driven landscape"),
        (0.985, 0.475, "right", "stalled whirlpool", "III  stalled whirlpool"),
        (0.985, 0.985, "right", "whirlpool", "IV  whirlpool"),
    ]:
        ax.text(
            fx,
            fy,
            lab,
            transform=ax.transAxes,
            ha=ha,
            va="top",
            zorder=8,
            fontsize=8.0,
            color=PALETTE[key],
            fontweight="bold",
        )
    ax.set_xlabel(r"response asymmetry  $\mathcal{R}$   (local)")
    ax.set_ylabel("dissipation  $\\sigma_{\\mathrm{EP}}$  (nats/step)   (global)")
    return ax


def plot_plane(
    readings: Iterable[Any] = (),
    *,
    ax: Axes | None = None,
    reference: bool = True,
    r_lo: float = 3e-5,
    r_hi: float = 3.0,
    e_lo: float = 3e-4,
    e_hi: float = 3.0,
    e_cut: float = 0.03,
) -> Axes:
    """Plot Diagnosis objects (or ``(r, epr, label)`` triples) in the irreversibility plane.

    A reading with only one coordinate is drawn as a band, never as a point on an
    invented axis value. That is the whole reason this function exists rather than a
    two-line scatter call.
    """
    use_paper_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 4.0))
    _plane_axes(ax, r_lo, r_hi, e_lo, e_hi, e_cut)

    if reference:
        for ref in REFERENCE_CLOUD:
            c = PALETTE[ref["quadrant"]]
            r, e = ref["r"], ref["epr"]
            if r is not None and e is not None:
                ax.plot(
                    [max(r, r_lo * 1.4)],
                    [max(e, e_lo * 1.4)],
                    "o",
                    color=c,
                    ms=4.2,
                    alpha=0.55,
                    zorder=3,
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                )
            elif r is not None:
                ax.axvline(r, color=c, lw=0.9, ls=(0, (1, 3)), alpha=0.55, zorder=2)
            elif e is not None:
                ax.axhline(e, color=c, lw=0.9, ls=(0, (1, 3)), alpha=0.55, zorder=2)

    for item in readings:
        r = getattr(getattr(item, "response", None), "value", None)
        e = getattr(getattr(item, "dissipation", None), "value", None)
        q = getattr(item, "quadrant", "undetermined")
        lab = getattr(item, "label", None) or q
        if isinstance(item, (tuple, list)):
            r, e = item[0], item[1]
            lab = item[2] if len(item) > 2 else "reading"
            q = "undetermined"
        c = PALETTE.get(q, PALETTE["undetermined"])
        if r is not None and e is not None:
            ax.plot(
                [max(r, r_lo * 1.4)],
                [max(e, e_lo * 1.4)],
                "*",
                color=c,
                ms=13,
                zorder=9,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=lab,
            )
        elif r is not None:
            ax.axvline(r, color=c, lw=1.6, zorder=6, label=f"{lab} (R only)")
        elif e is not None:
            ax.axhline(e, color=c, lw=1.6, zorder=6, label=f"{lab} (EPR only)")
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="lower right")
    return ax


def plot_branch(
    branch: Any,
    *,
    ax: Axes | None = None,
    mark_lambda: float | None = None,
    turning_points: Sequence[float] | None = None,
) -> Axes:
    """The QRE branch. No library in the ecosystem ships this; every paper redraws it."""
    use_paper_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 3.2))
    # getattr's default is evaluated eagerly, so branch[0] must NOT appear inside it:
    # a branch object with .lambdas but no __getitem__ would raise before the lookup.
    if hasattr(branch, "lambdas") and hasattr(branch, "sigmas"):
        lams = np.asarray(branch.lambdas, dtype=float)
        sig = np.asarray(branch.sigmas, dtype=float)
    else:
        lams = np.asarray(branch[0], dtype=float)
        sig = np.asarray(branch[1], dtype=float)
    sig2 = sig.reshape(len(lams), -1)
    for j in range(sig2.shape[1]):
        ax.plot(lams, sig2[:, j], lw=1.2, alpha=0.9)
    tps = turning_points if turning_points is not None else getattr(branch, "turning_points", ())
    for tp in tps or ():
        ax.axvline(float(tp), color=PALETTE["bound"], ls=(0, (5, 3)), lw=1.0, zorder=1)
    if mark_lambda is not None:
        ax.axvline(mark_lambda, color=PALETTE["refuted"], lw=1.3, zorder=2)
        ax.annotate(
            rf"$\hat\lambda$={mark_lambda:.3g}",
            (mark_lambda, 1.0),
            xycoords=("data", "axes fraction"),
            textcoords="offset points",
            xytext=(4, -10),
            fontsize=7.5,
            color=PALETTE["refuted"],
        )
    ax.set_xscale("log")
    ax.set_xlabel(r"logit precision  $\lambda$")
    ax.set_ylabel("equilibrium probability")
    ax.set_ylim(-0.02, 1.02)
    return ax


def plot_decomposition(
    alpha: float,
    *,
    potential_norm: float | None = None,
    harmonic_norm: float | None = None,
    ax: Axes | None = None,
) -> Axes:
    """The Candogan decomposition as a single stacked bar plus the calibration anchors."""
    use_paper_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 1.7))
    p = 1.0 - float(alpha)
    ax.barh([0], [p], color=PALETTE["landscape"], height=0.5, label="potential")
    ax.barh([0], [alpha], left=[p], color=PALETTE["whirlpool"], height=0.5, label="harmonic")
    for x, lab in [(0.0, "congestion"), (0.69, "Blotto"), (1.0, "RPS")]:
        ax.plot([x], [0.45], "v", color=PALETTE["ink"], ms=5)
        ax.text(x, 0.62, lab, ha="center", fontsize=7.0, color=PALETTE["ink"])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.45, 0.95)
    ax.set_yticks([])
    ax.set_xlabel(rf"harmonic fraction  $\alpha$ = {alpha:.3g}")
    if potential_norm is not None and harmonic_norm is not None:
        ax.set_title(
            f"$\\|u^P\\|$={potential_norm:.4g}   $\\|u^H\\|$={harmonic_norm:.4g}",
            loc="left",
            fontsize=7.5,
            fontweight="normal",
        )
    ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -1.05))
    return ax
