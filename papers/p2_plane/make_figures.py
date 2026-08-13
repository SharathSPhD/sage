"""Figure program for the flagship paper (papers/p2_plane).

Draws ONLY from committed artifacts in benchmarks/results/. No solves, no fitting,
no data invented here. Every panel names its source artifact in FIGURE_SOURCES so
the provenance appendix can be generated mechanically.

Run:  uv run python papers/p2_plane/make_figures.py
Out:  papers/p2_plane/figures/*.pdf  (vector, Type-42 fonts)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------------------
# House style. One palette for the paper, the app and strataq.viz.
# --------------------------------------------------------------------------------------

PALETTE = {
    # quadrant semantics -- identical in the app and in strataq.viz
    "landscape": "#2A9D8F",  # I   reciprocal, no flux
    "driven": "#4C72B0",  # II  reciprocal, circulating
    "stalled": "#DD8452",  # III asymmetric, no flux
    "whirlpool": "#C44E52",  # IV  asymmetric, circulating
    # series semantics -- dashed black is ALWAYS a bound or a reference
    "primary": "#2A9D8F",
    "refuted": "#C44E52",
    "mechanism": "#8172B3",
    "bound": "#000000",
    "grid": "#D9D9D9",
    "ink": "#1A1A1A",
}

mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "axes.titleweight": "bold",
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.3,
        "lines.markersize": 4.0,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)

# MDPI single column ~ 6.7in usable width for a full-width float.
W_FULL, W_HALF = 6.7, 3.3

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks" / "results"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

FIGURE_SOURCES: dict[str, list[str]] = {}


def load(name: str) -> dict:
    with open(RESULTS / f"{name}.json") as fh:
        return json.load(fh)


def stamp(fig_id: str, *artifacts: str) -> None:
    FIGURE_SOURCES[fig_id] = list(artifacts)


ALPHAS = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])


def panel_title(ax, text: str) -> None:
    ax.set_title(text, loc="left", pad=6)


# --------------------------------------------------------------------------------------
# Figure 1 -- the two routes.  Schematic: why the coordinates are different objects.
# --------------------------------------------------------------------------------------


def fig1_two_routes() -> None:
    fig, ax = plt.subplots(figsize=(W_FULL, 2.55))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 42)
    ax.axis("off")

    def box(
        x, y, w, h, label, fc="white", ec=PALETTE["ink"], fs=8.0, lw=0.8, style="round,pad=0.35"
    ):
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, y), w, h, boxstyle=style, linewidth=lw, edgecolor=ec, facecolor=fc
            )
        )
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, zorder=5)

    def arrow(x0, y0, x1, y1, color=PALETTE["ink"]):
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", lw=0.9, color=color, shrinkA=2, shrinkB=2),
        )

    box(1, 17, 15, 8, "finite game\n$u$, grid $m_i$", fc="#F2F2F2")
    box(20, 17, 15, 8, "normalise\n+ logit QRE\n$\\sigma^*(\\lambda)$", fc="#F2F2F2")
    arrow(16, 21, 20, 21)

    # upper route: local derivative
    box(
        41,
        29,
        21,
        9,
        "$\\chi^{\\mathrm{eq}}=(I-SB)^{-1}S$\nlocal derivative at $\\sigma^*$",
        ec=PALETTE["landscape"],
        lw=1.1,
    )
    box(
        67,
        29,
        15,
        9,
        "$\\mathcal{R}$\nresponse\nasymmetry",
        ec=PALETTE["landscape"],
        lw=1.4,
        fc="#EAF5F3",
    )
    arrow(35, 22.5, 41, 32, PALETTE["landscape"])
    arrow(62, 33.5, 67, 33.5, PALETTE["landscape"])

    # lower route: global flux
    box(
        41,
        4,
        21,
        9,
        "Glauber generator on\n$\\prod_i m_i$ profiles; $\\pi^*$, $J^*$",
        ec=PALETTE["whirlpool"],
        lw=1.1,
    )
    box(
        67,
        4,
        15,
        9,
        "$\\sigma_{\\mathrm{EP}}$\ndissipation",
        ec=PALETTE["whirlpool"],
        lw=1.4,
        fc="#F8ECEC",
    )
    arrow(35, 19.5, 41, 10, PALETTE["whirlpool"])
    arrow(62, 8.5, 67, 8.5, PALETTE["whirlpool"])

    box(86, 16.5, 13, 9, "the plane\n$(\\mathcal{R},\\sigma_{\\mathrm{EP}})$", fc="#F2F2F2", lw=1.0)
    arrow(82, 32, 88, 26)
    arrow(82, 8, 88, 15)

    ax.text(
        52,
        40.4,
        "local route  —  one equilibrium, a derivative",
        ha="center",
        fontsize=7.8,
        color=PALETTE["landscape"],
        style="italic",
    )
    ax.text(
        52,
        0.6,
        "global route  —  the whole profile space, a flux functional",
        ha="center",
        fontsize=7.8,
        color=PALETTE["whirlpool"],
        style="italic",
    )
    ax.text(
        50,
        21.0,
        "both vanish $\\Longleftrightarrow$ the normalised game is potential",
        ha="center",
        va="center",
        fontsize=7.6,
        color="#666666",
    )

    fig.savefig(OUT / "fig1_two_routes.pdf")
    plt.close(fig)
    stamp("fig1", "schematic (no artifact)")


# --------------------------------------------------------------------------------------
# Figure 2 -- the plane.  The claim, and where every measured system sits in it.
# --------------------------------------------------------------------------------------


def fig2_plane() -> None:
    dec = load("decoupling_mechanism")["metrics"]
    est = load("estimator_alpha_sweep")["metrics"]
    price = load("pricing_passthrough_R")["metrics"]
    caiso = load("electricity_irreversibility_dam")["metrics"]
    blotto = load("blotto_readings")["metrics"]
    load("sioux_falls_calibration")["metrics"]

    r_med = np.array(
        [dec[f"alpha_{a:.2f}_median_num"] / dec[f"alpha_{a:.2f}_median_den"] for a in ALPHAS]
    )
    epr = np.array([est[f"alpha_{a:.2f}_exact"] for a in ALPHAS])

    fig, ax = plt.subplots(figsize=(W_FULL, 4.1))

    r_lo, r_hi = 3e-5, 3.0
    e_lo, e_hi = 3e-4, 3.0
    r_cut, e_cut = 0.02, 0.03  # toolkit calibrated band edge; EPR null-scale edge

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(r_lo, r_hi)
    ax.set_ylim(e_lo, e_hi)

    quad = [
        ((r_lo, e_lo), r_cut / r_lo, e_cut / e_lo, "landscape"),
        ((r_lo, e_cut), r_cut / r_lo, e_hi / e_cut, "driven"),
        ((r_cut, e_lo), r_hi / r_cut, e_cut / e_lo, "stalled"),
        ((r_cut, e_cut), r_hi / r_cut, e_hi / e_cut, "whirlpool"),
    ]
    for (x0, y0), _w, _h, key in quad:
        ax.add_patch(
            mpatches.Rectangle(
                (x0, y0),
                (r_cut - x0) if x0 == r_lo else (r_hi - x0),
                (e_cut - y0) if y0 == e_lo else (e_hi - y0),
                facecolor=PALETTE[key],
                alpha=0.075,
                zorder=0,
                linewidth=0,
            )
        )
    ax.axvline(r_cut, color="#999999", lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax.axhline(e_cut, color="#999999", lw=0.7, ls=(0, (4, 3)), zorder=1)

    for xy, key, lab in [
        ((4.5e-5, 4.5e-4), "landscape", "I  landscape"),
        ((4.5e-5, 1.7), "driven", "II  driven landscape"),
        ((1.3, 4.5e-4), "stalled", "III  stalled whirlpool"),
        ((1.3, 1.7), "whirlpool", "IV  whirlpool"),
    ]:
        ax.text(
            *xy,
            lab,
            fontsize=8.0,
            color=PALETTE[key],
            fontweight="bold",
            ha="left" if xy[0] < 1e-2 else "right",
            va="bottom" if xy[1] < 1e-2 else "top",
            zorder=6,
        )

    # the synthetic alpha-family: what a one-dimensional theory predicts
    ax.plot(
        r_med,
        epr,
        "-o",
        color=PALETTE["ink"],
        lw=1.4,
        ms=3.4,
        zorder=4,
        markerfacecolor="white",
        markeredgewidth=0.9,
    )
    for a, x, y in zip(ALPHAS, r_med, epr, strict=False):
        if a in (0.05, 0.45, 0.95):
            ax.annotate(
                f"$\\alpha$={a:g}",
                (x, y),
                textcoords="offset points",
                xytext=(7, -8),
                fontsize=6.8,
                color=PALETTE["ink"],
            )

    # measured systems, with partial coordinates drawn as bands rather than points
    def band_v(x, xlo, xhi, label, color, ytext):
        ax.add_patch(
            mpatches.Rectangle(
                (xlo, e_lo),
                xhi - xlo,
                e_hi - e_lo,
                facecolor=color,
                alpha=0.16,
                zorder=2,
                linewidth=0,
            )
        )
        ax.plot([x, x], [e_lo, e_hi], color=color, lw=1.1, ls=(0, (1, 2)), zorder=3)
        ax.text(
            x,
            ytext,
            label,
            rotation=90,
            fontsize=6.9,
            color=color,
            ha="right",
            va="bottom",
            zorder=6,
        )

    def band_h(y, label, color, xtext):
        ax.axhspan(y * 0.72, y * 1.38, facecolor=color, alpha=0.16, zorder=2, linewidth=0)
        ax.plot([r_lo, r_hi], [y, y], color=color, lw=1.1, ls=(0, (1, 2)), zorder=3)
        ax.text(xtext, y * 1.45, label, fontsize=6.9, color=color, ha="left", va="bottom", zorder=6)

    band_v(
        price["R_empirical"],
        price["R_ci_low"],
        price["R_ci_high"],
        "Dominick's  $\\mathcal{R}$=0.0011  ($\\sigma_{\\mathrm{EP}}$ not read)",
        PALETTE["landscape"],
        6e-4,
    )
    band_h(
        caiso["kld_embed_per_hour"] * 24.0,
        "CAISO day-ahead  1.07 nats/day  ($\\mathcal{R}$ not read)",
        PALETTE["driven"],
        6e-5,
    )

    ax.plot(
        [blotto["R_b3_k3"]],
        [blotto["epr_b2_k2_asym"]],
        "s",
        color=PALETTE["whirlpool"],
        ms=5.5,
        zorder=7,
        markeredgecolor="white",
        markeredgewidth=0.7,
    )
    ax.annotate(
        "Blotto\n($\\alpha$=0.69)",
        (blotto["R_b3_k3"], blotto["epr_b2_k2_asym"]),
        textcoords="offset points",
        xytext=(9, 2),
        fontsize=6.9,
        color=PALETTE["whirlpool"],
    )

    ax.plot(
        [r_lo * 1.35],
        [e_lo * 1.35],
        "^",
        color=PALETTE["landscape"],
        ms=6,
        zorder=7,
        markeredgecolor="white",
        markeredgewidth=0.7,
        clip_on=False,
    )
    ax.annotate(
        "Sioux Falls\n$\\mathcal{R}=5.6\\times10^{-17}$, $\\sigma_{\\mathrm{EP}}=0$ exactly",
        (r_lo * 1.35, e_lo * 1.35),
        textcoords="offset points",
        xytext=(11, 3),
        fontsize=6.9,
        color=PALETTE["landscape"],
    )

    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.34, 6e-4),
            1.9,
            3.4e-3,
            boxstyle="round,pad=0.02",
            facecolor="white",
            edgecolor=PALETTE["stalled"],
            lw=0.9,
            zorder=8,
            mutation_aspect=1,
        )
    )
    ax.text(
        0.80,
        1.42e-3,
        "unoccupied\n— the target",
        ha="center",
        va="center",
        fontsize=7.2,
        color=PALETTE["stalled"],
        zorder=9,
    )

    ax.set_xlabel("response asymmetry  $\\mathcal{R}$   (local, derivative)")
    ax.set_ylabel("dissipation  $\\sigma_{\\mathrm{EP}}$  (nats/step)\n(global, flux)")

    handles = [
        Line2D(
            [],
            [],
            color=PALETTE["ink"],
            marker="o",
            markerfacecolor="white",
            lw=1.4,
            label="synthetic $\\alpha$-family (median over 100 games/level; "
            "$\\mathcal{R}$ at $\\lambda$=1.2, $\\sigma_{\\mathrm{EP}}$ at $\\lambda$=1.5)",
        ),
        Line2D([], [], color=PALETTE["whirlpool"], marker="s", lw=0, label="calibration anchor"),
        Line2D(
            [],
            [],
            color="#777777",
            lw=1.1,
            ls=(0, (1, 2)),
            label="measured system, one coordinate only",
        ),
    ]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 0.02), ncol=1)

    fig.savefig(OUT / "fig2_plane.pdf")
    plt.close(fig)
    stamp(
        "fig2",
        "decoupling_mechanism",
        "estimator_alpha_sweep",
        "pricing_passthrough_R",
        "electricity_irreversibility_dam",
        "blotto_readings",
        "sioux_falls_calibration",
    )


# --------------------------------------------------------------------------------------
# Figure 3 -- the money figure.  The collapse, the refuted repair, and the mechanism.
# --------------------------------------------------------------------------------------


def fig3_collapse() -> None:
    chain = load("chain_comovement")
    dec = load("decoupling_mechanism")["metrics"]
    m = chain["metrics"]

    rho_ratio = np.array([m[f"within_level_rho_alpha_{a:.2f}"] for a in ALPHAS])
    rho_num = np.array([dec[f"alpha_{a:.2f}_rho_num_epr"] for a in ALPHAS])
    rho_invden = np.array([dec[f"alpha_{a:.2f}_rho_ratio_invden"] for a in ALPHAS])
    marginal = m["spearman_epr_R_marginal"]

    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.9))

    ax = axes[0]
    ax.axhline(marginal, color=PALETTE["bound"], ls=(0, (5, 3)), lw=1.1, zorder=2)
    ax.text(
        0.07,
        marginal - 0.075,
        f"marginal $\\rho_S$ = {marginal:.3f}",
        fontsize=7.0,
        color=PALETTE["bound"],
        va="top",
    )
    ax.axhline(0.0, color="#BBBBBB", lw=0.7, zorder=1)
    ax.plot(
        ALPHAS,
        rho_ratio,
        "-o",
        color=PALETTE["primary"],
        zorder=4,
        markerfacecolor="white",
        markeredgewidth=1.0,
    )
    ax.plot(
        ALPHAS,
        rho_num,
        "-^",
        color=PALETTE["refuted"],
        ls=(0, (4, 2)),
        zorder=3,
        markerfacecolor="white",
        markeredgewidth=1.0,
    )
    ax.fill_between([0.7, 1.0], -0.6, 1.05, color="#CCCCCC", alpha=0.25, zorder=0, linewidth=0)
    ax.text(0.85, -0.5, "collapse\nregion", fontsize=7.0, color="#777777", ha="center")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.6, 1.05)
    ax.set_xlabel("harmonic fraction  $\\alpha$")
    ax.set_ylabel("$\\rho_S(\\sigma_{\\mathrm{EP}},\\ \\cdot\\ )$  within $\\alpha$-level")
    panel_title(ax, "Conditional co-movement collapses ($m=3$, $\\lambda=1.2$)")

    ax = axes[1]
    ax.axhline(0.0, color="#BBBBBB", lw=0.7, zorder=1)
    ax.plot(
        ALPHAS,
        rho_invden,
        "-s",
        color=PALETTE["mechanism"],
        zorder=4,
        markerfacecolor="white",
        markeredgewidth=1.0,
    )
    ax.annotate(
        f"{dec['h2_min_high_alpha_rho_ratio_invden']:.3f}",
        (ALPHAS[-1], rho_invden[-1]),
        textcoords="offset points",
        xytext=(-6, -13),
        fontsize=7.2,
        color=PALETTE["mechanism"],
        ha="right",
    )
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    med_num = np.array([dec[f"alpha_{a:.2f}_median_num"] for a in ALPHAS])
    med_den = np.array([dec[f"alpha_{a:.2f}_median_den"] for a in ALPHAS])
    ax2.plot(ALPHAS, med_num, ":", color="#999999", lw=1.1)
    ax2.plot(ALPHAS, med_den, "-.", color="#999999", lw=1.1)
    ax2.set_ylabel("median $\\|\\chi\\pm\\chi^{\\top}\\|_F$", color="#777777", fontsize=7.8)
    ax2.tick_params(axis="y", colors="#777777")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.6, 1.05)
    ax.set_xlabel("harmonic fraction  $\\alpha$")
    ax.set_ylabel("$\\rho_S(\\mathcal{R},\\ 1/\\|\\chi+\\chi^{\\top}\\|_F)$")
    panel_title(ax, "The mechanism: $\\mathcal{R}$ becomes denominator-driven")

    handles = [
        Line2D(
            [],
            [],
            color=PALETTE["primary"],
            marker="o",
            markerfacecolor="white",
            label="$\\mathcal{R}$ (the ratio)",
        ),
        Line2D(
            [],
            [],
            color=PALETTE["refuted"],
            marker="^",
            ls=(0, (4, 2)),
            markerfacecolor="white",
            label="numerator $\\|\\chi-\\chi^{\\top}\\|_F$ alone — the refuted repair (H1)",
        ),
        Line2D(
            [],
            [],
            color=PALETTE["mechanism"],
            marker="s",
            markerfacecolor="white",
            label="$1/$denominator (H2)",
        ),
        Line2D(
            [],
            [],
            color=PALETTE["bound"],
            ls=(0, (5, 3)),
            label="marginal (unconditional) reference",
        ),
        Line2D([], [], color="#999999", ls=":", label="median numerator"),
        Line2D([], [], color="#999999", ls="-.", label="median denominator"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.20), ncol=3)
    fig.savefig(OUT / "fig3_collapse.pdf")
    plt.close(fig)
    stamp("fig3", "chain_comovement", "decoupling_mechanism")


# --------------------------------------------------------------------------------------
# Figure 4 -- criticality escape and the frontier.
# --------------------------------------------------------------------------------------


def fig4_frontier() -> None:
    surf = load("phase_map_surface")
    fro = load("frontier_lambda_c")["metrics"]

    lams = np.array(surf["lambdas"], dtype=float)
    als = np.array(surf["alphas"], dtype=float)
    rho = np.array(surf["grid"]["rho"], dtype=float)  # [alpha][lambda]

    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.9))

    ax = axes[0]
    mesh = ax.pcolormesh(
        lams,
        als,
        rho,
        shading="nearest",
        cmap="magma",
        vmin=0.0,
        vmax=max(1.3, float(np.nanmax(rho))),
    )
    try:
        cs = ax.contour(lams, als, rho, levels=[1.0], colors="white", linewidths=1.4)
        ax.clabel(cs, fmt={1.0: r"$\rho=1$"}, fontsize=7.0)
    except Exception:  # pragma: no cover - contour may be empty on a coarse grid
        pass
    ax.set_xscale("log")
    ax.set_xlabel("logit precision  $\\lambda$")
    ax.set_ylabel("harmonic fraction  $\\alpha$")
    panel_title(ax, "Median $\\rho(SB)$ over the $(\\lambda,\\alpha)$ plane")
    cb = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.046)
    cb.ax.tick_params(labelsize=7.0)
    cb.set_label("$\\rho(SB)$", fontsize=7.8)

    ax = axes[1]
    lv = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    xs, ys = [], []
    for a in lv:
        v = fro.get(f"lambda_c_a{a:.2f}")
        if v is not None:
            xs.append(a)
            ys.append(v)
    ax.axvspan(0.28, 0.525, color="#CCCCCC", alpha=0.35, linewidth=0, zorder=0)
    ax.text(
        0.40,
        6.6,
        "no crossing:\n$\\rho(SB)<1$ at every $\\lambda$",
        fontsize=7.0,
        color="#666666",
        ha="center",
    )
    ax.plot(
        xs,
        ys,
        "-o",
        color=PALETTE["whirlpool"],
        markerfacecolor="white",
        markeredgewidth=1.0,
        zorder=3,
    )
    for a, v in zip(xs, ys, strict=False):
        if a in (0.55, 0.80):
            ax.annotate(
                f"{v:.2f}",
                (a, v),
                textcoords="offset points",
                xytext=(4, 5),
                fontsize=6.9,
                color=PALETTE["whirlpool"],
            )
    ax.set_yscale("log")
    ax.set_xlim(0.28, 0.84)
    ax.set_xlabel("harmonic fraction  $\\alpha$")
    ax.set_ylabel("critical precision  $\\lambda_c(\\alpha)$")
    panel_title(ax, "The supercritical frontier descends in $\\alpha$")

    fig.savefig(OUT / "fig4_frontier.pdf")
    plt.close(fig)
    stamp("fig4", "phase_map_surface", "frontier_lambda_c")


# --------------------------------------------------------------------------------------
# Figure 5 -- observability: the dissipation axis is readable from trajectories alone.
# --------------------------------------------------------------------------------------


def fig5_observability() -> None:
    est = load("estimator_alpha_sweep")["metrics"]

    exact = np.array([est[f"alpha_{a:.2f}_exact"] for a in ALPHAS])
    kld = np.array([est[f"alpha_{a:.2f}_kld"] for a in ALPHAS])
    tur = np.array([est[f"alpha_{a:.2f}_tur"] for a in ALPHAS])
    tur_lo = np.array([est[f"alpha_{a:.2f}_tur_ci_low"] for a in ALPHAS])

    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.75))

    ax = axes[0]
    ax.plot(ALPHAS, exact, "-", color=PALETTE["bound"], ls=(0, (5, 3)), lw=1.3, zorder=4)
    ax.plot(
        ALPHAS,
        kld,
        "-o",
        color=PALETTE["primary"],
        markerfacecolor="white",
        markeredgewidth=1.0,
        zorder=3,
    )
    ax.plot(
        ALPHAS,
        tur,
        "-s",
        color=PALETTE["mechanism"],
        markerfacecolor="white",
        markeredgewidth=1.0,
        zorder=3,
    )
    ax.fill_between(ALPHAS, tur_lo, tur, color=PALETTE["mechanism"], alpha=0.18, linewidth=0)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("harmonic fraction  $\\alpha$")
    ax.set_ylabel("$\\sigma_{\\mathrm{EP}}$  (nats/step)")
    panel_title(ax, "Trajectory estimators track the exact meter")

    ax = axes[1]
    ax.axhline(1.0, color=PALETTE["bound"], ls=(0, (5, 3)), lw=1.1)
    ax.plot(
        ALPHAS,
        tur / exact,
        "-s",
        color=PALETTE["mechanism"],
        markerfacecolor="white",
        markeredgewidth=1.0,
    )
    ax.plot(
        ALPHAS,
        kld / exact,
        "-o",
        color=PALETTE["primary"],
        markerfacecolor="white",
        markeredgewidth=1.0,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.4, 1.2)
    ax.set_xlabel("harmonic fraction  $\\alpha$")
    ax.set_ylabel("estimator / exact")
    panel_title(ax, "Tightness")

    handles = [
        Line2D([], [], color=PALETTE["bound"], ls=(0, (5, 3)), label="exact Schnakenberg EPR"),
        Line2D(
            [],
            [],
            color=PALETTE["primary"],
            marker="o",
            markerfacecolor="white",
            label="$k$-block KLD estimator",
        ),
        Line2D(
            [],
            [],
            color=PALETTE["mechanism"],
            marker="s",
            markerfacecolor="white",
            label="debiased finite-time TUR bound (shaded: to CI low)",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    fig.savefig(OUT / "fig5_observability.pdf")
    plt.close(fig)
    stamp("fig5", "estimator_alpha_sweep")


# --------------------------------------------------------------------------------------
# Figure 6 -- the honesty exhibit.  Both kill-shots, and the temperature confound.
# --------------------------------------------------------------------------------------


def _grid_from_keys(metrics: dict, arm: str) -> dict[int, list[str]]:
    """Discover the (m, alpha) cells an arm actually ran, from the artifact keys alone."""
    pat = re.compile(rf"^{arm}_m(\d+)_a([0-9.]+)_rho_epr_r$")
    grid: dict[int, list[str]] = {}
    for key in metrics:
        hit = pat.match(key)
        if hit:
            grid.setdefault(int(hit.group(1)), []).append(hit.group(2))
    for m_val in grid:
        grid[m_val] = sorted(grid[m_val], key=float)
    return grid


def fig6_robustness() -> None:
    met = load("plane_robustness")["metrics"]

    main_grid = _grid_from_keys(met, "main")
    ctrl_grid = _grid_from_keys(met, "d1")
    ms = sorted(main_grid)
    levels = main_grid[ms[0]]
    a_hi = levels[-1]
    alphas = np.array([float(a) for a in levels])

    series_colors = [
        PALETTE["primary"],
        PALETTE["driven"],
        PALETTE["mechanism"],
        PALETTE["whirlpool"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.95))
    fig.subplots_adjust(wspace=0.30)

    # ---- (a) the collapse at every action count ---------------------------------------
    ax = axes[0]
    ax.axhline(0.0, color=PALETTE["bound"], ls=(0, (5, 3)), lw=1.1, zorder=2)
    for i, m_val in enumerate(ms):
        rho = np.array([met[f"main_m{m_val}_a{a}_rho_epr_r"] for a in levels])
        lo = np.array([met[f"main_m{m_val}_a{a}_ci_low"] for a in levels])
        hi = np.array([met[f"main_m{m_val}_a{a}_ci_high"] for a in levels])
        ax.errorbar(
            alphas,
            rho,
            yerr=np.vstack([rho - lo, hi - rho]),
            fmt="-o",
            color=series_colors[i % len(series_colors)],
            markerfacecolor="white",
            markeredgewidth=1.0,
            ms=3.6,
            elinewidth=0.6,
            capsize=1.6,
            capthick=0.6,
            zorder=3 + i,
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("harmonic fraction  $\\alpha$")
    ax.set_ylabel("$\\rho_S(\\sigma_{\\mathrm{EP}},\\mathcal{R})$  within $\\alpha$-level")
    panel_title(ax, "The collapse holds at every $m$")

    # ---- (b) fixed scale versus the constant-RMS control ------------------------------
    ax = axes[1]
    ax.axhline(0.0, color=PALETTE["bound"], ls=(0, (5, 3)), lw=1.1, zorder=2)
    m_ctrl = sorted(m for m in ctrl_grid if a_hi in ctrl_grid[m])
    for arm, xs, color, marker, ls in (
        ("main", ms, PALETTE["ink"], "o", "-"),
        ("d1", m_ctrl, PALETTE["stalled"], "s", (0, (4, 2))),
    ):
        rho = np.array([met[f"{arm}_m{m_val}_a{a_hi}_rho_epr_r"] for m_val in xs])
        lo = np.array([met[f"{arm}_m{m_val}_a{a_hi}_ci_low"] for m_val in xs])
        hi = np.array([met[f"{arm}_m{m_val}_a{a_hi}_ci_high"] for m_val in xs])
        lam = np.array([met[f"{arm}_m{m_val}_a{a_hi}_median_lambda_normalised"] for m_val in xs])
        ax.errorbar(
            xs,
            rho,
            yerr=np.vstack([rho - lo, hi - rho]),
            fmt=marker,
            ls=ls,
            color=color,
            markerfacecolor="white",
            markeredgewidth=1.0,
            ms=4.0,
            elinewidth=0.7,
            capsize=2.0,
            capthick=0.7,
            zorder=4,
        )
        dy = 13 if arm == "main" else -17
        for x, y, lv in zip(xs, rho, lam, strict=False):
            ax.annotate(
                f"$\\bar\\lambda$={lv:.2f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, dy),
                ha="center",
                fontsize=6.4,
                color=color,
            )
    ax.set_xticks(ms)
    ax.set_xlim(min(ms) - 0.45, max(ms) + 0.45)
    ax.set_xlabel("actions per player  $m$")
    ax.set_ylabel(f"$\\rho_S$ at $\\alpha={float(a_hi):g}$")
    panel_title(ax, "The $m$-trend is temperature")

    handles = [
        Line2D(
            [],
            [],
            color=series_colors[i % len(series_colors)],
            marker="o",
            markerfacecolor="white",
            label=f"$m={m_val}$",
        )
        for i, m_val in enumerate(ms)
    ] + [
        Line2D(
            [],
            [],
            color=PALETTE["ink"],
            marker="o",
            markerfacecolor="white",
            label="fixed Frobenius scale (main arm)",
        ),
        Line2D(
            [],
            [],
            color=PALETTE["stalled"],
            marker="s",
            ls=(0, (4, 2)),
            markerfacecolor="white",
            label="constant per-entry RMS (registered control)",
        ),
        Line2D([], [], color=PALETTE["bound"], ls=(0, (5, 3)), label="zero reference"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=4)
    fig.savefig(OUT / "fig6_robustness.pdf")
    plt.close(fig)
    stamp("fig6", "plane_robustness")


if __name__ == "__main__":
    fig1_two_routes()
    fig2_plane()
    fig3_collapse()
    fig4_frontier()
    fig5_observability()
    fig6_robustness()
    with open(OUT / "figure_sources.json", "w") as fh:
        json.dump(FIGURE_SOURCES, fh, indent=2)
    print("wrote:", ", ".join(sorted(p.name for p in OUT.glob("*.pdf"))))
