"""Figures for p3_noneq — regenerated from committed gate artifacts only.

Run: uv run python papers/p3_noneq/make_figures.py
No solves happen here; the paper draws what the gates already checked.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "benchmarks" / "results"
FIGS = Path(__file__).resolve().parent / "figures"
FIGS.mkdir(exist_ok=True)


def phase_wedge() -> None:
    d = json.loads((RESULTS / "phase_map_surface.json").read_text())
    lam, alpha = np.array(d["lambdas"]), np.array(d["alphas"])
    rho = np.array(d["grid"]["rho"])
    sup = np.array(d["grid"]["supercritical_frac"])

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), constrained_layout=True)
    im0 = axes[0].pcolormesh(lam, alpha, rho, shading="nearest", cmap="viridis")
    axes[0].contour(lam, alpha, rho, levels=[1.0], colors="w", linewidths=1.4)
    axes[0].set(xscale="log", xlabel=r"$\lambda$", ylabel=r"$\alpha$", title=r"median $\rho(SB)$")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(lam, alpha, sup, shading="nearest", cmap="inferno", vmin=0, vmax=1)
    axes[1].set(xscale="log", xlabel=r"$\lambda$", title="supercritical fraction")
    fig.colorbar(im1, ax=axes[1])
    fig.savefig(FIGS / "phase_wedge.pdf")
    plt.close(fig)


def estimator_tracking() -> None:
    d = json.loads((RESULTS / "estimator_alpha_sweep.json").read_text())
    m = d["metrics"]
    alphas = sorted({float(k.split("_")[1]) for k in m if k.startswith("alpha_")})
    exact = np.array([m[f"alpha_{a:.2f}_exact"] for a in alphas])
    kld = np.array([m[f"alpha_{a:.2f}_kld"] for a in alphas])
    tur = np.array([m[f"alpha_{a:.2f}_tur"] for a in alphas])
    lo = np.array([m[f"alpha_{a:.2f}_tur_ci_low"] for a in alphas])

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), constrained_layout=True)
    axes[0].plot(alphas, exact, "k-", lw=2, label=r"exact $\sigma_{\rm EP}$")
    axes[0].plot(alphas, kld, "o--", color="tab:green", ms=4, label="KLD $k{=}1$")
    axes[0].plot(alphas, tur, "s--", color="tab:orange", ms=4, label="TUR point")
    axes[0].fill_between(
        alphas, lo, tur, color="tab:orange", alpha=0.25, label="TUR certification band"
    )
    axes[0].set(
        xlabel=r"$\alpha$", ylabel="EPR (nats / unit time)", title="trajectory estimators vs exact"
    )
    axes[0].legend(fontsize=8)

    pos = exact > 1e-6
    axes[1].plot(np.array(alphas)[pos], (tur / exact)[pos], "s-", color="tab:orange", ms=4)
    axes[1].axhline(1.0, color="k", ls=":", lw=1)
    axes[1].set(
        xlabel=r"$\alpha$",
        ylabel="TUR bound / exact EPR",
        title="TUR tightness: saturation near equilibrium",
    )
    fig.savefig(FIGS / "estimator_tracking.pdf")
    plt.close(fig)


def decoupling() -> None:
    d = json.loads((RESULTS / "chain_comovement.json").read_text())
    m = d["metrics"]
    levels = sorted({float(k.split("_")[-1]) for k in m if k.startswith("within_level_rho_alpha_")})
    rho = [m[f"within_level_rho_alpha_{a}"] for a in levels]

    fig, ax = plt.subplots(figsize=(4.6, 3.2), constrained_layout=True)
    ax.axhline(0, color="k", lw=0.8)
    ax.plot(levels, rho, "o-", color="tab:red", ms=5)
    ax.set(
        xlabel=r"$\alpha$ level",
        ylabel=r"within-level Spearman $\rho(\sigma_{\rm EP}, \mathcal{R})$",
        title="the meters decouple, then anti-align",
    )
    fig.savefig(FIGS / "decoupling.pdf")
    plt.close(fig)


if __name__ == "__main__":
    phase_wedge()
    estimator_tracking()
    decoupling()
    print("figures ->", FIGS)
