"""Regenerate the paper's figures from the benchmark artifacts (never hand-drawn)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
REPO = HERE.parents[1]
surface = json.loads((REPO / "benchmarks" / "results" / "phase_map_surface.json").read_text())

lambdas, alphas, grid = surface["lambdas"], surface["alphas"], surface["grid"]
panels = [
    ("rho", r"$\rho(SB)$"),
    ("reciprocity", r"$\mathcal{R}$"),
    ("epr", r"$\sigma_{\mathrm{EP}}$"),
]

fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
for ax, (key, label) in zip(axes, panels, strict=True):
    data = grid[key]
    im = ax.imshow(
        data,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=[0, len(lambdas), 0, len(alphas)],
    )
    if key == "rho":
        # criticality contour
        cs = ax.contour(
            [i + 0.5 for i in range(len(lambdas))],
            [i + 0.5 for i in range(len(alphas))],
            data,
            levels=[1.0],
            colors="red",
            linewidths=1.5,
        )
        ax.clabel(cs, fmt={1.0: r"$\rho=1$"}, fontsize=8)
    ax.set_xticks([i + 0.5 for i in range(len(lambdas))])
    ax.set_xticklabels([f"{v:g}" for v in lambdas], fontsize=7, rotation=45)
    ax.set_yticks([i + 0.5 for i in range(0, len(alphas), 2)])
    ax.set_yticklabels([f"{alphas[i]:g}" for i in range(0, len(alphas), 2)], fontsize=7)
    ax.set_title(label, fontsize=11)
    ax.set_xlabel(r"$\lambda$", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.85)
axes[0].set_ylabel(r"$\alpha$", fontsize=10)
fig.savefig(HERE / "figures" / "phase_map.pdf")
print("wrote figures/phase_map.pdf")
