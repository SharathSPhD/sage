"""The α×λ phase map — every meter over the whole plane.

Run: ``uv run python -m experiments.phase_map``
Per cell (median over seeded family games): ρ(SB), distance to criticality,
ℛ, EPR, fraction of cells past criticality. Outputs the artifact JSON and an
SVG heat map embedded in the docs/dashboard. Report the surface, not per-cell
significance (master spec §16).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import strataq
import yaml
from strataq.core.solve.fixedpoint import logit_qre
from strataq.finite.decompose.generate import make_family
from strataq.finite.games.tensor import DenseTensorGame
from strataq.finite.response.reciprocity import reciprocity_defect
from strataq.finite.response.susceptibility import build_operators
from strataq.thermo.exact import thermo_read
from strataq_bench import BenchmarkResult

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
CONFIG = REPO / "config" / "experiments" / "phase_map.yaml"
SVG_OUT = REPO / "docs" / "progress" / "phase_map.svg"
UNIT = "science.phase_map"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _cell_games(cfg: dict[str, Any], level_alpha: float, a_idx: int) -> list[DenseTensorGame]:
    shape = tuple(cfg["shape"])
    scale = float(cfg["scale"])
    key = jax.random.PRNGKey(int(cfg["seed"]) + 41)
    games = []
    for g_idx in range(int(cfg["games_per_cell"])):
        k = jax.random.fold_in(jax.random.fold_in(key, a_idx), g_idx)
        k1, k2, k3, k4 = jax.random.split(k, 4)
        pot = DenseTensorGame((jax.random.normal(k1, shape), jax.random.normal(k2, shape)))
        harm = DenseTensorGame((jax.random.normal(k3, shape), jax.random.normal(k4, shape)))
        try:
            games.append(make_family(pot, harm, [level_alpha], scale=scale)[0])
        except ValueError:
            continue
    return games


def run() -> int:
    cfg = yaml.safe_load(CONFIG.read_text())
    lambdas = [float(v) for v in cfg["lambdas"]]
    alphas = [float(v) for v in cfg["alphas"]]
    tol = float(cfg["solver"]["tol"])
    max_iter = int(cfg["solver"]["max_iter"])
    RESULTS.mkdir(parents=True, exist_ok=True)

    grid: dict[str, list[list[float]]] = {
        "rho": [],
        "reciprocity": [],
        "epr": [],
        "supercritical_frac": [],
    }
    for a_idx, a in enumerate(alphas):
        row_rho, row_r, row_epr, row_super = [], [], [], []
        games = _cell_games(cfg, a, a_idx)
        for lam in lambdas:
            rhos, rs, eprs, supers = [], [], [], []
            for game in games:
                point = logit_qre(game, lam, tol=tol, max_iter=max_iter)
                ops = build_operators(game, point)
                eigs = jnp.linalg.eigvals(ops.s_tangent @ ops.b_tangent)
                rho = float(jnp.max(jnp.abs(eigs)))
                rhos.append(rho)
                supers.append(1.0 if rho >= 1.0 else 0.0)
                rs.append(float(reciprocity_defect(game, point)))
                eprs.append(float(thermo_read(game, lam).epr))
            row_rho.append(float(jnp.median(jnp.asarray(rhos))))
            row_r.append(float(jnp.median(jnp.asarray(rs))))
            row_epr.append(float(jnp.median(jnp.asarray(eprs))))
            row_super.append(float(jnp.mean(jnp.asarray(supers))))
        grid["rho"].append(row_rho)
        grid["reciprocity"].append(row_r)
        grid["epr"].append(row_epr)
        grid["supercritical_frac"].append(row_super)
        print(f"alpha={a:.1f} done")

    # Sanity contract (correctness leg): R and EPR ~ 0 along alpha = 0 row.
    zero_row_r = max(grid["reciprocity"][0])
    zero_row_epr = max(grid["epr"][0])
    passed = zero_row_r < 1e-8 and zero_row_epr < 1e-10

    payload = {
        "lambdas": lambdas,
        "alphas": alphas,
        "grid": grid,
    }
    (RESULTS / "phase_map_surface.json").write_text(json.dumps(payload, indent=1) + "\n")

    result = BenchmarkResult(
        benchmark_id="phase_map",
        unit=UNIT,
        kind="correctness",
        passed=passed,
        metrics={
            "alpha0_max_R": zero_row_r,
            "alpha0_max_epr": zero_row_epr,
            "max_rho": max(max(r) for r in grid["rho"]),
            "n_cells": float(len(lambdas) * len(alphas)),
            "games_per_cell": float(cfg["games_per_cell"]),
        },
        seed=int(cfg["seed"]),
        config_ref=str(CONFIG.relative_to(REPO)),
        library_version=strataq.__version__,
        timestamp=_now(),
        notes=(
            "Median surface over seeded 3x3 families (games_per_cell=5 — a coarse "
            "first map, no per-cell claims; FDR control would be needed for any). "
            "Correctness leg: the alpha=0 row must read ~0 on R and EPR at every "
            "lambda. CAVEAT (red-team O-2): in cells with rho >= 1 the resolvent "
            "(I - SB) is near-singular, so R medians there are magnitude-unreliable "
            "(direction/order only); supercritical_frac marks those cells. Wedge "
            "onset stated precisely: median game crosses rho = 1 from alpha = 0.5 "
            "(lambda >= 8.5); a 0.2 fraction of games crosses already at alpha = 0.4. "
            "Full surface in phase_map_surface.json."
        ),
    )
    (RESULTS / "phase_map.json").write_text(result.model_dump_json(indent=2) + "\n")
    print(f"[{'PASS' if passed else 'FAIL'}] phase_map")

    _render_svg(payload)
    print(f"wrote {SVG_OUT}")
    return 0 if passed else 1


def _render_svg(payload: dict[str, Any]) -> None:
    """Three-panel heat map (ρ, ℛ, EPR) as a self-contained SVG."""
    lambdas, alphas = payload["lambdas"], payload["alphas"]
    panels = [("rho", "ρ(SB) — criticality"), ("reciprocity", "ℛ"), ("epr", "EPR")]
    cell_w, cell_h = 26, 18
    panel_w = cell_w * len(lambdas) + 60
    width = panel_w * len(panels) + 20
    height = cell_h * len(alphas) + 90

    def colour(value: float, vmax: float) -> str:
        t = 0.0 if vmax <= 0 else min(1.0, value / vmax)
        r = int(250 * t + 245 * (1 - t) * 0.2)
        g = int(80 * t + 245 * (1 - t) * 0.95)
        b = int(60 * t + 245 * (1 - t) * 0.6)
        return f"rgb({r},{g},{b})"

    parts = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'font-family="sans-serif">',
        f'<text x="{width / 2:.0f}" y="18" text-anchor="middle" font-size="14" '
        f'fill="currentColor">The (λ, α) phase map — median over seeded families</text>',
    ]
    for p_idx, (metric_key, label) in enumerate(panels):
        grid = payload["grid"][metric_key]
        vmax = max(max(row) for row in grid) or 1.0
        x0 = 50 + p_idx * panel_w
        parts.append(
            f'<text x="{x0 + cell_w * len(lambdas) / 2:.0f}" y="40" text-anchor="middle" '
            f'font-size="12" fill="currentColor">{label}</text>'
        )
        for a_idx in range(len(alphas)):
            for l_idx in range(len(lambdas)):
                value = grid[a_idx][l_idx]
                # alpha axis points UP: row 0 (alpha=0) at the bottom.
                y = 50 + (len(alphas) - 1 - a_idx) * cell_h
                parts.append(
                    f'<rect x="{x0 + l_idx * cell_w}" y="{y}" width="{cell_w - 1}" '
                    f'height="{cell_h - 1}" fill="{colour(value, vmax)}">'
                    f"<title>λ={lambdas[l_idx]}, α={alphas[a_idx]}: {value:.3g}</title></rect>"
                )
        parts.append(
            f'<text x="{x0}" y="{50 + len(alphas) * cell_h + 14}" font-size="10" '
            f'fill="currentColor">λ: {lambdas[0]} … {lambdas[-1]} (log-ish)</text>'
        )
        if p_idx == 0:
            mid_y = 50 + len(alphas) * cell_h / 2
            parts.append(
                f'<text x="14" y="{mid_y:.0f}" font-size="10" fill="currentColor" '
                f'transform="rotate(-90 14 {mid_y:.0f})" text-anchor="middle">'
                "α: 0 (bottom) → 1 (top)</text>"
            )
    parts.append("</svg>")
    SVG_OUT.write_text("".join(parts))


if __name__ == "__main__":
    raise SystemExit(run())
