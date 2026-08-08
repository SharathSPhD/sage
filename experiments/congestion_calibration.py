"""Congestion calibration — the α = 0 anchor on real network data.

Run: ``uv run python -m experiments.congestion_calibration``
Two artifacts:
- population_identities: Beckmann-gradient identity, Fisk KKT, DF symmetry,
  toll-χ vs finite differences on synthetic networks (unit population.core);
- sioux_falls_calibration: the meters on real Sioux Falls data (R = 0, KKT),
  plus the θ→∞ diagnostic against the repo's best-known UE link flows —
  reported as a diagnostic, not a gate threshold, because k-shortest route
  sets restrict route choice (documented modelling scope).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import strataq
from strataq.domains.congestion import TNTPLoader, load_best_known_flows, routing_network_from_tntp
from strataq.population.games.routing import RoutingNetwork, solve_sue
from strataq.population.response.susceptibility import (
    df_symmetry_defect,
    population_reciprocity_defect,
    toll_susceptibility,
    toll_susceptibility_fd,
)
from strataq_bench import BenchmarkResult

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
SEED = 20260808
THETA = 0.5


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _write(result: BenchmarkResult) -> None:
    (RESULTS / f"{result.benchmark_id}.json").write_text(result.model_dump_json(indent=2) + "\n")
    print(f"[{'PASS' if result.passed else 'FAIL'}] {result.benchmark_id}")


def _braess() -> RoutingNetwork:
    incidence = jnp.array([[1, 1, 0, 0, 0], [0, 0, 1, 1, 0], [1, 0, 0, 1, 1]], dtype=jnp.float64)
    return RoutingNetwork(
        incidence=incidence,
        od_index=jnp.zeros(3, dtype=jnp.int32),
        demand=jnp.array([6.0]),
        free_flow=jnp.array([1.0, 45.0, 45.0, 1.0, 1.0]),
        b_coeff=jnp.array([10.0, 0.0, 0.0, 10.0, 0.0]),
        capacity=jnp.ones(5),
        power=jnp.ones(5),
    )


def run() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    failures = 0

    # ---- 1. population_identities (unit population.core) --------------------
    net = _braess()
    x_probe = jnp.linspace(1.0, 2.0, net.n_routes)
    beckmann_gap = float(
        jnp.max(jnp.abs(jax.grad(net.beckmann)(x_probe) - net.route_costs(x_probe)))
    )
    x, _, _ = solve_sue(net, 2.0, tol=1e-14)
    g = net.route_costs(x) + jnp.log(x) / 2.0
    kkt_spread = float(jnp.max(g) - jnp.min(g))
    df_defect = df_symmetry_defect(net, x)
    chi, _ = toll_susceptibility(net, 1.2)
    fd_gap = float(jnp.max(jnp.abs(chi - toll_susceptibility_fd(net, 1.2))))
    r_synth = population_reciprocity_defect(net, 1.5)

    passed = (
        beckmann_gap < 1e-12
        and kkt_spread < 1e-10
        and df_defect < 1e-14
        and fd_gap < 1e-5
        and r_synth < 1e-10
    )
    failures += not passed
    _write(
        BenchmarkResult(
            benchmark_id="population_identities",
            unit="population.core",
            kind="correctness",
            passed=passed,
            metrics={
                "beckmann_grad_gap": beckmann_gap,
                "fisk_kkt_spread": kkt_spread,
                "df_symmetry_defect": df_defect,
                "toll_chi_fd_gap": fd_gap,
                "reciprocity_braess": r_synth,
            },
            seed=SEED,
            library_version=strataq.__version__,
            timestamp=_now(),
            notes="Engine 2 exact identities on the Braess diamond (K8 tier).",
        )
    )

    # ---- 2. sioux_falls_calibration (unit domains.congestion) ----------------
    tntp = TNTPLoader().load()
    od_pairs = sorted(tntp.demand, key=tntp.demand.get, reverse=True)[:8]
    network = routing_network_from_tntp(tntp, od_pairs, k_routes=3)

    x, _, _ = solve_sue(network, THETA, tol=1e-12)
    costs = network.route_costs(x)
    kkt = 0.0
    for od in range(network.n_od):
        idx = jnp.where(network.od_index == od)[0]
        g = costs[idx] + jnp.log(x[idx]) / THETA
        kkt = max(kkt, float(jnp.max(g) - jnp.min(g)))
    r_real = population_reciprocity_defect(network, THETA)
    df_real = df_symmetry_defect(network, x)

    # Diagnostic (not gated): theta -> large vs best-known UE, same OD subset.
    x_ue, _, _ = solve_sue(network, 50.0, tol=1e-10)
    link_flow_model = network.link_flows(x_ue)
    best = load_best_known_flows()
    # Compare only links our restricted assignment actually uses.
    diag_gaps = []
    for l_idx, (a, b) in enumerate(zip(tntp.init_node, tntp.term_node, strict=True)):
        model_v = float(link_flow_model[l_idx])
        if model_v > 1.0 and (a, b) in best:
            diag_gaps.append(abs(model_v - best[(a, b)]) / best[(a, b)])
    ue_diag = float(jnp.mean(jnp.asarray(diag_gaps))) if diag_gaps else float("nan")

    passed = r_real < 1e-10 and df_real < 1e-12 and kkt < 1e-8
    failures += not passed
    _write(
        BenchmarkResult(
            benchmark_id="sioux_falls_calibration",
            unit="domains.congestion",
            kind="correctness",
            passed=passed,
            metrics={
                "reciprocity_defect": r_real,
                "df_symmetry_defect": df_real,
                "fisk_kkt_spread": kkt,
                "n_od_pairs": float(len(od_pairs)),
                "n_routes": float(network.n_routes),
                "ue_link_flow_rel_gap_diagnostic": ue_diag,
            },
            seed=SEED,
            library_version=strataq.__version__,
            timestamp=_now(),
            notes=(
                "The calibration standard: R = 0 and DF symmetric on real Sioux Falls "
                "data (top-8 OD pairs, k=3 route sets). The UE link-flow gap vs the "
                "repo's best-known flows is a DIAGNOSTIC ONLY: our restricted route "
                "sets and OD subset make exact agreement impossible by construction; "
                "the number contextualises scale, it does not certify."
            ),
        )
    )
    return failures


if __name__ == "__main__":
    raise SystemExit(run())
