"""R11 — two kill-shots on the flagship two-coordinate (ℛ, EPR) claim.

The programme claims that response asymmetry ℛ (a LOCAL derivative at the QRE
point, from χ^eq = (I − SB)⁻¹S) and the Schnakenberg EPR of the Glauber chain
(a GLOBAL stationary-flux functional) share a zero but are otherwise
INDEPENDENT coordinates. The whole evidential base for "otherwise independent"
is the within-α-level collapse of ρ_S(EPR, ℛ) recorded in
``benchmarks/results/chain_comovement.json`` (F-0004) and the refuted repair in
``decoupling_mechanism.json`` (F-0007) — and every reading is at N=2, m=3.

Two ways that could be wrong, both tested here:

K2  FINITE SIZE — the collapse is an artefact of a 4-dimensional tangent space
    and a 9-state chain, and recovers as m grows. Swept over m ∈ {3,4,5,6} at
    α ∈ {0.05, 0.45, 0.75, 0.85, 0.95}, with a registered scale control (D1)
    because a fixed Frobenius scale makes larger m a COLDER game.
K3  SOLVER — the collapse is an artefact of the damped fixed-point solver. The
    α ∈ {0.85, 0.95} cells at m=3 are recomputed on the IDENTICAL games with
    magnetic mirror descent and with the damped solver at 100× tighter
    tolerance. EPR does not depend on the solver at all, so the solver can
    only enter this correlation through ℛ.

REGISTRATION HISTORY — READ THIS BEFORE READING ANY NUMBER BELOW.

Criteria K2-T1…T4 and K3-T1 are registered in
``config/experiments/plane_robustness.yaml``, committed at b89d5df (2026-08-12
23:26:36) before this file existed. That is NOT the whole story, and the
partial story was the finding of the R11 red-team review (objection O-1):

* An EARLIER registration of the SAME unit and the SAME K2 kill-shot,
  ``config/experiments/plane_finite_size.yaml``, was committed at a6533e7
  (2026-08-12 23:02:25).
* ``experiments/plane_finite_size.py`` was written at 23:04:39 and EXECUTED at
  23:04:49 (``experiments/__pycache__/plane_finite_size.cpython-311.pyc``); a
  second run started at 23:17:47
  (``benchmarks/results/plane_finite_size.resolved.yaml``).
* Only afterwards, at 23:26:36, was ``plane_robustness.yaml`` committed with a
  DIFFERENT primary statistic for the same question, and the change ran in the
  direction of survival:
      first registration : refute iff rho_hi(6) − rho_hi(3) >  0.40
                           survive iff rho_hi(6) − rho_hi(3) <= 0.20
      replacement        : refute iff gap_ratio(6) <  0.50
                           survive iff gap_ratio(m) >= 0.50 for all m
  On this run's data rho_hi(6) − rho_hi(3) = +0.3639, so the FIRST registration
  returns INDETERMINATE while the replacement returns SURVIVES.

This script therefore adjudicates BOTH registrations on the same data and
reports both. The headline K2-T1 verdict is INDETERMINATE: where two
registrations of one criterion disagree and the earlier one was live when the
data were first produced, the earlier one binds. No threshold in either config
has been changed.

POST-HOC ADDITIONS (made after the result existed, in response to the red team,
and marked as such wherever they appear):

* O-2 — ``gap_ratio`` is the deciding statistic of the replacement
  registration, but it is not a rho_S, so the registered "bootstrap CIs on
  every reported rho_S" never gave it an interval. Its CI is computed here
  (:func:`_bootstrap_gap_ratio`) and it CROSSES the 0.50 bar.
* O-3 — the registered ``n_justification`` says "5600 games solved in total".
  That is wrong in both directions; the corrected decomposition is appended to
  the artifact's ``n_justification``.
* the ``passed`` flag now follows the house convention (see
  ``smalln_certification.json``, ``decoupling_mechanism.json``): passed=True
  means the registered adjudication ran and no guard fired. The verdict lives
  in ``metrics``/``notes``.

Run: ``uv run python -m experiments.plane_robustness``
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import strataq
import yaml
from strataq.core.solve.fixedpoint import logit_qre
from strataq.core.solve.mirror import logit_qre_mirror
from strataq.core.types import QREPoint
from strataq.finite.decompose.generate import make_family
from strataq.finite.games.tensor import DenseTensorGame
from strataq.finite.response.reciprocity import reciprocity_defect_of
from strataq.finite.response.susceptibility import chi_equilibrium
from strataq.thermo.exact import thermo_read
from strataq_bench import BenchmarkResult, EffectSize

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
CONFIG = REPO / "config" / "experiments" / "plane_robustness.yaml"
# The SUPERSEDED registration of the same kill-shot (commit a6533e7). Read, not
# written: this file is the record of what the criterion was when the first two
# runs were launched, and it is adjudicated on this run's data alongside the
# replacement so the substitution is visible in the artifact itself.
SUPERSEDED_CONFIG = REPO / "config" / "experiments" / "plane_finite_size.yaml"
UNIT = "science.plane"

# POST-HOC (red team O-2). Not a registered threshold and not a criterion: a
# resample count for an interval the registration failed to demand on its own
# deciding statistic. 4000 resamples put the Monte-Carlo error at ~1.6% of the
# CI width; the interval it produces is reported, never adjudicated against.
GAP_RATIO_RESAMPLES = 4000


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def _spearman(x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Rank correlation — byte-identical helper to ``decoupling_mechanism.py``."""

    def ranks(v: jnp.ndarray) -> jnp.ndarray:
        order = jnp.argsort(v)
        return jnp.zeros_like(v).at[order].set(jnp.arange(v.shape[0], dtype=v.dtype))

    rx = ranks(x) - (x.shape[0] - 1) / 2.0
    ry = ranks(y) - (y.shape[0] - 1) / 2.0
    return float((rx @ ry) / jnp.sqrt((rx @ rx) * (ry @ ry)))


def _spearman_rows(xs: jnp.ndarray, ys: jnp.ndarray) -> jnp.ndarray:
    """Row-wise Spearman for a batch of resamples.

    Identical arithmetic to :func:`_spearman` applied per row (argsort ranks,
    ties broken by index, both stable) — vectorised only so that 2000
    resamples × 28 cells is seconds rather than minutes.
    """
    n = xs.shape[1]
    rows = jnp.arange(xs.shape[0])[:, None]
    positions = jnp.broadcast_to(jnp.arange(n, dtype=xs.dtype), xs.shape)

    def ranks(v: jnp.ndarray) -> jnp.ndarray:
        order = jnp.argsort(v, axis=1)
        return jnp.zeros_like(v).at[rows, order].set(positions)

    rx = ranks(xs) - (n - 1) / 2.0
    ry = ranks(ys) - (n - 1) / 2.0
    num = jnp.sum(rx * ry, axis=1)
    den = jnp.sqrt(jnp.sum(rx * rx, axis=1) * jnp.sum(ry * ry, axis=1))
    return num / den


def _bootstrap_ci(
    x: jnp.ndarray, y: jnp.ndarray, *, key: jnp.ndarray, n_resamples: int, ci_level: float
) -> tuple[float, float]:
    """Percentile bootstrap CI on ρ_S(x, y), resampling GAMES within the cell (K2-T3).

    The (EPR, ℛ) pairing inside a game is never broken: one index vector
    selects both coordinates.
    """
    n = int(x.shape[0])
    idx = jax.random.randint(key, (n_resamples, n), 0, n)
    boot = _spearman_rows(x[idx], y[idx])
    tail = (1.0 - ci_level) / 2.0
    return float(jnp.quantile(boot, tail)), float(jnp.quantile(boot, 1.0 - tail))


def _bootstrap_gap_ratio(
    cells: dict[int, dict[float, dict[str, Any]]],
    m_values: list[int],
    *,
    a_lo: float,
    a_hi: float,
    key: jnp.ndarray,
    n_resamples: int,
    ci_level: float,
    bar: float,
) -> dict[int, dict[str, float]]:
    """POST-HOC (red team O-2): percentile bootstrap on ``gap_ratio(m)``.

    ``gap_ratio(m) = (rho_lo(m) − rho_hi(m)) / (rho_lo(m_min) − rho_hi(m_min))``
    is the deciding statistic of the replacement registration's K2-T1, and the
    registration demanded intervals only on every reported ``rho_S`` — which
    ``gap_ratio`` is not. It therefore went to adjudication as a bare point
    estimate against a 0.50 bar.

    Resampling matches the registered scheme for the rho_S intervals: GAMES
    WITHIN THE CELL, with replacement, the (EPR, ℛ) pairing inside a game never
    broken. Each replicate resamples every cell that enters the ratio and the
    SAME replicate's m_min cells form the denominator, so the numerator's and
    denominator's shared sampling variation is preserved rather than pretended
    away. ``p_below_bar`` is the bootstrap mass beneath ``bar``; it is a
    reported quantity, not a test — no threshold is adjudicated against it.
    """
    m_min = m_values[0]
    boots: dict[int, jnp.ndarray] = {}
    for m_idx, m in enumerate(m_values):
        parts = []
        for level_idx, alpha in enumerate((a_lo, a_hi)):
            cell = cells[m][alpha]
            k = jax.random.fold_in(jax.random.fold_in(key, m_idx), level_idx)
            n = int(cell["r"].shape[0])
            idx = jax.random.randint(k, (n_resamples, n), 0, n)
            parts.append(_spearman_rows(cell["epr"][idx], cell["r"][idx]))
        boots[m] = parts[0] - parts[1]  # gap(m) per replicate
    tail = (1.0 - ci_level) / 2.0
    out: dict[int, dict[str, float]] = {}
    for m in m_values:
        ratio = boots[m] / boots[m_min]
        out[m] = {
            "ci_low": float(jnp.quantile(ratio, tail)),
            "ci_high": float(jnp.quantile(ratio, 1.0 - tail)),
            "sd": float(jnp.std(ratio)),
            "p_below_bar": float(jnp.mean(ratio < bar)),
        }
    return out


# --------------------------------------------------------------------------
# game construction and reads
# --------------------------------------------------------------------------
def _sources(key: jnp.ndarray, m: int) -> tuple[DenseTensorGame, DenseTensorGame]:
    """Two independent Gaussian source games — the F-0004 draw, exactly."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    shape = (m, m)
    pot = DenseTensorGame((jax.random.normal(k1, shape), jax.random.normal(k2, shape)))
    harm = DenseTensorGame((jax.random.normal(k3, shape), jax.random.normal(k4, shape)))
    return pot, harm


def _solve(
    game: DenseTensorGame, lam: float, *, method: str, tol: float, max_iter: int
) -> QREPoint:
    if method == "mirror":
        return logit_qre_mirror(game, lam, tol=tol, max_iter=max_iter)
    return logit_qre(game, lam, tol=tol, max_iter=max_iter)


def _cell(
    *,
    stream_key: jnp.ndarray,
    level_idx: int,
    m: int,
    alpha: float,
    n_games: int,
    scale: float,
    lam: float,
    method: str,
    tol: float,
    max_iter: int,
) -> dict[str, Any]:
    """One (m, α) cell.

    PRNG threading reproduces F-0004 exactly: ``fold_in(fold_in(key,
    level_idx), game_idx)`` with ``level_idx`` indexing the CANONICAL ten-level
    list, then ``split(·, 4)``. At m = 3 and ``game_idx < 100`` these are
    literally the chain_comovement games.
    """
    rs: list[float] = []
    eprs: list[float] = []
    dists: list[float] = []
    ranges: list[float] = []
    lam_norms: list[float] = []
    near_critical = 0
    non_converged = 0
    rejected = 0
    for g_idx in range(n_games):
        k = jax.random.fold_in(jax.random.fold_in(stream_key, level_idx), g_idx)
        pot, harm = _sources(k, m)
        try:
            game = make_family(pot, harm, [alpha], scale=scale)[0]
        except ValueError:
            rejected += 1
            continue
        point = _solve(game, lam, method=method, tol=tol, max_iter=max_iter)
        non_converged += int(not bool(point.converged))
        resp = chi_equilibrium(game, point)
        near_critical += int(bool(resp.near_critical))
        dists.append(float(resp.distance_to_criticality))
        rs.append(float(reciprocity_defect_of(resp.chi_tangent)))
        eprs.append(float(thermo_read(game, lam).epr))
        ranges.append(float(point.payoff_range))
        lam_norms.append(float(point.lambda_normalised[0]))
    r_arr = jnp.asarray(rs)
    epr_arr = jnp.asarray(eprs)
    return {
        "r": r_arr,
        "epr": epr_arr,
        "rho": _spearman(epr_arr, r_arr),
        "n": int(r_arr.shape[0]),
        "near_critical_frac": near_critical / n_games,
        "non_converged": non_converged,
        "rejected": rejected,
        "median_distance_to_criticality": float(jnp.median(jnp.asarray(dists))),
        "median_payoff_range": float(jnp.median(jnp.asarray(ranges))),
        "median_lambda_normalised": float(jnp.median(jnp.asarray(lam_norms))),
        "median_epr": float(jnp.median(epr_arr)),
        "median_r": float(jnp.median(r_arr)),
    }


def _arm(
    *,
    stream_key: jnp.ndarray,
    canonical: list[float],
    m_values: list[int],
    levels: list[float],
    n_games: int,
    lam: float,
    scale_of_m: dict[int, float],
    method: str,
    tol: float,
    max_iter: int,
    label: str,
) -> dict[int, dict[float, dict[str, Any]]]:
    cells: dict[int, dict[float, dict[str, Any]]] = {}
    for m in m_values:
        cells[m] = {}
        for alpha in levels:
            cell = _cell(
                stream_key=stream_key,
                level_idx=canonical.index(alpha),
                m=m,
                alpha=alpha,
                n_games=n_games,
                scale=scale_of_m[m],
                lam=lam,
                method=method,
                tol=tol,
                max_iter=max_iter,
            )
            cells[m][alpha] = cell
            print(
                f"[{label}] m={m} alpha={alpha:.2f}: rho_S(EPR,R)={cell['rho']:+.4f}  "
                f"near_crit={cell['near_critical_frac']:.2f}  nonconv={cell['non_converged']}  "
                f"lam_norm={cell['median_lambda_normalised']:.2f}  "
                f"med_EPR={cell['median_epr']:.4f}  med_R={cell['median_r']:.4f}",
                flush=True,
            )
    return cells


# --------------------------------------------------------------------------
# adjudication — K2-T1, K2-T2, K2-T4 exactly as registered
# --------------------------------------------------------------------------
def _adjudicate_t1(
    cells: dict[int, dict[float, dict[str, Any]]],
    cis: dict[int, tuple[float, float]],
    crit: dict[str, Any],
    m_values: list[int],
) -> dict[str, Any]:
    a_lo, a_hi = float(crit["alpha_lo"]), float(crit["alpha_hi"])
    rho_hi = [cells[m][a_hi]["rho"] for m in m_values]
    rho_lo = [cells[m][a_lo]["rho"] for m in m_values]
    gap = [lo - hi for lo, hi in zip(rho_lo, rho_hi, strict=True)]
    gap_ratio = [g / gap[0] for g in gap]

    mono_tol = float(crit["mono_tol"])
    mono = all(rho_hi[k + 1] >= rho_hi[k] - mono_tol for k in range(len(rho_hi) - 1))
    ci_low = [cis[m][0] for m in m_values]
    ci_high = [cis[m][1] for m in m_values]
    ends_disjoint = ci_low[-1] > ci_high[0]
    all_overlap = max(ci_low) <= min(ci_high)

    shrink = float(crit["gap_shrink_fail"])
    ceiling = float(crit["survive_ceiling"])
    fail = mono and gap_ratio[-1] < shrink and ends_disjoint
    survive = (
        all(g >= shrink for g in gap_ratio[1:])
        and all(r <= ceiling for r in rho_hi)
        and all(h < ceiling for h in ci_high)
    )
    baseline_ok = all(r >= float(crit["baseline_rho_min"]) for r in rho_lo)
    return {
        "rho_hi": rho_hi,
        "rho_lo": rho_lo,
        "gap": gap,
        "gap_ratio": gap_ratio,
        "mono": mono,
        "ends_disjoint": ends_disjoint,
        "all_cis_overlap": all_overlap,
        "ci_high_under_ceiling": all(h < ceiling for h in ci_high),
        "t1_fail_finite_size": fail,
        "t1_survive": survive,
        "baseline_ok": baseline_ok,
    }


def _adjudicate_superseded(
    adj: dict[str, Any], sup_crit: dict[str, Any], sup_cis: dict[int, tuple[float, float]]
) -> dict[str, Any]:
    """The SUPERSEDED registration's K2-T1, on this run's data (red team O-1).

    ``config/experiments/plane_finite_size.yaml`` (commit a6533e7, live when the
    23:04:49 and 23:17:47 runs were launched) adjudicated the SAME kill-shot on
    a different primary statistic: the raw recovery ``rho_hi(m_max) −
    rho_hi(m_min)`` against 0.40 (refute) / 0.20 (survive), rather than
    ``gap_ratio(m_max)`` against 0.50. Its T2/T3/T4 structure is otherwise the
    same and its ceiling is the same 0.35.

    Nothing here is recomputed or re-tuned: the thresholds are read verbatim
    from the superseded file, which is committed and stays committed.
    """
    rho_hi = adj["rho_hi"]
    rho_lo = adj["rho_lo"]
    delta = rho_hi[-1] - rho_hi[0]
    ceiling = float(sup_crit["survive_ceiling"])
    mono_tol = float(sup_crit["mono_tol"])
    mono = all(rho_hi[k + 1] >= rho_hi[k] - mono_tol for k in range(len(rho_hi) - 1))
    ci_low = [c[0] for c in sup_cis.values()]
    ci_high = [c[1] for c in sup_cis.values()]
    ends_disjoint = ci_low[-1] > ci_high[0]
    ci_under_ceiling = all(h < ceiling for h in ci_high)
    # T2 of the superseded file: collapse(m) = rho_hi(m) − rho_lo(m) <= −0.40.
    collapse = [hi - lo for hi, lo in zip(rho_hi, rho_lo, strict=True)]
    t2_ok = all(c <= float(sup_crit["collapse_drop_max"]) for c in collapse)
    baseline_ok = all(r >= float(sup_crit["baseline_rho_min"]) for r in rho_lo)
    refute = mono and delta > float(sup_crit["recovery_delta_refute"]) and ends_disjoint
    survive = (
        all(r <= ceiling for r in rho_hi)
        and delta <= float(sup_crit["recovery_delta_survive"])
        and ci_under_ceiling
    )
    verdict = "REFUTED" if refute else ("SURVIVES" if survive else "INDETERMINATE")
    return {
        "delta_rho_hi": delta,
        "refute_bar": float(sup_crit["recovery_delta_refute"]),
        "survive_bar": float(sup_crit["recovery_delta_survive"]),
        "monotone": mono,
        "ends_disjoint": ends_disjoint,
        "ci_high_under_ceiling": ci_under_ceiling,
        "t2_collapse_ok": t2_ok,
        "baseline_ok": baseline_ok,
        "verdict": verdict,
    }


def _onset(cells: dict[float, dict[str, Any]], levels: list[float], crit: dict[str, Any]) -> float:
    """Smallest α at which ρ_S drops below ``onset_rho``; sentinel if none (K2-T2)."""
    for alpha in sorted(levels):
        if cells[alpha]["rho"] < float(crit["onset_rho"]):
            return alpha
    return float(crit["onset_none_sentinel"])


def run() -> int:
    cfg = yaml.safe_load(CONFIG.read_text())
    seed = int(cfg["seed"])
    lam = float(cfg["lam"])
    lam_ctl = float(cfg["lam_control"])
    stream = cfg["seed_stream"]
    canonical = [float(a) for a in stream["canonical_levels"]]
    levels = [float(a) for a in cfg["sweep"]["levels"]]
    m_values = [int(m) for m in cfg["sweep"]["m_values"]]
    per_cell = int(cfg["sweep"]["games_per_cell"])
    scale = float(cfg["sweep"]["scale"])
    tol_ref = float(cfg["solver"]["tol_ref"])
    tol_tight = float(cfg["solver"]["tol_tight"])
    max_iter = int(cfg["solver"]["max_iter"])
    crit = cfg["criteria"]
    boot = cfg["bootstrap"]
    n_resamples = int(boot["n_resamples"])
    ci_level = float(boot["ci_level"])
    max_near = float(cfg["diagnostics"]["max_near_critical_frac"])
    a_lo, a_hi = float(crit["alpha_lo"]), float(crit["alpha_hi"])
    ref_n = int(crit["reference_n"])

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "plane_robustness.resolved.yaml").write_text(
        yaml.safe_dump(
            {"config": cfg, "library_version": strataq.__version__, "run_at": _now()},
            sort_keys=False,
        )
    )

    main_key = jax.random.PRNGKey(int(stream["root"]) + int(stream["offset"]))

    # ---- K2 main arm: fixed scale, the F-0004 seed family ------------------
    main = _arm(
        stream_key=main_key,
        canonical=canonical,
        m_values=m_values,
        levels=levels,
        n_games=per_cell,
        lam=lam,
        scale_of_m=dict.fromkeys(m_values, scale),
        method="damped",
        tol=tol_ref,
        max_iter=max_iter,
        label="K2-main",
    )

    # ---- K2-T3: bootstrap CI on every reported rho -------------------------
    boot_key = jax.random.PRNGKey(seed + int(boot["seed_offset"]))
    main_cis: dict[int, dict[float, tuple[float, float]]] = {}
    for m_idx, m in enumerate(m_values):
        main_cis[m] = {}
        for level_idx, alpha in enumerate(levels):
            k = jax.random.fold_in(jax.random.fold_in(boot_key, m_idx), level_idx)
            cell = main[m][alpha]
            main_cis[m][alpha] = _bootstrap_ci(
                cell["epr"], cell["r"], key=k, n_resamples=n_resamples, ci_level=ci_level
            )
    hi_cis = {m: main_cis[m][a_hi] for m in m_values}
    t1_main = _adjudicate_t1(main, hi_cis, crit, m_values)

    # ---- O-1: the SUPERSEDED registration, adjudicated on the same data ------
    sup_crit = yaml.safe_load(SUPERSEDED_CONFIG.read_text())["criteria"]
    sup = _adjudicate_superseded(t1_main, sup_crit, hi_cis)
    print(
        f"[O-1] superseded registration ({SUPERSEDED_CONFIG.name}, a6533e7): "
        f"delta rho_hi = {sup['delta_rho_hi']:+.4f} vs refute>{sup['refute_bar']:.2f} / "
        f"survive<={sup['survive_bar']:.2f}  =>  {sup['verdict']}",
        flush=True,
    )

    # ---- K2-T2: collapse onset ---------------------------------------------
    onsets = {m: _onset(main[m], levels, crit) for m in m_values}
    onset_drift = {m: onsets[m] - onsets[m_values[0]] for m in m_values}
    t2_pass = all(d <= float(crit["onset_drift_tol"]) + 1e-12 for d in onset_drift.values())
    print(f"[K2-T2] onsets={onsets}  drift={onset_drift}  pass={t2_pass}", flush=True)

    # ---- K2-T4(v): replication anchor against the published F-0004 numbers --
    replication: dict[str, float] = {}
    replication_ok = True
    for alpha_key, published in crit["reference_rho"].items():
        alpha = float(alpha_key)
        cell = main[3][alpha]
        rho_first = _spearman(cell["epr"][:ref_n], cell["r"][:ref_n])
        replication[alpha_key] = rho_first
        dev = abs(rho_first - float(published))
        replication_ok &= dev <= float(crit["replication_tol"])
        print(
            f"[K2-T4v] replication alpha={alpha}: first-{ref_n} rho={rho_first:+.5f} "
            f"vs published {float(published):+.5f}  |dev|={dev:.5f}",
            flush=True,
        )

    # ---- D1: registered scale control, constant per-entry payoff RMS -------
    sc = cfg["scale_control"]
    d1_levels = [float(a) for a in sc["levels"]]
    d1_n = int(sc["games_per_cell"])
    m_min = min(m_values)
    d1 = _arm(
        stream_key=jax.random.PRNGKey(seed + int(sc["seed_offset"])),
        canonical=canonical,
        m_values=m_values,
        levels=d1_levels,
        n_games=d1_n,
        lam=lam,
        scale_of_m={m: scale * m / m_min for m in m_values},
        method="damped",
        tol=tol_ref,
        max_iter=max_iter,
        label="D1-scale",
    )
    d1_key = jax.random.PRNGKey(seed + int(sc["seed_offset"]) + int(boot["seed_offset"]))
    d1_cis: dict[int, dict[float, tuple[float, float]]] = {}
    for m_idx, m in enumerate(m_values):
        d1_cis[m] = {}
        for level_idx, alpha in enumerate(d1_levels):
            k = jax.random.fold_in(jax.random.fold_in(d1_key, m_idx), level_idx)
            cell = d1[m][alpha]
            d1_cis[m][alpha] = _bootstrap_ci(
                cell["epr"], cell["r"], key=k, n_resamples=n_resamples, ci_level=ci_level
            )
    t1_d1 = _adjudicate_t1(d1, {m: d1_cis[m][a_hi] for m in m_values}, crit, m_values)

    def _verdict(adj: dict[str, Any]) -> str:
        if adj["t1_fail_finite_size"]:
            return "FAIL_FINITE_SIZE"
        if adj["t1_survive"]:
            return "SURVIVES"
        return "INDETERMINATE"

    v_main, v_d1 = _verdict(t1_main), _verdict(t1_d1)
    arms_agree = v_main == v_d1

    # ---- D2: registered lambda control at the tasked lam = 1.5 -------------
    lc = cfg["lambda_control"]
    d2_levels = [float(a) for a in lc["levels"]]
    d2_m = [int(m) for m in lc["m_values"]]
    d2 = _arm(
        stream_key=jax.random.PRNGKey(seed + int(lc["seed_offset"])),
        canonical=canonical,
        m_values=d2_m,
        levels=d2_levels,
        n_games=int(lc["games_per_cell"]),
        lam=lam_ctl,
        scale_of_m=dict.fromkeys(d2_m, scale),
        method="damped",
        tol=tol_ref,
        max_iter=max_iter,
        label="D2-lambda",
    )
    d2_key = jax.random.PRNGKey(seed + int(lc["seed_offset"]) + int(boot["seed_offset"]))
    d2_cis: dict[int, dict[float, tuple[float, float]]] = {}
    for m_idx, m in enumerate(d2_m):
        d2_cis[m] = {}
        for level_idx, alpha in enumerate(d2_levels):
            k = jax.random.fold_in(jax.random.fold_in(d2_key, m_idx), level_idx)
            cell = d2[m][alpha]
            d2_cis[m][alpha] = _bootstrap_ci(
                cell["epr"], cell["r"], key=k, n_resamples=n_resamples, ci_level=ci_level
            )

    # ---- K3: solver control on the IDENTICAL games -------------------------
    solver_cfg = cfg["solver_control"]
    k3_m = int(solver_cfg["m"])
    k3_levels = [float(a) for a in solver_cfg["levels"]]
    k3: dict[str, dict[float, dict[str, float]]] = {}
    k3_max_delta = 0.0
    for arm_name in [str(a) for a in solver_cfg["arms"]]:
        k3[arm_name] = {}
        method = "mirror" if arm_name == "mirror" else "damped"
        tol = tol_ref if arm_name == "mirror" else tol_tight
        for alpha in k3_levels:
            level_idx = canonical.index(alpha)
            ref_cell = main[k3_m][alpha]
            epr_ref = ref_cell["epr"][:ref_n]
            r_ref = ref_cell["r"][:ref_n]
            rho_ref = _spearman(epr_ref, r_ref)
            r_arm: list[float] = []
            sigma_dev = 0.0
            nonconv = 0
            for g_idx in range(ref_n):
                k = jax.random.fold_in(jax.random.fold_in(main_key, level_idx), g_idx)
                pot, harm = _sources(k, k3_m)
                game = make_family(pot, harm, [alpha], scale=scale)[0]
                p_ref = _solve(game, lam, method="damped", tol=tol_ref, max_iter=max_iter)
                p_arm = _solve(game, lam, method=method, tol=tol, max_iter=max_iter)
                nonconv += int(not bool(p_arm.converged))
                sigma_dev = max(
                    sigma_dev,
                    max(
                        float(jnp.max(jnp.abs(a - b)))
                        for a, b in zip(p_arm.sigma, p_ref.sigma, strict=True)
                    ),
                )
                r_arm.append(float(reciprocity_defect_of(chi_equilibrium(game, p_arm).chi_tangent)))
            r_arm_arr = jnp.asarray(r_arm)
            rho_arm = _spearman(epr_ref, r_arm_arr)
            delta = abs(rho_arm - rho_ref)
            k3_max_delta = max(k3_max_delta, delta)
            k3[arm_name][alpha] = {
                "rho_ref": rho_ref,
                "rho_arm": rho_arm,
                "delta_rho": delta,
                "max_sigma_dev": sigma_dev,
                "rho_r_arm_vs_r_ref": _spearman(r_arm_arr, r_ref),
                "max_abs_r_dev": float(jnp.max(jnp.abs(r_arm_arr - r_ref))),
                "non_converged": float(nonconv),
            }
            print(
                f"[K3] arm={arm_name} alpha={alpha}: rho_ref={rho_ref:+.5f} "
                f"rho_arm={rho_arm:+.5f} |delta|={delta:.5f} "
                f"max|dsigma|={sigma_dev:.2e} nonconv={nonconv}",
                flush=True,
            )
    k3_pass = k3_max_delta < float(crit["delta_rho_max"])

    # ---- O-2 (POST-HOC): intervals on gap_ratio, the deciding statistic ------
    gr_key = jax.random.PRNGKey(seed + int(boot["seed_offset"]) + 1)
    gr_main = _bootstrap_gap_ratio(
        main,
        m_values,
        a_lo=a_lo,
        a_hi=a_hi,
        key=gr_key,
        n_resamples=GAP_RATIO_RESAMPLES,
        ci_level=ci_level,
        bar=float(crit["gap_shrink_fail"]),
    )
    gr_d1 = _bootstrap_gap_ratio(
        d1,
        m_values,
        a_lo=a_lo,
        a_hi=a_hi,
        key=jax.random.fold_in(gr_key, 1),
        n_resamples=GAP_RATIO_RESAMPLES,
        ci_level=ci_level,
        bar=float(crit["gap_shrink_fail"]),
    )
    gr_d2 = _bootstrap_gap_ratio(
        d2,
        d2_m,
        a_lo=a_lo,
        a_hi=a_hi,
        key=jax.random.fold_in(gr_key, 2),
        n_resamples=GAP_RATIO_RESAMPLES,
        ci_level=ci_level,
        bar=float(crit["gap_shrink_fail"]),
    )
    for m in m_values:
        print(
            f"[O-2] gap_ratio main m={m}: {t1_main['gap_ratio'][m_values.index(m)]:.4f} "
            f"[{gr_main[m]['ci_low']:.3f}, {gr_main[m]['ci_high']:.3f}] "
            f"sd={gr_main[m]['sd']:.3f}  P(<0.50)={gr_main[m]['p_below_bar']:.3f}",
            flush=True,
        )

    # ---- K2-T4: the INDETERMINATE guards -----------------------------------
    guard_fired = any(
        cell["near_critical_frac"] > max_near or cell["non_converged"] > 0 or cell["rejected"] > 0
        for arm in (main, d1, d2)
        for per_m in arm.values()
        for cell in per_m.values()
    )
    t4 = {
        "i_nonmonotone_with_size_dependence": (not t1_main["mono"])
        and t1_main["gap_ratio"][-1] < float(crit["indeterminate_gap_ratio"]),
        "ii_all_cis_overlap_no_ceiling": t1_main["all_cis_overlap"]
        and not t1_main["ci_high_under_ceiling"],
        "iii_baseline_absent": not t1_main["baseline_ok"],
        "iv_diagnostic_guard": guard_fired,
        "v_replication_anchor_failed": not replication_ok,
    }
    indeterminate = any(t4.values())

    if indeterminate:
        replacement_verdict = "INDETERMINATE"
    elif not arms_agree:
        replacement_verdict = "INDETERMINATE"  # registered D1 disagreement rule
    elif v_main == "FAIL_FINITE_SIZE":
        replacement_verdict = "FAIL_FINITE_SIZE"
    elif v_main == "SURVIVES":
        replacement_verdict = "SURVIVES-NARROWED" if not t2_pass else "SURVIVES"
    else:
        replacement_verdict = "INDETERMINATE"

    # ---- HEADLINE VERDICT (red team O-1) ------------------------------------
    # Two registrations of one criterion, and they disagree on this data. The
    # earlier one (a6533e7) was live when the 23:04:49 and 23:17:47 runs were
    # launched; the later one (b89d5df) was committed after them and moved the
    # deciding statistic in the direction of survival. Where they disagree the
    # earlier binds, and the unit reports the THIRD outcome its own
    # pre-registration defined for exactly this situation.
    registrations_agree = sup["verdict"] == replacement_verdict.split("-")[0]
    verdict = replacement_verdict if registrations_agree else "INDETERMINATE"

    # ---- artifact -----------------------------------------------------------
    metrics: dict[str, float] = {
        "verdict_survives": float(verdict.startswith("SURVIVES")),
        "verdict_fail_finite_size": float(verdict == "FAIL_FINITE_SIZE"),
        "verdict_indeterminate": float(verdict == "INDETERMINATE"),
        # O-1: both registrations, on the same data.
        "registrations_agree": float(registrations_agree),
        "superseded_verdict_indeterminate": float(sup["verdict"] == "INDETERMINATE"),
        "superseded_verdict_survives": float(sup["verdict"] == "SURVIVES"),
        "superseded_verdict_refuted": float(sup["verdict"] == "REFUTED"),
        "superseded_delta_rho_hi": sup["delta_rho_hi"],
        "superseded_refute_bar": sup["refute_bar"],
        "superseded_survive_bar": sup["survive_bar"],
        "superseded_t2_collapse_ok": float(sup["t2_collapse_ok"]),
        "replacement_verdict_survives": float(replacement_verdict.startswith("SURVIVES")),
        "replacement_verdict_indeterminate": float(replacement_verdict == "INDETERMINATE"),
        "k2_t1_fail_finite_size": float(t1_main["t1_fail_finite_size"]),
        "k2_t1_survive": float(t1_main["t1_survive"]),
        "k2_t1_monotone_rho_hi": float(t1_main["mono"]),
        "k2_t1_ends_disjoint": float(t1_main["ends_disjoint"]),
        "k2_t1_all_cis_overlap": float(t1_main["all_cis_overlap"]),
        "k2_t1_ci_high_under_ceiling": float(t1_main["ci_high_under_ceiling"]),
        "k2_t1_baseline_ok": float(t1_main["baseline_ok"]),
        "k2_t2_pass": float(t2_pass),
        "k3_t1_pass": float(k3_pass),
        "k3_max_delta_rho": k3_max_delta,
        "d1_arms_agree": float(arms_agree),
        "t4_i_nonmonotone": float(t4["i_nonmonotone_with_size_dependence"]),
        "t4_ii_cis_indiscriminate": float(t4["ii_all_cis_overlap_no_ceiling"]),
        "t4_iii_baseline_absent": float(t4["iii_baseline_absent"]),
        "t4_iv_diagnostic_guard": float(t4["iv_diagnostic_guard"]),
        "t4_v_replication_failed": float(t4["v_replication_anchor_failed"]),
    }
    for m, onset in onsets.items():
        metrics[f"k2_t2_onset_m{m}"] = onset
        metrics[f"k2_t2_onset_drift_m{m}"] = onset_drift[m]
    for i, m in enumerate(m_values):
        metrics[f"main_gap_m{m}"] = t1_main["gap"][i]
        metrics[f"main_gap_ratio_m{m}"] = t1_main["gap_ratio"][i]
        metrics[f"d1_gap_m{m}"] = t1_d1["gap"][i]
        metrics[f"d1_gap_ratio_m{m}"] = t1_d1["gap_ratio"][i]
        # O-2 (post-hoc): the deciding statistic's interval.
        metrics[f"main_gap_ratio_ci_low_m{m}"] = gr_main[m]["ci_low"]
        metrics[f"main_gap_ratio_ci_high_m{m}"] = gr_main[m]["ci_high"]
        metrics[f"main_gap_ratio_sd_m{m}"] = gr_main[m]["sd"]
        metrics[f"main_gap_ratio_p_below_bar_m{m}"] = gr_main[m]["p_below_bar"]
        metrics[f"d1_gap_ratio_ci_low_m{m}"] = gr_d1[m]["ci_low"]
        metrics[f"d1_gap_ratio_ci_high_m{m}"] = gr_d1[m]["ci_high"]
        metrics[f"d1_gap_ratio_p_below_bar_m{m}"] = gr_d1[m]["p_below_bar"]
    # D2 reports gap(m) at lam_control next to gap(m) at lam, as registered.
    for m in d2_m:
        metrics[f"d2_gap_m{m}"] = d2[m][a_lo]["rho"] - d2[m][a_hi]["rho"]
    metrics["d2_gap_ratio_mmax_over_mmin"] = (
        metrics[f"d2_gap_m{max(d2_m)}"] / metrics[f"d2_gap_m{min(d2_m)}"]
    )
    metrics["d2_gap_ratio_ci_low"] = gr_d2[max(d2_m)]["ci_low"]
    metrics["d2_gap_ratio_ci_high"] = gr_d2[max(d2_m)]["ci_high"]
    metrics["d2_gap_ratio_p_below_bar"] = gr_d2[max(d2_m)]["p_below_bar"]
    for alpha_key, val in replication.items():
        metrics[f"replication_first{ref_n}_rho_alpha_{alpha_key}"] = val
        metrics[f"replication_dev_alpha_{alpha_key}"] = abs(
            val - float(crit["reference_rho"][alpha_key])
        )
    for arm_name, arm_cells, arm_cis in (
        ("main", main, main_cis),
        ("d1", d1, d1_cis),
        ("d2", d2, d2_cis),
    ):
        for m, per_m in arm_cells.items():
            for alpha, cell in per_m.items():
                pre = f"{arm_name}_m{m}_a{alpha:.2f}"
                metrics[f"{pre}_rho_epr_r"] = cell["rho"]
                metrics[f"{pre}_ci_low"] = arm_cis[m][alpha][0]
                metrics[f"{pre}_ci_high"] = arm_cis[m][alpha][1]
                metrics[f"{pre}_n"] = float(cell["n"])
                metrics[f"{pre}_near_critical_frac"] = cell["near_critical_frac"]
                metrics[f"{pre}_non_converged"] = float(cell["non_converged"])
                metrics[f"{pre}_rejected"] = float(cell["rejected"])
                metrics[f"{pre}_median_dist_crit"] = cell["median_distance_to_criticality"]
                metrics[f"{pre}_median_lambda_normalised"] = cell["median_lambda_normalised"]
                metrics[f"{pre}_median_payoff_range"] = cell["median_payoff_range"]
                metrics[f"{pre}_median_epr"] = cell["median_epr"]
                metrics[f"{pre}_median_r"] = cell["median_r"]
    for arm_name, per_alpha in k3.items():
        for alpha, stats in per_alpha.items():
            for stat, val in stats.items():
                metrics[f"k3_{arm_name}_a{alpha:.2f}_{stat}"] = val

    effects = [
        EffectSize(
            name=f"rho_S_{arm_name}_m{m}_alpha_{alpha:.2f}",
            value=arm_cells[m][alpha]["rho"],
            ci_low=arm_cis[m][alpha][0],
            ci_high=arm_cis[m][alpha][1],
            ci_level=ci_level,
            method=(
                f"percentile bootstrap ({n_resamples} resamples, games within cell, "
                f"n={arm_cells[m][alpha]['n']})"
            ),
        )
        for arm_name, arm_cells, arm_cis in (
            ("main", main, main_cis),
            ("scale_control", d1, d1_cis),
            ("lambda_control", d2, d2_cis),
        )
        for m in arm_cells
        for alpha in arm_cells[m]
    ]
    # POST-HOC (red team O-2). gap_ratio is the statistic K2-T1's refutation
    # branch is actually decided on, and it reached adjudication without an
    # interval because the registration asked for intervals on every rho_S and
    # gap_ratio is not one. gap_ratio(m_min) = 1 by construction, so its
    # interval is degenerate and is reported as such rather than omitted.
    effects += [
        EffectSize(
            name=f"gap_ratio_{arm_name}_m{m}",
            value=gap_ratios[m_list.index(m)],
            ci_low=gr[m]["ci_low"],
            ci_high=gr[m]["ci_high"],
            ci_level=ci_level,
            method=(
                f"POST-HOC percentile bootstrap ({GAP_RATIO_RESAMPLES} resamples, games "
                f"within cell, both alpha cells resampled per replicate, m={min(m_list)} "
                "as the shared denominator); NOT registered — added after the run in "
                "response to red-team objection O-2"
            ),
        )
        for arm_name, m_list, gap_ratios, gr in (
            ("main", m_values, t1_main["gap_ratio"], gr_main),
            ("scale_control", m_values, t1_d1["gap_ratio"], gr_d1),
            (
                "lambda_control",
                d2_m,
                [metrics[f"d2_gap_m{m}"] / metrics[f"d2_gap_m{min(d2_m)}"] for m in d2_m],
                gr_d2,
            ),
        )
        for m in m_list
    ]

    # n (red team O-3). The registered n_justification's "5600 games solved in
    # total" is wrong in both directions and the artifact must not repeat it.
    # DISTINCT GAMES DRAWN: main 4x5x200 = 4000, D1 4x2x100 = 800, D2 2x2x100 =
    # 400  =>  5200. SOLVES: those 5200 once each, plus K3's 2 arms x 2 levels x
    # 100 games x 2 solves (each K3 game is solved by the reference solver AND
    # the arm solver)  =>  6000. The 400 K3 "games" are RE-DRAWS of games
    # already inside the main m=3, alpha in {0.85, 0.95} cells — by design,
    # since K3's criterion is a paired comparison on identical games — so they
    # add solves and comparisons, not sample size.
    n_distinct_games = (
        per_cell * len(levels) * len(m_values)
        + d1_n * len(d1_levels) * len(m_values)
        + int(lc["games_per_cell"]) * len(d2_levels) * len(d2_m)
    )
    n_k3_solves = 2 * ref_n * len(k3_levels) * len(solver_cfg["arms"])
    n_total = n_distinct_games
    metrics["n_distinct_games_drawn"] = float(n_distinct_games)
    metrics["n_solves_total"] = float(n_distinct_games + n_k3_solves)
    metrics["n_k3_redraws_of_main_m3_games"] = float(
        ref_n * len(k3_levels) * len(solver_cfg["arms"])
    )
    # HOUSE CONVENTION (smalln_certification.json, decoupling_mechanism.json):
    # passed = "the registered adjudication ran and no guard fired". The VERDICT
    # lives in metrics/notes. A successful kill-shot, or an INDETERMINATE, is a
    # legitimate scientific outcome and must not be recorded as a failed run;
    # only an uninterpretable one (a diagnostic guard, or a broken replication
    # anchor that makes every number meaningless) is passed=False.
    adjudication_ran = True
    passed = adjudication_ran and not (
        t4["iv_diagnostic_guard"] or t4["v_replication_anchor_failed"]
    )
    result = BenchmarkResult(
        benchmark_id="plane_robustness",
        unit=UNIT,
        kind="statistical",
        passed=passed,
        metrics=metrics,
        effect_sizes=effects,
        n=n_total,
        n_justification=(
            " ".join(str(cfg["n_justification"]).split())
            + " CORRECTION, post-run (red-team objection O-3): the registered text above "
            "says '5600 games solved in total'. That number is wrong in both directions "
            "and n is NOT 5600. Distinct games drawn: 4000 (main, 4 m x 5 levels x 200) "
            f"+ 800 (D1) + 400 (D2) = {n_distinct_games}, and that is the n recorded in "
            "this artifact. Total solves: those "
            f"{n_distinct_games} once each, plus {n_k3_solves} in K3 — 2 arms x 2 levels "
            "x 100 games x 2 solves per game, because each K3 game is solved by BOTH the "
            "reference damped solver and the arm solver so the comparison is paired — "
            f"= {n_distinct_games + n_k3_solves} solves. The 400 K3 'games' are RE-DRAWS "
            "of games already counted in the main m=3, alpha in {0.85, 0.95} cells (the "
            "same seed family, by design, since K3's 0.05 bar is only meaningful on "
            "identical games), so they contribute solves and paired comparisons but no "
            "additional sample. The registered figure added the 400 re-draws as if they "
            "were new games while omitting the 400 duplicate reference solves; the two "
            "errors do not cancel."
        ),
        seed=seed,
        config_ref=str(CONFIG.relative_to(REPO)),
        library_version=strataq.__version__,
        timestamp=_now(),
        notes=(
            f"R11 kill-shots K2 (finite size) and K3 (solver) on the two-coordinate "
            f"(R, EPR) claim. K2-T1 HEADLINE VERDICT: {verdict}. "
            "CRITERION SUBSTITUTION, disclosed on the record (red-team O-1): this unit "
            "has TWO registrations of the same K2 kill-shot. The first, "
            "config/experiments/plane_finite_size.yaml, was committed at a6533e7 "
            "(2026-08-12 23:02:25); experiments/plane_finite_size.py was written at "
            "23:04:39 and EXECUTED at 23:04:49, and a second run started at 23:17:47 "
            "(benchmarks/results/plane_finite_size.resolved.yaml). The replacement, "
            f"{CONFIG.name}, was committed at b89d5df at 23:26:36 — AFTER those runs — "
            "and changed the deciding statistic from rho_hi(6) - rho_hi(3) (refute > "
            "0.40, survive <= 0.20) to gap_ratio(6) (refute < 0.50, survive >= 0.50 at "
            "every m), in the direction of survival. On this data delta rho_hi = "
            f"{sup['delta_rho_hi']:+.4f}, which is neither > 0.40 nor <= 0.20: the FIRST "
            f"registration returns {sup['verdict']} while the replacement returns "
            f"{replacement_verdict}. The registrations disagree, the earlier one was live "
            "when the data were first produced, so the headline verdict is INDETERMINATE "
            "— the third outcome both registrations defined in advance for exactly this "
            "case. The first registration's other tests hold on this data (collapse(m) "
            f"<= -0.40 at every m: {sup['t2_collapse_ok']}); only its primary fails to "
            "resolve. No threshold in either config has been changed, and no config has "
            "been amended after a result existed — the substitution was a NEW file, "
            "which is why it is disclosed here rather than being invisible in a diff. "
            "O-2 (POST-HOC ADDITION, made after the result existed): gap_ratio is the "
            "statistic the refutation branch is decided on and it had no interval, "
            "because the registration demanded intervals on every reported rho_S and "
            "gap_ratio is not a rho_S. Bootstrapped now "
            f"({GAP_RATIO_RESAMPLES} resamples, games within cell): gap_ratio(m=6) = "
            f"{t1_main['gap_ratio'][-1]:.4f}, 95% CI "
            f"[{gr_main[m_values[-1]]['ci_low']:.3f}, "
            f"{gr_main[m_values[-1]]['ci_high']:.3f}], "
            f"P(gap_ratio < 0.50) = {gr_main[m_values[-1]]['p_below_bar']:.2f}, "
            f"margin {(t1_main['gap_ratio'][-1] - 0.50) / gr_main[m_values[-1]]['sd']:.2f} "
            "bootstrap SD. The interval CROSSES the 0.50 bar, so the surviving branch "
            "of the replacement registration is not certified by its own T3 principle "
            "('a point estimate under the ceiling with a CI crossing it does not certify "
            "survival'), which was written for rho_S and applies here in substance. "
            "WHAT IS ROBUST AND CARRIES THE CLAIM: the CEILING criterion. High-alpha "
            "coupling never recovers at any tested m — worst ci_high = "
            f"{max(main_cis[m][a_hi][1] for m in m_values):+.4f} against the 0.35 "
            "ceiling — and the alpha=0.05 and alpha=0.95 intervals are disjoint at every "
            "m. Two-coordinate independence needs decorrelation, and decorrelation is "
            f"what is established. K3-T1: {'PASS' if k3_pass else 'FAIL'} "
            f"(max |delta rho_S| = {k3_max_delta:.5f} against the registered 0.05 bar) — "
            "BUT REPORT THIS AS A DIAGNOSTIC, NOT A KILL-SHOT PASSED WITH MARGIN. "
            "Spearman is rank-based and the two solver arms perturb R by ~1e-12, far "
            "below any gap between adjacent R values in a 100-game sample, so NO "
            "reordering was arithmetically possible and |delta rho| = 0 was forced "
            "before the run rather than discovered by it. The informative content of K3 "
            "is the diagnostic it reports: the solver contributes ~1e-12 to R "
            f"(max sup-norm |d sigma| and max |d R| are recorded in the k3_* metrics). "
            f"K2-T2 onset drift: {'PASS' if t2_pass else 'FAIL'} (onsets {onsets}). "
            "rho_hi(m) [alpha=0.95] = "
            + ", ".join(
                f"m={m}: {main[m][a_hi]['rho']:+.4f} "
                f"[{main_cis[m][a_hi][0]:+.4f}, {main_cis[m][a_hi][1]:+.4f}]"
                for m in m_values
            )
            + "; rho_lo(m) [alpha=0.05] = "
            + ", ".join(f"m={m}: {main[m][a_lo]['rho']:+.4f}" for m in m_values)
            + "; gap_ratio(m) = "
            + ", ".join(
                f"m={m}: {g:.3f}" for m, g in zip(m_values, t1_main["gap_ratio"], strict=True)
            )
            + f". D1 scale control verdict {v_d1} (arms agree: {arms_agree}); D1 is "
            "UNPAIRED to the main arm (different seed offset, 100 vs 200 games/cell) and "
            "does NOT hold lambda_normalised constant — it drives lambda_norm +40% while "
            "the main arm drives it -32%, so the two arms BRACKET the confound rather "
            "than controlling it, and D1 should be re-run paired. "
            f"lam = {lam} matches the reference artifact chain_comovement.json (NOT 1.5, "
            f"which the tasking misquoted); D2 (lam = {lam_ctl}, the tasked value) has "
            f"its own gap ratio {metrics['d2_gap_ratio_mmax_over_mmin']:.3f}, a "
            f"{100 * (metrics['d2_gap_ratio_mmax_over_mmin'] - 0.5):.1f}-point margin, "
            "and is likewise unpaired. "
            f"passed={passed} follows the house convention (the registered adjudication "
            "ran and no guard fired); it is NOT a verdict."
        ),
    )
    (RESULTS / "plane_robustness.json").write_text(result.model_dump_json(indent=2) + "\n")
    print(
        f"[{'ADJUDICATED' if passed else 'GUARD FIRED'}] plane_robustness  "
        f"K2 HEADLINE={verdict}  (superseded registration: {sup['verdict']}; "
        f"replacement registration: {replacement_verdict})  "
        f"K2-T2={'PASS' if t2_pass else 'FAIL'}  K3-T1={'PASS' if k3_pass else 'FAIL'} "
        f"(forced, see notes)  n={n_total} distinct games, "
        f"{n_distinct_games + n_k3_solves} solves",
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(run())
