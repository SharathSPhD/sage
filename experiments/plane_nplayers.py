"""R12 — the two kill-shots a referee asks for next, on the (ℛ, EPR) plane claim.

The programme claims that response asymmetry ℛ (a LOCAL derivative at the QRE
point, from χ^eq = (I − SB)⁻¹S) and the Schnakenberg EPR of the Glauber chain
(a GLOBAL stationary-flux functional of the generator) share a zero but are
otherwise INDEPENDENT coordinates. After R11 (F-0022) the finite-size criterion
in m is INDETERMINATE and what carries the claim is the CEILING: within-level
ρ_S(EPR, ℛ) at α = 0.95 never recovers above 0.35 at any tested m, with the
upper interval endpoint under the ceiling everywhere. Two things that ceiling
has never been asked:

A  N-SCALING. Every reading in the programme is N = 2. Swept over N ∈ {2, 3, 4}
   at fixed m = 3. If high-α coupling recovers above the ceiling at N > 2 with
   interval support, the two-coordinate claim is an N = 2 artefact and dies.
B  THE NUMERATOR AT LARGER m. F-0007's refutation of the ratio-artefact
   objection rests on ρ_S(‖χ−χᵀ‖, EPR) = −0.36775 at m = 3 and nothing else,
   and F-0022 showed the m = 3 SIGN of the ratio's correlation does not
   generalise. The same ceiling is applied to the NUMERATOR across
   m ∈ {3, 4, 5, 6}, on the same games as the ratio (paired).

C1 is a PAIRED, λ̄-MATCHED-BY-CONSTRUCTION control on both axes: identical seed
stream, identical game order, only ``scale`` differs, with ``scale`` set from a
DISJOINT calibration draw so median λ̄ = λ·payoff_range is held fixed across the
sweep rather than straddled. That discharges F-0022's registered follow-up (i),
which named R11's unpaired, unmatched D1 as that run's single biggest defect.
C1 is BINDING: if the arms disagree on a primary verdict, that primary is
INDETERMINATE.

REGISTRATION. Criteria A-T1, A-T2, B-T1, C1, T3 and T4 are registered in
``config/experiments/plane_nplayers.yaml``, committed at b9648f8 BEFORE this
file existed (verified with ``git ls-files`` at commit time: the config was in
the tree, this path was not). That config is and will remain the ONLY criteria
file for unit ``science.plane.nplayers`` — R11's red-team objection O-1 was a
SECOND registration of one kill-shot written after runs had been launched, and
the lesson is taken literally here. No threshold below is read from anywhere
but that file.

Run: ``uv run python -m experiments.plane_nplayers``
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
from strataq.finite.decompose.generate import make_family
from strataq.finite.decompose.hodge import hodge_decompose
from strataq.finite.games.tensor import DenseTensorGame
from strataq.finite.response.reciprocity import reciprocity_defect_of
from strataq.finite.response.susceptibility import chi_equilibrium
from strataq.thermo.exact import thermo_read
from strataq_bench import BenchmarkResult, EffectSize

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
CONFIG = REPO / "config" / "experiments" / "plane_nplayers.yaml"
UNIT = "science.plane.nplayers"

CellKey = tuple[int, int, float, float, int]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# statistics — helpers byte-identical to plane_robustness.py / decoupling_mechanism.py
# --------------------------------------------------------------------------
def _spearman(x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Rank correlation; ties broken by index, as in every prior artifact."""

    def ranks(v: jnp.ndarray) -> jnp.ndarray:
        order = jnp.argsort(v)
        return jnp.zeros_like(v).at[order].set(jnp.arange(v.shape[0], dtype=v.dtype))

    rx = ranks(x) - (x.shape[0] - 1) / 2.0
    ry = ranks(y) - (y.shape[0] - 1) / 2.0
    return float((rx @ ry) / jnp.sqrt((rx @ rx) * (ry @ ry)))


def _spearman_rows(xs: jnp.ndarray, ys: jnp.ndarray) -> jnp.ndarray:
    """Row-wise Spearman for a batch of bootstrap resamples."""
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


def _quantiles(values: jnp.ndarray, ci_level: float) -> tuple[float, float]:
    tail = (1.0 - ci_level) / 2.0
    return float(jnp.quantile(values, tail)), float(jnp.quantile(values, 1.0 - tail))


def _bootstrap_ci(
    x: jnp.ndarray, y: jnp.ndarray, *, key: jnp.ndarray, n_resamples: int, ci_level: float
) -> tuple[float, float]:
    """Percentile bootstrap CI on ρ_S(x, y), resampling GAMES within the cell (T3).

    One index vector selects both coordinates, so the per-game pairing of
    (EPR, ℛ, A, D) is never broken.
    """
    n = int(x.shape[0])
    idx = jax.random.randint(key, (n_resamples, n), 0, n)
    return _quantiles(_spearman_rows(x[idx], y[idx]), ci_level)


def _bootstrap_paired_delta(
    epr: jnp.ndarray,
    first: jnp.ndarray,
    second: jnp.ndarray,
    *,
    key: jnp.ndarray,
    n_resamples: int,
    ci_level: float,
) -> tuple[float, float]:
    """CI on ρ_S(EPR, first) − ρ_S(EPR, second) with BOTH recomputed on the same
    resampled game indices, so the shared sampling variation cancels as it does
    in the data (T3, reported-not-adjudicated contrasts)."""
    n = int(epr.shape[0])
    idx = jax.random.randint(key, (n_resamples, n), 0, n)
    boot = _spearman_rows(epr[idx], first[idx]) - _spearman_rows(epr[idx], second[idx])
    return _quantiles(boot, ci_level)


def _bootstrap_gap(
    cell_lo: dict[str, Any],
    cell_hi: dict[str, Any],
    stat: str,
    *,
    key: jnp.ndarray,
    n_resamples: int,
    ci_level: float,
) -> tuple[float, float]:
    """CI on gap = ρ_S(α_lo) − ρ_S(α_hi). The two cells are DIFFERENT games, so
    they are resampled independently. Reported as data; adjudicated against
    nothing (registered: the gap-shrink family of criteria is not resolvable at
    feasible n — F-0022)."""
    k_lo, k_hi = jax.random.split(key)
    boots = []
    for k, cell in ((k_lo, cell_lo), (k_hi, cell_hi)):
        n = int(cell["epr"].shape[0])
        idx = jax.random.randint(k, (n_resamples, n), 0, n)
        boots.append(_spearman_rows(cell["epr"][idx], cell[stat][idx]))
    return _quantiles(boots[0] - boots[1], ci_level)


# --------------------------------------------------------------------------
# game construction and reads
# --------------------------------------------------------------------------
def _sources(key: jnp.ndarray, m: int, n_players: int) -> tuple[DenseTensorGame, DenseTensorGame]:
    """N-player Gaussian source games.

    ``split(key, 2N)``: the first N arrays of shape ``(m,)*N`` are the potential
    source, the next N the harmonic source. At N = 2 this is exactly
    ``k1, k2, k3, k4 = split(key, 4)`` with (k1, k2) potential and (k3, k4)
    harmonic — bit-for-bit the ``dynamics_calibration.py`` draw that produced
    F-0004 and F-0007, which is what makes the replication anchors exact.
    """
    keys = jax.random.split(key, 2 * n_players)
    shape = (m,) * n_players
    pot = DenseTensorGame(tuple(jax.random.normal(k, shape) for k in keys[:n_players]))
    harm = DenseTensorGame(tuple(jax.random.normal(k, shape) for k in keys[n_players:]))
    return pot, harm


def _cell(
    *,
    stream_key: jnp.ndarray,
    level_idx: int,
    n_players: int,
    m: int,
    alpha: float,
    n_games: int,
    scale: float,
    lam: float,
    tol: float,
    max_iter: int,
) -> dict[str, Any]:
    """One (arm, N, m, α) cell: ℛ, the numerator A, the denominator D, EPR."""
    rs: list[float] = []
    nums: list[float] = []
    dens: list[float] = []
    eprs: list[float] = []
    dists: list[float] = []
    lam_bars: list[float] = []
    near_critical = 0
    non_converged = 0
    rejected = 0
    alpha_dev = 0.0
    for g_idx in range(n_games):
        k = jax.random.fold_in(jax.random.fold_in(stream_key, level_idx), g_idx)
        pot, harm = _sources(k, m, n_players)
        try:
            game = make_family(pot, harm, [alpha], scale=scale)[0]
        except ValueError:
            rejected += 1
            continue
        # T4(v): make_family's exactness is asserted for two players in its
        # docstring; at N = 3 and N = 4 it is CHECKED here, not assumed.
        alpha_dev = max(alpha_dev, abs(float(hodge_decompose(game).alpha) - alpha))
        point = logit_qre(game, lam, tol=tol, max_iter=max_iter)
        non_converged += int(not bool(point.converged))
        resp = chi_equilibrium(game, point)
        near_critical += int(bool(resp.near_critical))
        chi = resp.chi_tangent
        dists.append(float(resp.distance_to_criticality))
        rs.append(float(reciprocity_defect_of(chi)))
        nums.append(float(jnp.linalg.norm(chi - chi.T)))
        dens.append(float(jnp.linalg.norm(chi + chi.T)))
        eprs.append(float(thermo_read(game, lam).epr))
        lam_bars.append(float(point.lambda_normalised[0]))
    r_arr = jnp.asarray(rs)
    num_arr = jnp.asarray(nums)
    den_arr = jnp.asarray(dens)
    epr_arr = jnp.asarray(eprs)
    return {
        "r": r_arr,
        "num": num_arr,
        "den": den_arr,
        "epr": epr_arr,
        "rho_ratio": _spearman(epr_arr, r_arr),
        "rho_num": _spearman(epr_arr, num_arr),
        "rho_r_invden": _spearman(r_arr, 1.0 / den_arr),
        "n": int(r_arr.shape[0]),
        "scale": scale,
        "near_critical_frac": near_critical / n_games,
        "non_converged": non_converged,
        "rejected": rejected,
        "alpha_dev": alpha_dev,
        "median_distance_to_criticality": float(jnp.median(jnp.asarray(dists))),
        "median_lambda_bar": float(jnp.median(jnp.asarray(lam_bars))),
        "median_epr": float(jnp.median(epr_arr)),
        "median_r": float(jnp.median(r_arr)),
        "median_num": float(jnp.median(num_arr)),
        "median_den": float(jnp.median(den_arr)),
    }


def _calibration_range(
    *,
    calib_key: jnp.ndarray,
    level_idx: int,
    n_players: int,
    m: int,
    alpha: float,
    n_games: int,
) -> float:
    """Median payoff_range at scale = 1 on a DISJOINT seed stream (C1).

    These games are never solved and never enter any statistic; they exist only
    to set ``scale`` so that median λ̄ is fixed across the sweep by construction.
    Disjointness is the point: ``scale`` is never tuned on the games it is then
    measured on.
    """
    ranges: list[float] = []
    for g_idx in range(n_games):
        k = jax.random.fold_in(jax.random.fold_in(calib_key, level_idx), g_idx)
        pot, harm = _sources(k, m, n_players)
        game = make_family(pot, harm, [alpha], scale=1.0)[0]
        ranges.append(float(game.payoff_range))
    return float(jnp.median(jnp.asarray(ranges)))


# --------------------------------------------------------------------------
# adjudication — A-T1 / B-T1 exactly as registered
# --------------------------------------------------------------------------
def _adjudicate_ceiling(
    axis: list[int],
    reference: int,
    rho_hi: list[float],
    ci_hi: list[tuple[float, float]],
    rho_lo: list[float],
    crit: dict[str, Any],
) -> dict[str, Any]:
    """The registered ceiling criterion, in the identical CI form as R11.

    HOLDS iff ρ_hi ≤ 0.35 and ci_high(ρ_hi) < 0.35 at EVERY axis value.
    DIES  iff at SOME axis value beyond the reference, ρ_hi > 0.35 AND
          ci_low(ρ_hi) > 0.35 — interval-supported recovery above the ceiling.
    Everything else is INDETERMINATE; the branches are deliberately not
    exhaustive and there is no default to confirmation. The low-α precondition
    ρ_lo ≥ 0.55 is part of the criterion: without baseline coupling there is no
    collapse to measure, and that is INDETERMINATE, not a pass.
    """
    ceiling = float(crit["survive_ceiling"])
    floor = float(crit["baseline_rho_min"])
    anti = float(crit["strong_anticorrelation_at"])
    holds = all(r <= ceiling for r in rho_hi) and all(c[1] < ceiling for c in ci_hi)
    dies_at = [
        v
        for v, r, c in zip(axis, rho_hi, ci_hi, strict=True)
        if v != reference and r > ceiling and c[0] > ceiling
    ]
    baseline_ok = all(r >= floor for r in rho_lo)
    baseline_fail_at = [v for v, r in zip(axis, rho_lo, strict=True) if r < floor]
    strong_anti_at = [v for v, r in zip(axis, rho_hi, strict=True) if r <= anti]
    if dies_at:
        verdict = "DIES"
    elif holds and baseline_ok:
        verdict = "HOLDS"
    else:
        verdict = "INDETERMINATE"
    return {
        "verdict": verdict,
        "holds_raw": holds,
        "dies_at": dies_at,
        "baseline_ok": baseline_ok,
        "baseline_fail_at": baseline_fail_at,
        "strong_anticorrelation_at": strong_anti_at,
        "worst_ci_high": max(c[1] for c in ci_hi),
        "worst_rho_hi": max(rho_hi),
    }


def _combine(v_main: str, v_ctl: str, guard_fired: bool, t2_pass: bool | None = None) -> str:
    """Registered combination rule: guards first, then the BINDING C1 arm."""
    if guard_fired:
        return "INDETERMINATE"
    if v_main != v_ctl:
        return "INDETERMINATE"
    if v_main == "HOLDS" and t2_pass is False:
        return "HOLDS-NARROWED"
    return v_main


def _onset(
    per_alpha: dict[float, dict[str, Any]], levels: list[float], crit: dict[str, Any]
) -> float:
    """Smallest α with ρ_ratio below ``onset_rho``; sentinel if none (A-T2)."""
    for alpha in sorted(levels):
        if per_alpha[alpha]["rho_ratio"] < float(crit["onset_rho"]):
            return alpha
    return float(crit["onset_none_sentinel"])


def run() -> int:
    cfg = yaml.safe_load(CONFIG.read_text())
    seed = int(cfg["seed"])
    lam = float(cfg["lam"])
    stream = cfg["seed_stream"]
    canonical = [float(a) for a in stream["canonical_levels"]]
    levels = [float(a) for a in cfg["sweep"]["levels"]]
    per_cell = int(cfg["sweep"]["games_per_cell"])
    scale = float(cfg["sweep"]["scale"])
    a_cfg = cfg["kill_shot_a"]
    b_cfg = cfg["kill_shot_b"]
    n_players_values = [int(v) for v in a_cfg["n_players"]]
    a_m = int(a_cfg["m"])
    a_ref = int(a_cfg["reference_n_players"])
    m_values = [int(v) for v in b_cfg["m_values"]]
    b_n_players = int(b_cfg["n_players"])
    b_ref = int(b_cfg["reference_m"])
    tol = float(cfg["solver"]["tol"])
    max_iter = int(cfg["solver"]["max_iter"])
    ctl_levels = [float(a) for a in cfg["control"]["levels"]]
    calib_games = int(cfg["calibration"]["games"])
    crit = cfg["criteria"]
    boot = cfg["bootstrap"]
    n_resamples = int(boot["n_resamples"])
    ci_level = float(boot["ci_level"])
    max_near = float(cfg["diagnostics"]["max_near_critical_frac"])
    a_lo, a_hi = float(crit["alpha_lo"]), float(crit["alpha_hi"])
    ref_n = int(crit["reference_n"])

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "plane_nplayers.resolved.yaml").write_text(
        yaml.safe_dump(
            {"config": cfg, "library_version": strataq.__version__, "run_at": _now()},
            sort_keys=False,
        )
    )

    main_key = jax.random.PRNGKey(int(stream["root"]) + int(stream["offset"]))
    calib_key = jax.random.PRNGKey(seed + int(cfg["calibration"]["seed_offset"]))
    boot_key = jax.random.PRNGKey(seed + int(boot["seed_offset"]))

    # Distinct solved cells are computed ONCE. The A sweep at N = 2 and the B
    # sweep at m = 3 are the same games; so is every C1 cell at its reference
    # size, where the matched scale is exactly the main arm's. The cache is what
    # makes "paired" literal and what makes the n accounting exact.
    cache: dict[CellKey, dict[str, Any]] = {}

    def cell(n_players: int, m: int, alpha: float, cell_scale: float, label: str) -> dict[str, Any]:
        key: CellKey = (n_players, m, alpha, cell_scale, per_cell)
        if key in cache:
            return cache[key]
        out = _cell(
            stream_key=main_key,
            level_idx=canonical.index(alpha),
            n_players=n_players,
            m=m,
            alpha=alpha,
            n_games=per_cell,
            scale=cell_scale,
            lam=lam,
            tol=tol,
            max_iter=max_iter,
        )
        cache[key] = out
        print(
            f"[{label}] N={n_players} m={m} a={alpha:.2f} scale={cell_scale:.4f}: "
            f"rho(EPR,R)={out['rho_ratio']:+.4f} rho(EPR,NUM)={out['rho_num']:+.4f} "
            f"lam_bar={out['median_lambda_bar']:.3f} near_crit={out['near_critical_frac']:.3f} "
            f"nonconv={out['non_converged']} rej={out['rejected']} "
            f"da={out['alpha_dev']:.1e}",
            flush=True,
        )
        return out

    # ---- kill-shot A: main arm, fixed scale --------------------------------
    a_main: dict[int, dict[float, dict[str, Any]]] = {}
    for n_players in n_players_values:
        a_main[n_players] = {a: cell(n_players, a_m, a, scale, "A-main") for a in levels}

    # ---- kill-shot B: main arm, fixed scale (m = 3 row is the N = 2 row of A)
    b_main: dict[int, dict[float, dict[str, Any]]] = {}
    for m in m_values:
        b_main[m] = {a: cell(b_n_players, m, a, scale, "B-main") for a in levels}

    # ---- C1 calibration: median payoff_range at scale = 1, disjoint stream ---
    a_scales: dict[int, dict[float, float]] = {}
    b_scales: dict[int, dict[float, float]] = {}
    for alpha in ctl_levels:
        lvl = canonical.index(alpha)
        ref_range_a = _calibration_range(
            calib_key=calib_key,
            level_idx=lvl,
            n_players=a_ref,
            m=a_m,
            alpha=alpha,
            n_games=calib_games,
        )
        for n_players in n_players_values:
            rng = _calibration_range(
                calib_key=calib_key,
                level_idx=lvl,
                n_players=n_players,
                m=a_m,
                alpha=alpha,
                n_games=calib_games,
            )
            a_scales.setdefault(n_players, {})[alpha] = scale * ref_range_a / rng
        ref_range_b = _calibration_range(
            calib_key=calib_key,
            level_idx=lvl,
            n_players=b_n_players,
            m=b_ref,
            alpha=alpha,
            n_games=calib_games,
        )
        for m in m_values:
            rng = _calibration_range(
                calib_key=calib_key,
                level_idx=lvl,
                n_players=b_n_players,
                m=m,
                alpha=alpha,
                n_games=calib_games,
            )
            b_scales.setdefault(m, {})[alpha] = scale * ref_range_b / rng
    print(f"[C1] matched scales A: {a_scales}", flush=True)
    print(f"[C1] matched scales B: {b_scales}", flush=True)

    # ---- C1 arms: same games, same order, only scale differs ----------------
    a_ctl: dict[int, dict[float, dict[str, Any]]] = {}
    for n_players in n_players_values:
        a_ctl[n_players] = {
            a: cell(n_players, a_m, a, a_scales[n_players][a], "C1a") for a in ctl_levels
        }
    b_ctl: dict[int, dict[float, dict[str, Any]]] = {}
    for m in m_values:
        b_ctl[m] = {a: cell(b_n_players, m, a, b_scales[m][a], "C1b") for a in ctl_levels}

    # ---- T3: bootstrap CI on every reported rho_S ---------------------------
    cis: dict[tuple[str, int, float, str], tuple[float, float]] = {}
    arms: list[tuple[str, dict[int, dict[float, dict[str, Any]]]]] = [
        ("a_main", a_main),
        ("b_main", b_main),
        ("a_ctl", a_ctl),
        ("b_ctl", b_ctl),
    ]
    for arm_idx, (arm_name, arm) in enumerate(arms):
        for axis_idx, (axis_value, per_alpha) in enumerate(sorted(arm.items())):
            for level_idx, (alpha, c) in enumerate(sorted(per_alpha.items())):
                base = jax.random.fold_in(
                    jax.random.fold_in(jax.random.fold_in(boot_key, arm_idx), axis_idx), level_idx
                )
                for stat_idx, stat in enumerate(("r", "num")):
                    cis[(arm_name, axis_value, alpha, stat)] = _bootstrap_ci(
                        c["epr"],
                        c[stat],
                        key=jax.random.fold_in(base, 100 + stat_idx),
                        n_resamples=n_resamples,
                        ci_level=ci_level,
                    )

    # ---- A-T1, on the main arm and on the BINDING C1a arm -------------------
    adj_a_main = _adjudicate_ceiling(
        n_players_values,
        a_ref,
        [a_main[n][a_hi]["rho_ratio"] for n in n_players_values],
        [cis[("a_main", n, a_hi, "r")] for n in n_players_values],
        [a_main[n][a_lo]["rho_ratio"] for n in n_players_values],
        crit,
    )
    adj_a_ctl = _adjudicate_ceiling(
        n_players_values,
        a_ref,
        [a_ctl[n][a_hi]["rho_ratio"] for n in n_players_values],
        [cis[("a_ctl", n, a_hi, "r")] for n in n_players_values],
        [a_ctl[n][a_lo]["rho_ratio"] for n in n_players_values],
        crit,
    )

    # ---- B-T1, on the numerator, main arm and BINDING C1b arm ---------------
    adj_b_main = _adjudicate_ceiling(
        m_values,
        b_ref,
        [b_main[m][a_hi]["rho_num"] for m in m_values],
        [cis[("b_main", m, a_hi, "num")] for m in m_values],
        [b_main[m][a_lo]["rho_num"] for m in m_values],
        crit,
    )
    adj_b_ctl = _adjudicate_ceiling(
        m_values,
        b_ref,
        [b_ctl[m][a_hi]["rho_num"] for m in m_values],
        [cis[("b_ctl", m, a_hi, "num")] for m in m_values],
        [b_ctl[m][a_lo]["rho_num"] for m in m_values],
        crit,
    )

    # ---- A-T2: onset drift in N --------------------------------------------
    onsets = {n: _onset(a_main[n], levels, crit) for n in n_players_values}
    onset_drift = {n: onsets[n] - onsets[a_ref] for n in n_players_values}
    t2_pass = all(d <= float(crit["onset_drift_tol"]) + 1e-12 for d in onset_drift.values())
    print(f"[A-T2] onsets={onsets} drift={onset_drift} pass={t2_pass}", flush=True)

    # ---- T4(iv): was the λ̄ match actually achieved? -------------------------
    match_dev = 0.0
    match_ratios: dict[str, float] = {}
    for name, arm_ctl, ref_axis in (("a", a_ctl, a_ref), ("b", b_ctl, b_ref)):
        for axis_value, per_alpha in arm_ctl.items():
            for alpha, c in per_alpha.items():
                ratio = c["median_lambda_bar"] / arm_ctl[ref_axis][alpha]["median_lambda_bar"]
                match_ratios[f"{name}_{axis_value}_a{alpha:.2f}"] = ratio
                match_dev = max(match_dev, abs(ratio - 1.0))
    match_ok = match_dev <= float(crit["lambda_bar_match_tol"])
    print(f"[C1] lambda_bar match: max|ratio-1| = {match_dev:.4f}  ok={match_ok}", flush=True)

    # ---- T4(vi): wiring — the C1 reference cell IS the main-arm cell ---------
    pair_dev = max(
        abs(a_ctl[a_ref][alpha]["rho_ratio"] - a_main[a_ref][alpha]["rho_ratio"])
        for alpha in ctl_levels
    )
    pair_dev = max(
        pair_dev,
        max(
            abs(b_ctl[b_ref][alpha]["rho_num"] - b_main[b_ref][alpha]["rho_num"])
            for alpha in ctl_levels
        ),
    )
    pairing_ok = pair_dev <= float(crit["pairing_tol"])
    print(f"[T4vi] pairing wiring check: max|d rho| = {pair_dev:.2e} ok={pairing_ok}", flush=True)

    # ---- T4(vii): replication anchors against the published numbers ----------
    anchors: dict[str, float] = {}
    anchor_ok = True
    for alpha_key, published in crit["reference_rho_ratio"].items():
        alpha = float(alpha_key)
        c = a_main[a_ref][alpha]
        got = _spearman(c["epr"][:ref_n], c["r"][:ref_n])
        dev = abs(got - float(published))
        anchors[f"ratio_a{alpha_key}"] = got
        anchors[f"ratio_dev_a{alpha_key}"] = dev
        anchor_ok &= dev <= float(crit["replication_tol"])
        print(
            f"[T4vii] F-0004 anchor a={alpha}: first-{ref_n} rho={got:+.6f} "
            f"vs published {float(published):+.6f} |dev|={dev:.6f}",
            flush=True,
        )
    for alpha_key, published in crit["reference_rho_num"].items():
        alpha = float(alpha_key)
        c = b_main[b_ref][alpha]
        got = _spearman(c["epr"][:ref_n], c["num"][:ref_n])
        dev = abs(got - float(published))
        anchors[f"num_a{alpha_key}"] = got
        anchors[f"num_dev_a{alpha_key}"] = dev
        anchor_ok &= dev <= float(crit["replication_tol"])
        print(
            f"[T4vii] F-0007 anchor a={alpha}: first-{ref_n} rho_num={got:+.6f} "
            f"vs published {float(published):+.6f} |dev|={dev:.6f}",
            flush=True,
        )

    # ---- T4(v): diagnostic guards over every solved cell ---------------------
    diag_guard = any(
        c["near_critical_frac"] > max_near
        or c["non_converged"] > 0
        or c["rejected"] > 0
        or c["alpha_dev"] > float(crit["alpha_construction_tol"])
        for c in cache.values()
    )
    guard_fired = diag_guard or not match_ok or not pairing_ok or not anchor_ok

    verdict_a = _combine(adj_a_main["verdict"], adj_a_ctl["verdict"], guard_fired, t2_pass)
    verdict_b = _combine(adj_b_main["verdict"], adj_b_ctl["verdict"], guard_fired)

    # ---- reported-not-adjudicated contrasts, each with its own interval ------
    delta_num_ratio: dict[int, dict[str, float]] = {}
    for m_idx, m in enumerate(m_values):
        c = b_main[m][a_hi]
        lo, hi = _bootstrap_paired_delta(
            c["epr"],
            c["num"],
            c["r"],
            key=jax.random.fold_in(jax.random.fold_in(boot_key, 7000), m_idx),
            n_resamples=n_resamples,
            ci_level=ci_level,
        )
        delta_num_ratio[m] = {
            "value": c["rho_num"] - c["rho_ratio"],
            "ci_low": lo,
            "ci_high": hi,
        }
    gaps: dict[str, dict[str, float]] = {}
    for tag, arm, axis_vals, stat in (
        ("a_main", a_main, n_players_values, "r"),
        ("b_main", b_main, m_values, "num"),
        ("a_ctl", a_ctl, n_players_values, "r"),
        ("b_ctl", b_ctl, m_values, "num"),
    ):
        for axis_idx, axis_value in enumerate(axis_vals):
            lo, hi = _bootstrap_gap(
                arm[axis_value][a_lo],
                arm[axis_value][a_hi],
                stat,
                key=jax.random.fold_in(
                    jax.random.fold_in(jax.random.fold_in(boot_key, 8000), len(tag)), axis_idx
                ),
                n_resamples=n_resamples,
                ci_level=ci_level,
            )
            key_stat = "rho_ratio" if stat == "r" else "rho_num"
            gaps[f"{tag}_{axis_value}"] = {
                "value": arm[axis_value][a_lo][key_stat] - arm[axis_value][a_hi][key_stat],
                "ci_low": lo,
                "ci_high": hi,
            }

    # ---- artifact ------------------------------------------------------------
    metrics: dict[str, float] = {
        "verdict_a_holds": float(verdict_a.startswith("HOLDS")),
        "verdict_a_dies": float(verdict_a == "DIES"),
        "verdict_a_indeterminate": float(verdict_a == "INDETERMINATE"),
        "verdict_b_holds": float(verdict_b.startswith("HOLDS")),
        "verdict_b_dies": float(verdict_b == "DIES"),
        "verdict_b_indeterminate": float(verdict_b == "INDETERMINATE"),
        "a_main_holds_raw": float(adj_a_main["holds_raw"]),
        "a_ctl_holds_raw": float(adj_a_ctl["holds_raw"]),
        "a_arms_agree": float(adj_a_main["verdict"] == adj_a_ctl["verdict"]),
        "b_arms_agree": float(adj_b_main["verdict"] == adj_b_ctl["verdict"]),
        "a_main_worst_ci_high": adj_a_main["worst_ci_high"],
        "a_ctl_worst_ci_high": adj_a_ctl["worst_ci_high"],
        "b_main_worst_ci_high": adj_b_main["worst_ci_high"],
        "b_ctl_worst_ci_high": adj_b_ctl["worst_ci_high"],
        "a_main_baseline_ok": float(adj_a_main["baseline_ok"]),
        "a_ctl_baseline_ok": float(adj_a_ctl["baseline_ok"]),
        "b_main_baseline_ok": float(adj_b_main["baseline_ok"]),
        "b_ctl_baseline_ok": float(adj_b_ctl["baseline_ok"]),
        "a_t2_pass": float(t2_pass),
        "t4_iv_lambda_bar_match_ok": float(match_ok),
        "t4_iv_max_match_dev": match_dev,
        "t4_v_diagnostic_guard": float(diag_guard),
        "t4_vi_pairing_ok": float(pairing_ok),
        "t4_vi_max_pair_dev": pair_dev,
        "t4_vii_anchors_ok": float(anchor_ok),
        "guard_fired": float(guard_fired),
        "survive_ceiling": float(crit["survive_ceiling"]),
    }
    for n_players in n_players_values:
        metrics[f"a_t2_onset_n{n_players}"] = onsets[n_players]
        metrics[f"a_t2_onset_drift_n{n_players}"] = onset_drift[n_players]
    for m, contrast in delta_num_ratio.items():
        for stat, val in contrast.items():
            metrics[f"delta_num_minus_ratio_m{m}_{stat}"] = val
    for gap_name, gap_stats in gaps.items():
        for stat, val in gap_stats.items():
            metrics[f"gap_{gap_name}_{stat}"] = val
    for anchor_name, val in anchors.items():
        metrics[f"anchor_{anchor_name}"] = val
    for ratio_name, val in match_ratios.items():
        metrics[f"lambda_bar_ratio_{ratio_name}"] = val
    for arm_name, arm in arms:
        for axis_value, per_alpha in arm.items():
            for alpha, c in per_alpha.items():
                pre = f"{arm_name}_x{axis_value}_a{alpha:.2f}"
                metrics[f"{pre}_rho_epr_r"] = c["rho_ratio"]
                metrics[f"{pre}_rho_epr_num"] = c["rho_num"]
                metrics[f"{pre}_rho_r_invden"] = c["rho_r_invden"]
                metrics[f"{pre}_ci_low_r"] = cis[(arm_name, axis_value, alpha, "r")][0]
                metrics[f"{pre}_ci_high_r"] = cis[(arm_name, axis_value, alpha, "r")][1]
                metrics[f"{pre}_ci_low_num"] = cis[(arm_name, axis_value, alpha, "num")][0]
                metrics[f"{pre}_ci_high_num"] = cis[(arm_name, axis_value, alpha, "num")][1]
                metrics[f"{pre}_n"] = float(c["n"])
                metrics[f"{pre}_scale"] = c["scale"]
                metrics[f"{pre}_median_lambda_bar"] = c["median_lambda_bar"]
                metrics[f"{pre}_near_critical_frac"] = c["near_critical_frac"]
                metrics[f"{pre}_non_converged"] = float(c["non_converged"])
                metrics[f"{pre}_rejected"] = float(c["rejected"])
                metrics[f"{pre}_alpha_dev"] = c["alpha_dev"]
                metrics[f"{pre}_median_dist_crit"] = c["median_distance_to_criticality"]
                metrics[f"{pre}_median_epr"] = c["median_epr"]
                metrics[f"{pre}_median_r"] = c["median_r"]
                metrics[f"{pre}_median_num"] = c["median_num"]
                metrics[f"{pre}_median_den"] = c["median_den"]

    effects: list[EffectSize] = []
    for arm_name, arm in arms:
        for axis_value, per_alpha in arm.items():
            for alpha, c in per_alpha.items():
                for stat, label in (("r", "ratio_R"), ("num", "numerator_A")):
                    lo, hi = cis[(arm_name, axis_value, alpha, stat)]
                    effects.append(
                        EffectSize(
                            name=f"rho_S_EPR_{label}_{arm_name}_x{axis_value}_alpha_{alpha:.2f}",
                            value=c["rho_ratio"] if stat == "r" else c["rho_num"],
                            ci_low=lo,
                            ci_high=hi,
                            ci_level=ci_level,
                            method=(
                                f"percentile bootstrap ({n_resamples} resamples, games within "
                                f"cell, n={c['n']}); REGISTERED T3"
                            ),
                        )
                    )
    effects += [
        EffectSize(
            name=f"delta_rho_numerator_minus_ratio_m{m}_alpha_0.95",
            value=contrast["value"],
            ci_low=contrast["ci_low"],
            ci_high=contrast["ci_high"],
            ci_level=ci_level,
            method=(
                f"PAIRED percentile bootstrap ({n_resamples} resamples, identical resampled "
                "game indices for both statistics); REPORTED, adjudicated against nothing"
            ),
        )
        for m, contrast in delta_num_ratio.items()
    ]
    effects += [
        EffectSize(
            name=f"gap_{gap_name}",
            value=gap_stats["value"],
            ci_low=gap_stats["ci_low"],
            ci_high=gap_stats["ci_high"],
            ci_level=ci_level,
            method=(
                f"percentile bootstrap ({n_resamples} resamples, the two alpha cells resampled "
                "independently because they are different games); REPORTED as data — the "
                "gap-shrink family of criteria was deliberately NOT registered here, because "
                "F-0022 established it is not resolvable at feasible n"
            ),
        )
        for gap_name, gap_stats in gaps.items()
    ]

    n_solved = sum(int(c["n"]) for c in cache.values())
    n_cells = len(cache)
    n_source_draws = per_cell * (
        len(n_players_values) * len(levels) + (len(m_values) - 1) * len(levels)
    )
    n_calibration = calib_games * 2 * (len(n_players_values) + len(m_values) - 1)
    metrics["n_distinct_cells"] = float(n_cells)
    metrics["n_games_solved"] = float(n_solved)
    metrics["n_distinct_source_draws"] = float(n_source_draws)
    metrics["n_calibration_draws_never_solved"] = float(n_calibration)

    passed = anchor_ok and not diag_guard
    result = BenchmarkResult(
        benchmark_id="plane_nplayers",
        unit=UNIT,
        kind="statistical",
        passed=passed,
        metrics=metrics,
        effect_sizes=effects,
        n=n_solved,
        n_justification=(
            " ".join(str(cfg["n_justification"]).split())
            + " AS EXECUTED, for checking against the registered decomposition above: "
            f"{n_cells} distinct solved cells, {n_solved} games solved, "
            f"{n_source_draws} distinct source draws (the C1 arms re-scale draws the main "
            "arms already made — that is what paired means — and every C1 cell at its "
            "reference size is the main-arm cell itself, resolved through the cell cache "
            f"rather than re-solved), plus {n_calibration} calibration draws that are "
            "never solved and never enter any statistic. n recorded here is games solved."
        ),
        seed=seed,
        config_ref=str(CONFIG.relative_to(REPO)),
        library_version=strataq.__version__,
        timestamp=_now(),
        notes=(
            f"R12 kill-shots on the two-coordinate (R, EPR) claim. "
            f"A (N-SCALING, N in {n_players_values} at m={a_m}): {verdict_a}. "
            f"B (NUMERATOR ACROSS m, m in {m_values} at N={b_n_players}): {verdict_b}. "
            "Criteria registered in config/experiments/plane_nplayers.yaml at commit "
            "b9648f8, BEFORE this experiment file existed; that is and remains the ONLY "
            "criteria file for this unit, which is the direct lesson of R11's red-team "
            "objection O-1 (F-0022). "
            "A-T1 [main arm] rho_S(EPR,R) at alpha=0.95 = "
            + ", ".join(
                f"N={n}: {a_main[n][a_hi]['rho_ratio']:+.4f} "
                f"[{cis[('a_main', n, a_hi, 'r')][0]:+.4f}, "
                f"{cis[('a_main', n, a_hi, 'r')][1]:+.4f}]"
                for n in n_players_values
            )
            + f" against the registered ceiling {float(crit['survive_ceiling'])}; "
            "baseline rho_S at alpha=0.05 = "
            + ", ".join(f"N={n}: {a_main[n][a_lo]['rho_ratio']:+.4f}" for n in n_players_values)
            + f" (floor {float(crit['baseline_rho_min'])}). "
            f"A-T1 [C1a, paired and lambda_bar-matched] verdict {adj_a_ctl['verdict']}, "
            f"main-arm verdict {adj_a_main['verdict']}, arms agree "
            f"{adj_a_main['verdict'] == adj_a_ctl['verdict']}. "
            "B-T1 [main arm] rho_S(EPR, NUMERATOR) at alpha=0.95 = "
            + ", ".join(
                f"m={m}: {b_main[m][a_hi]['rho_num']:+.4f} "
                f"[{cis[('b_main', m, a_hi, 'num')][0]:+.4f}, "
                f"{cis[('b_main', m, a_hi, 'num')][1]:+.4f}]"
                for m in m_values
            )
            + "; the RATIO on the same games at alpha=0.95 = "
            + ", ".join(f"m={m}: {b_main[m][a_hi]['rho_ratio']:+.4f}" for m in m_values)
            + f". B-T1 [C1b] verdict {adj_b_ctl['verdict']}, main-arm verdict "
            f"{adj_b_main['verdict']}, arms agree "
            f"{adj_b_main['verdict'] == adj_b_ctl['verdict']}. "
            f"A-T2 onset drift {'PASS' if t2_pass else 'FAIL'} (onsets {onsets}). "
            f"C1 lambda_bar match achieved to max|ratio-1| = {match_dev:.4f} against the "
            f"registered 0.10 tolerance; pairing wiring check max|d rho| = {pair_dev:.2e}. "
            f"Replication anchors: F-0004 ratio and F-0007 numerator reproduced on the "
            f"first {ref_n} games of every N=2, m=3 cell, worst deviation "
            f"{max(v for k, v in anchors.items() if 'dev' in k):.6f} against a 0.02 "
            "tolerance — the m=3, N=2 rows are re-executions of the published readings, "
            "not fresh samples of them. "
            f"Strong-anticorrelation flag (reported, not a criterion): A at N in "
            f"{adj_a_main['strong_anticorrelation_at']}, B at m in "
            f"{adj_b_main['strong_anticorrelation_at']}. "
            "The gap-shrink family of criteria is deliberately absent from the "
            "registration (F-0022 showed it is not resolvable at feasible n); gap values "
            "appear in effect_sizes with intervals as DATA and decide nothing. "
            f"passed={passed} follows the house convention — the registered adjudication "
            "ran, the anchors hold and no data-quality guard fired; it is NOT a verdict."
        ),
    )
    (RESULTS / "plane_nplayers.json").write_text(result.model_dump_json(indent=2) + "\n")
    print(
        f"[{'ADJUDICATED' if passed else 'GUARD FIRED'}] plane_nplayers  "
        f"A={verdict_a}  B={verdict_b}  "
        f"cells={n_cells} solved={n_solved} draws={n_source_draws}",
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(run())
