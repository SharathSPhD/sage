"""strataq.fit -- the QRE estimation workflow, packaged end to end.

Bland & Turocy (*GEB* 2025), "Quantal response equilibrium as a structural model
for estimation: the missing manual", exists because the workflow is not in any
software. ``pygambit.qre.logit_estimate`` returns ``.lam``, ``.profile`` and
``.log_like`` -- no standard error, no confidence interval, no likelihood-ratio
test -- and its *input* is a profile of aggregated counts, so subject, round and
treatment are destroyed before the likelihood is written. Every published CI on
lambda is therefore hand-rolled, and hand-rolled differently in each lab.

This module is the reference implementation of that manual:

* **tidy data in.** One row per observed choice, with whatever subject / round /
  treatment columns your experiment software emitted. Aggregated counts (what
  :mod:`strataq.estimate.lam` already takes) are accepted unchanged.
* **the panel survives the likelihood.** The point estimate is the same either
  way -- individual categorical draws and their multinomial aggregate differ by a
  constant -- but the *uncertainty* is not: with a subject column the bootstrap
  resamples **subjects**, so within-subject correlation widens the interval
  instead of being silently averaged away. Without one, this module says so.
* **one interval, named.** ``fit().summary()`` prints lambda-hat, an interval with
  its method spelled out, n, the log-likelihood, and likelihood-ratio tests
  against both nested boundaries -- Nash (lambda -> inf) and uniform (lambda = 0).
* **``by=`` reports heterogeneity, it does not average it.** Per-group lambda plus
  an LR test of homogeneity, with disagreement flagged.
* **refusal is a bound, not an exception.** Where the likelihood is flat, lambda
  is reported as the interval it truly is -- the whole search window -- and the
  point estimate is withheld rather than quoted.

Nothing statistical is invented here. The estimator is
:func:`strataq.estimate.lam.lambda_mle` (gated), the four-estimator redundancy
check is :func:`strataq.estimate.lam.agreement_protocol` (gated), and the
Bayesian route is :func:`strataq.estimate.bayes.grid_posterior` (gated). This
module is the packaging.

References
----------
McKelvey--Palfrey *GEB* 1995 (logit QRE); Bland & Turocy *GEB* 2025 (the manual);
Self & Liang *JASA* 1987 (LR tests on a parameter-space boundary). Tier: derived.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
from jax import Array

from strataq import __version__
from strataq.core.defaults import base_config
from strataq.core.solve.fixedpoint import logit_qre
from strataq.estimate.bayes import precompute_sigmas, refined_posterior
from strataq.estimate.lam import LambdaAgreement, agreement_protocol, lambda_mle
from strataq.finite.games.tensor import DenseTensorGame

__all__ = ["GroupFit", "LRTest", "LambdaFit", "Summary", "fit"]

Method = Literal["mle", "agreement", "bayes"]
CIMethod = Literal["bootstrap", "profile", "posterior", "none"]

# Column names this module will recognise without being told. Matching is
# case-insensitive and exact (never substring: "player_id" is a SUBJECT in oTree
# and a ROLE nowhere, so guessing from substrings silently mis-assigns choices).
_PLAYER_COLS = ("player", "role", "position", "player_role", "id_in_group")
_ACTION_COLS = ("action", "choice", "strategy", "decision", "chosen", "a")
_SUBJECT_COLS = ("subject", "subject_id", "participant", "participant_code", "cluster")

_TINY = 1e-300
_ZERO_PROB = 1e-12  # below this a QRE probability is "numerically zero" for a likelihood


class Summary(str):
    """A report that displays as itself in a REPL and prints with ``print``."""

    def __repr__(self) -> str:
        return str(self)


# ---------------------------------------------------------------------------
# chi-square upper tail, without adding scipy to a pure-JAX dependency set.
# Numerical Recipes 6.2 (series below the crossover, continued fraction above).
# ---------------------------------------------------------------------------

_ITMAX = 400
_EPS = 3.0e-14


def _gamma_p_series(a: float, x: float) -> float:
    ap = a
    term = 1.0 / a
    total = term
    for _ in range(_ITMAX):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * _EPS:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_q_cf(a: float, x: float) -> float:
    b = x + 1.0 - a
    c = 1.0 / _TINY
    d = 1.0 / b
    h = d
    for i in range(1, _ITMAX + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _TINY:
            d = _TINY
        c = b + an / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _chi2_sf(x: float, df: int) -> float:
    """P(chi^2_df > x). Exact to ~1e-13; no scipy."""
    if math.isnan(x):
        return float("nan")
    if math.isinf(x):
        return 0.0
    if x <= 0.0 or df <= 0:
        return 1.0
    a, y = 0.5 * df, 0.5 * x
    return float(1.0 - _gamma_p_series(a, y)) if y < a + 1.0 else float(_gamma_q_cf(a, y))


def _fmt_p(p: float) -> str:
    if math.isnan(p):
        return "n/a"
    return "< 1e-16" if p < 1e-16 else f"{p:.4g}"


# ---------------------------------------------------------------------------
# Ingestion: tidy long-form OR aggregated counts, one code path after this.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Panel:
    """Observed choices with the panel structure kept, not aggregated away."""

    counts: tuple[np.ndarray, ...]  # per player, per action
    subject_counts: np.ndarray | None  # (n_subjects, total_actions) cluster matrix
    group_keys: tuple[str, ...]  # one label per observation group (may be empty)
    group_panels: tuple[tuple[str, _Panel], ...] = ()
    n_obs: int = 0
    n_subjects: int = 0
    tidy: bool = False
    columns_used: dict[str, str] = field(default_factory=dict)


def _as_columns(data: Any) -> dict[str, np.ndarray] | None:
    """Column name -> values for anything table-like; ``None`` if not a table."""
    raw: Any = None
    if isinstance(data, Mapping):
        raw = data
    else:
        to_dict = getattr(data, "to_dict", None)
        if callable(to_dict):
            for args, kwargs in ((), {"as_series": False}), (("list",), {}):
                try:
                    candidate = to_dict(*args, **kwargs)
                except (TypeError, ValueError):
                    continue
                if isinstance(candidate, Mapping):
                    raw = candidate
                    break
        if raw is None and isinstance(data, Sequence) and len(data) > 0:
            first = data[0]
            if isinstance(first, Mapping):
                keys = list(first.keys())
                raw = {k: [row[k] for row in data] for k in keys}
    if raw is None:
        return None
    return {str(k): np.asarray(list(v), dtype=object) for k, v in raw.items()}


def _find_column(columns: Mapping[str, np.ndarray], candidates: Sequence[str]) -> str | None:
    lowered = {name.lower(): name for name in columns}
    for want in candidates:
        if want in lowered:
            return lowered[want]
    return None


def _to_index(values: np.ndarray, n_levels: int, what: str) -> tuple[np.ndarray, list[str]]:
    """Map a column to 0-based indices, returning the label order it assumed."""
    try:
        idx = np.asarray([int(v) for v in values], dtype=np.int64)
    except (TypeError, ValueError):
        labels = sorted({str(v) for v in values})
        lookup = {lab: i for i, lab in enumerate(labels)}
        idx = np.asarray([lookup[str(v)] for v in values], dtype=np.int64)
        return idx, labels
    if idx.size and (idx.min() < 0 or idx.max() >= n_levels):
        raise ValueError(
            f"{what} values must be 0-based indices in [0, {n_levels - 1}]; got "
            f"[{idx.min()}, {idx.max()}]. Pass labels (strings) if your data are 1-based "
            "or named, and the label order will be reported in provenance."
        )
    return idx, []


def _is_two_player_symmetric(game: DenseTensorGame) -> bool:
    if game.n_players != 2:
        return False
    u0, u1 = game.payoffs
    return bool(jnp.allclose(u1, u0.T))


def _counts_from_rows(
    player: np.ndarray, action: np.ndarray, num_actions: tuple[int, ...]
) -> tuple[np.ndarray, ...]:
    out = []
    for p, m in enumerate(num_actions):
        out.append(np.bincount(action[player == p], minlength=m).astype(float)[:m])
    return tuple(out)


def _offsets(num_actions: tuple[int, ...]) -> tuple[np.ndarray, int]:
    starts = np.concatenate([[0], np.cumsum(np.asarray(num_actions))])
    return starts[:-1].astype(np.int64), int(starts[-1])


def _panel_from_counts(game: DenseTensorGame, data: Any) -> _Panel:
    try:
        rows = [np.asarray(c, dtype=float).reshape(-1) for c in data]
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "data= must be a tidy table (one row per observed choice) or one count "
            f"vector per player; got {type(data).__name__}."
        ) from exc
    if len(rows) != game.n_players:
        raise ValueError(
            f"aggregated counts need one vector per player: the game has {game.n_players} "
            f"players, data has {len(rows)} vectors."
        )
    for i, (c, m) in enumerate(zip(rows, game.num_actions, strict=True)):
        if c.shape != (m,):
            raise ValueError(f"counts for player {i} must have length {m}, got {c.shape}")
        if not np.all(np.isfinite(c)) or np.any(c < 0):
            raise ValueError(f"counts for player {i} must be finite and nonnegative")
    total = int(sum(float(c.sum()) for c in rows))
    if total == 0:
        raise ValueError("no observations: every count is zero")
    return _Panel(
        counts=tuple(rows),
        subject_counts=None,
        group_keys=(),
        n_obs=total,
        n_subjects=0,
        tidy=False,
    )


def _panel_from_tidy(
    game: DenseTensorGame,
    columns: dict[str, np.ndarray],
    *,
    player: str | int | None,
    action: str | None,
    subject: str | None,
    by: str | Sequence[str] | None,
    warnings: list[str],
    provenance: dict[str, Any],
) -> _Panel:
    n_rows = len(next(iter(columns.values()))) if columns else 0
    if n_rows == 0:
        raise ValueError("data= is an empty table: nothing to estimate from")
    for name, values in columns.items():
        if len(values) != n_rows:
            raise ValueError(f"column {name!r} has {len(values)} rows, expected {n_rows}")

    action_col = action or _find_column(columns, _ACTION_COLS)
    if action_col is None:
        raise ValueError(
            "no action column found. Name one of "
            f"{list(_ACTION_COLS)} or pass action='<column>'. Columns present: "
            f"{sorted(columns)}"
        )
    if action_col not in columns:
        raise ValueError(f"action column {action_col!r} not in the table: {sorted(columns)}")

    used: dict[str, str] = {"action": action_col}
    max_actions = max(game.num_actions)

    # -- which player made each choice ---------------------------------------------
    if isinstance(player, int):
        player_idx = np.full(n_rows, int(player), dtype=np.int64)
        used["player"] = f"constant {player}"
    else:
        player_col = player or _find_column(columns, _PLAYER_COLS)
        if player_col is not None:
            if player_col not in columns:
                raise ValueError(f"player column {player_col!r} not in the table")
            player_idx, labels = _to_index(columns[player_col], game.n_players, "player")
            used["player"] = player_col
            if labels:
                provenance["player_label_order"] = labels
        elif game.n_players == 1:
            player_idx = np.zeros(n_rows, dtype=np.int64)
            used["player"] = "single-player game"
        elif _is_two_player_symmetric(game):
            player_idx = np.zeros(n_rows, dtype=np.int64)
            used["player"] = "pooled (game is symmetric)"
            warnings.append(
                "no player/role column: the game is symmetric under exchanging the two "
                "players, so every choice was pooled onto player 0's marginal. That is "
                "exact HERE and wrong for any asymmetric game -- pass player='<column>' "
                "if your rows carry a role."
            )
        else:
            raise ValueError(
                "no player/role column found and the game is not symmetric, so choices "
                "cannot be assigned to players. Name a column one of "
                f"{list(_PLAYER_COLS)}, or pass player='<column>' (or player=<int> if "
                "every row is the same player)."
            )
    if player_idx.size and (player_idx.min() < 0 or player_idx.max() >= game.n_players):
        raise ValueError(f"player indices must lie in [0, {game.n_players - 1}]")

    action_idx, action_labels = _to_index(columns[action_col], max_actions, "action")
    if action_labels:
        provenance["action_label_order"] = action_labels
        warnings.append(
            "the action column holds labels, not indices: they were mapped to actions in "
            f"sorted order {action_labels}. Check that this matches the payoff tensor's "
            "axis order -- a mismatch silently estimates the wrong model."
        )
    for p in range(game.n_players):
        mask = player_idx == p
        if mask.any() and int(action_idx[mask].max()) >= game.num_actions[p]:
            raise ValueError(
                f"player {p} has an action index {int(action_idx[mask].max())} but only "
                f"{game.num_actions[p]} actions"
            )

    # -- the cluster the bootstrap must resample ------------------------------------
    subject_col = subject or _find_column(columns, _SUBJECT_COLS)
    subject_ids: np.ndarray | None = None
    if subject_col is not None:
        if subject_col not in columns:
            raise ValueError(f"subject column {subject_col!r} not in the table")
        subject_ids = np.asarray([str(v) for v in columns[subject_col]])
        used["subject"] = subject_col
    else:
        warnings.append(
            "no subject column: the bootstrap resamples individual CHOICES, which assumes "
            "they are independent. Experimental choices are correlated within subject, so "
            "an interval computed this way is too narrow -- pass subject='<column>' to "
            "cluster it. (This is exactly what aggregating to counts throws away.)"
        )

    starts, total_actions = _offsets(game.num_actions)
    flat = starts[player_idx] + action_idx

    subject_counts: np.ndarray | None = None
    n_subjects = 0
    if subject_ids is not None:
        uniq, inverse = np.unique(subject_ids, return_inverse=True)
        n_subjects = int(uniq.size)
        subject_counts = np.zeros((n_subjects, total_actions), dtype=float)
        np.add.at(subject_counts, (inverse, flat), 1.0)

    def _build(mask: np.ndarray) -> _Panel:
        sub: np.ndarray | None = None
        n_sub = 0
        if subject_ids is not None:
            uniq_g, inv_g = np.unique(subject_ids[mask], return_inverse=True)
            n_sub = int(uniq_g.size)
            sub = np.zeros((n_sub, total_actions), dtype=float)
            np.add.at(sub, (inv_g, flat[mask]), 1.0)
        return _Panel(
            counts=_counts_from_rows(player_idx[mask], action_idx[mask], game.num_actions),
            subject_counts=sub,
            group_keys=(),
            n_obs=int(mask.sum()),
            n_subjects=n_sub,
            tidy=True,
            columns_used=used,
        )

    groups: list[tuple[str, _Panel]] = []
    if by is not None:
        by_cols = [by] if isinstance(by, str) else list(by)
        for name in by_cols:
            if name not in columns:
                raise ValueError(f"by={name!r} is not a column: {sorted(columns)}")
        keys = np.asarray(["|".join(str(columns[c][i]) for c in by_cols) for i in range(n_rows)])
        used["by"] = ", ".join(by_cols)
        for key in sorted(set(keys.tolist())):
            groups.append((str(key), _build(keys == key)))

    whole = _build(np.ones(n_rows, dtype=bool))
    return _Panel(
        counts=whole.counts,
        subject_counts=subject_counts,
        group_keys=(),
        group_panels=tuple(groups),
        n_obs=n_rows,
        n_subjects=n_subjects,
        tidy=True,
        columns_used=used,
    )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LRTest:
    """A nested likelihood-ratio test, with its boundary caveat attached."""

    name: str
    stat: float
    df: int
    p: float
    p_boundary: float | None
    note: str

    def line(self) -> str:
        stat = "inf" if math.isinf(self.stat) else f"{self.stat:.4g}"
        out = f"  {self.name:<30} LR = {stat:>12}   df = {self.df}   p {_fmt_p(self.p)}"
        if self.p_boundary is not None:
            out += f"   [boundary-corrected p {_fmt_p(self.p_boundary)}]"
        return out


@dataclass(frozen=True)
class GroupFit:
    """One group's lambda under ``by=``. Never averaged into the others."""

    key: str
    lam_hat: float
    ci_low: float
    ci_high: float
    kind: Literal["point", "unidentified"]
    loglik: float
    n_obs: int
    n_subjects: int
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class LambdaFit:
    """The whole estimate: a number, an interval that names its method, and the caveats."""

    lam_hat: float
    ci_low: float
    ci_high: float
    ci_method: str
    ci_level: float
    kind: Literal["point", "unidentified"]
    loglik: float
    n_obs: int
    n_subjects: int
    n_players: int
    method: str
    lr_nash: LRTest | None = None
    lr_uniform: LRTest | None = None
    groups: tuple[GroupFit, ...] = ()
    homogeneity: LRTest | None = None
    agreement: LambdaAgreement | None = None
    warnings: tuple[str, ...] = ()
    refusals: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    _game: Any = field(default=None, repr=False, compare=False)
    _freqs: tuple[np.ndarray, ...] = field(default=(), repr=False, compare=False)

    @property
    def identified(self) -> bool:
        """False when the likelihood is flat -- the interval is the honest answer."""
        return self.kind == "point"

    def __repr__(self) -> str:
        head = (
            "lambda NOT IDENTIFIED"
            if self.kind == "unidentified"
            else f"lambda_hat = {self.lam_hat:.6g}"
        )
        return (
            f"LambdaFit({head}, {self.ci_level:.0%} CI [{self.ci_low:.4g}, {self.ci_high:.4g}], "
            f"n={self.n_obs}, logL={self.loglik:.4f}, "
            f"{len(self.refusals)} refusal(s), {len(self.warnings)} warning(s))"
        )

    def summary(self) -> Summary:
        """The full report. ``print(fit.summary())`` and ``fit.summary()`` agree."""
        rule = "=" * 78
        thin = "-" * 78
        lines = ["strataq.fit -- logit QRE as a structural model", rule]
        if self.kind == "unidentified":
            lines += [
                "lambda_hat         : NOT IDENTIFIED (withheld, see REFUSALS)",
                f"{self.ci_level:.0%} bound           : "
                f"[{self.ci_low:.6g}, {self.ci_high:.6g}]   ({self.ci_method})",
            ]
        else:
            lines += [
                f"lambda_hat         : {self.lam_hat:.6g}",
                f"{self.ci_level:.0%} CI              : "
                f"[{self.ci_low:.6g}, {self.ci_high:.6g}]   ({self.ci_method})",
            ]
        subj = (
            f" from {self.n_subjects} subjects"
            if self.n_subjects
            else " (no subject column -- choices treated as independent)"
        )
        lines += [
            f"n                  : {self.n_obs} choices{subj}, {self.n_players} players",
            f"log-likelihood     : {self.loglik:.6f}"
            "   (multinomial kernel; the combinatorial constant is dropped)",
            f"estimator          : {self.method}",
        ]
        tests = [t for t in (self.lr_nash, self.lr_uniform) if t is not None]
        if tests:
            lines += ["", "LIKELIHOOD-RATIO TESTS (both nulls are nested boundaries)", thin]
            lines += [t.line() for t in tests]
            lines += [f"    {t.name}: {t.note}" for t in tests]
        if self.groups:
            by = self.provenance.get("by", "group")
            lines += ["", f"PER-GROUP LAMBDA (by = {by!r})", thin]
            lines.append(
                f"  {'group':<22}{'lambda':>12}{'ci_low':>12}{'ci_high':>12}"
                f"{'n':>9}{'subjects':>10}"
            )
            for g in self.groups:
                lam = "unidentified" if g.kind == "unidentified" else f"{g.lam_hat:.4g}"
                lines.append(
                    f"  {g.key[:22]:<22}{lam:>12}{g.ci_low:>12.4g}{g.ci_high:>12.4g}"
                    f"{g.n_obs:>9}{g.n_subjects:>10}"
                )
            if self.homogeneity is not None:
                lines += ["", self.homogeneity.line(), f"    {self.homogeneity.note}"]
        if self.agreement is not None:
            lines += ["", "FOUR-ESTIMATOR AGREEMENT (redundancy, not averaging)", thin]
            for name, est in self.agreement.estimates.items():
                lines.append(
                    f"  {name:<16}lambda = {est.lam:.6g}   [{est.ci_low:.4g}, {est.ci_high:.4g}]"
                )
            lines.append(
                f"  relative spread  {self.agreement.agreement_gap:.4g}"
                f"   disagreement_flag = {self.agreement.disagreement_flag}"
            )
        if self.refusals:
            lines += ["", "REFUSALS (reported as bounds, not silences)", thin]
            lines += [f"  - {r}" for r in self.refusals]
        if self.warnings:
            lines += ["", "WARNINGS", thin]
            lines += [f"  - {w}" for w in self.warnings]
        lines += ["", "PROVENANCE", thin]
        lines += [f"  {k:<24}: {v}" for k, v in self.provenance.items()]
        return Summary("\n".join(lines))

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serialisable view of the whole fit (nothing dropped)."""

        def _test(t: LRTest | None) -> dict[str, Any] | None:
            if t is None:
                return None
            stat = None if math.isinf(t.stat) else t.stat
            return {
                "name": t.name,
                "stat": stat,
                "stat_is_infinite": math.isinf(t.stat),
                "df": t.df,
                "p": t.p,
                "p_boundary": t.p_boundary,
                "note": t.note,
            }

        return {
            "lam_hat": None if self.kind == "unidentified" else self.lam_hat,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_method": self.ci_method,
            "ci_level": self.ci_level,
            "kind": self.kind,
            "identified": self.identified,
            "loglik": self.loglik,
            "n_obs": self.n_obs,
            "n_subjects": self.n_subjects,
            "n_players": self.n_players,
            "method": self.method,
            "lr_nash": _test(self.lr_nash),
            "lr_uniform": _test(self.lr_uniform),
            "homogeneity": _test(self.homogeneity),
            "groups": [
                {
                    "key": g.key,
                    "lam_hat": None if g.kind == "unidentified" else g.lam_hat,
                    "ci_low": g.ci_low,
                    "ci_high": g.ci_high,
                    "kind": g.kind,
                    "loglik": g.loglik,
                    "n_obs": g.n_obs,
                    "n_subjects": g.n_subjects,
                    "warnings": list(g.warnings),
                }
                for g in self.groups
            ],
            "agreement": (
                None
                if self.agreement is None
                else {
                    "estimates": {
                        k: {"lam": e.lam, "ci_low": e.ci_low, "ci_high": e.ci_high}
                        for k, e in self.agreement.estimates.items()
                    },
                    "agreement_gap": self.agreement.agreement_gap,
                    "disagreement_flag": self.agreement.disagreement_flag,
                }
            ),
            "warnings": list(self.warnings),
            "refusals": list(self.refusals),
            "provenance": dict(self.provenance),
        }

    def plot(self, ax: Any = None, *, lam_max: float | None = None) -> Any:
        """The branch with lambda-hat marked and the observed frequencies overlaid.

        Requires ``strataq[viz]`` (matplotlib). Returns the Axes so the caller can
        keep composing.
        """
        from strataq.core.solve.homotopy import logit_branch
        from strataq.viz import PALETTE, plot_branch

        if self._game is None:
            raise ValueError("this LambdaFit carries no game and cannot be plotted")
        top = lam_max if lam_max is not None else max(3.0 * max(self.lam_hat, 1e-3), 1.0)
        branch = logit_branch(self._game, float(top))
        lams = np.asarray(branch.lambdas, dtype=float)
        folds = lams[np.asarray(branch.turning_points, dtype=bool)]
        mark = None if self.kind == "unidentified" else float(self.lam_hat)
        ax = plot_branch(branch, ax=ax, mark_lambda=mark, turning_points=folds.tolist())
        x = float(self.lam_hat) if mark is not None else float(np.sqrt(self.ci_low * self.ci_high))
        labelled = False
        for freqs in self._freqs:
            for p in np.asarray(freqs, dtype=float):
                ax.plot(
                    [x],
                    [float(p)],
                    marker="o",
                    ms=5.0,
                    mfc="none",
                    mec=PALETTE["ink"],
                    mew=1.0,
                    ls="none",
                    zorder=10,
                    label=None if labelled else "observed frequency",
                )
                labelled = True
        if labelled:
            ax.legend(loc="best")
        return ax


# ---------------------------------------------------------------------------
# The estimator, assembled from gated parts
# ---------------------------------------------------------------------------


def _grid_sigmas(
    game: DenseTensorGame, grid: np.ndarray
) -> tuple[list[tuple[Array, ...]], np.ndarray]:
    """Every QRE solve on the grid, once: the mixtures and their log matrix.

    The solves do not depend on the data, so the bootstrap, the posterior and
    every group fit read this instead of re-solving. Shape (n_grid, sum m_i).
    """
    sigmas = precompute_sigmas(game, grid)
    rows = [
        np.concatenate([np.log(np.maximum(np.asarray(s, dtype=float), _TINY)) for s in sigma])
        for sigma in sigmas
    ]
    return sigmas, np.stack(rows)


def _flat_counts(counts: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(c, dtype=float) for c in counts])


def _loglik_uniform(counts: Sequence[np.ndarray], num_actions: tuple[int, ...]) -> float:
    """log L at lambda = 0 exactly: sigma*(0) is uniform, no solve needed."""
    return float(
        sum(float(np.sum(c)) * math.log(1.0 / m) for c, m in zip(counts, num_actions, strict=True))
    )


def _nash_limit(game: DenseTensorGame, lam_start: float) -> tuple[tuple[Array, ...], float]:
    """sigma at lambda -> inf, by warm-started continuation up the branch.

    Returns the limiting mixture and the largest lambda the solver actually
    reached -- the report quotes that lambda rather than pretending the limit
    was evaluated exactly.
    """
    prange = float(np.asarray(game.payoff_range))
    target = min(2000.0 / max(prange, 1e-9), 1e6)
    start = max(float(lam_start), 1e-2)
    if target <= start:
        target = start * 100.0
    ladder = np.geomspace(start, target, 24)
    point = logit_qre(game, float(ladder[0]))
    reached = float(ladder[0])
    for lam in ladder[1:]:
        nxt = logit_qre(game, float(lam), init=point.sigma)
        if not bool(nxt.converged):
            break
        point, reached = nxt, float(lam)
    return point.sigma, reached


def _argmax_refined(lls: np.ndarray, log_grid: np.ndarray) -> np.ndarray:
    """Parabolic refinement of a grid argmax, row by row.

    ``lls`` is (B, G) over a UNIFORMLY log-spaced grid. A log-likelihood is
    locally quadratic in log-lambda near its maximum, so the vertex of the
    parabola through the peak and its two neighbours is the maximiser to
    O(h^3). Without this the bootstrap is quantised to the grid step and
    COLLAPSES TO A SINGLE POINT at large n -- a zero-width "confidence"
    interval, which is the exact failure mode this module exists to prevent.
    """
    n_grid = lls.shape[1]
    k = np.argmax(lls, axis=1)
    kc = np.clip(k, 1, n_grid - 2)
    rows = np.arange(lls.shape[0])
    y0, y1, y2 = lls[rows, kc - 1], lls[rows, kc], lls[rows, kc + 1]
    denom = y0 - 2.0 * y1 + y2
    # denom < 0 is the concave (true maximum) case; anything else keeps the node
    delta = np.where(denom < -1e-300, 0.5 * (y0 - y2) / np.where(denom == 0.0, 1.0, denom), 0.0)
    delta = np.clip(delta, -1.0, 1.0)
    delta = np.where((k == 0) | (k == n_grid - 1), 0.0, delta)  # no bracketing triple
    step = float(log_grid[1] - log_grid[0])
    return np.asarray(np.exp(log_grid[kc] + delta * step))


def _bootstrap_ci(
    panel: _Panel,
    log_sigma: np.ndarray,
    grid: np.ndarray,
    *,
    n_boot: int,
    level: float,
    rng: np.random.Generator,
) -> tuple[float, float, str, bool]:
    """Percentile bootstrap, clustered on subject whenever a subject column exists."""
    cluster = panel.subject_counts
    clustered = cluster is not None and cluster.shape[0] > 1
    if cluster is not None and clustered:
        n_sub = cluster.shape[0]
        weights = rng.multinomial(n_sub, np.full(n_sub, 1.0 / n_sub), size=n_boot).astype(float)
        draws = weights @ cluster
        label = f"cluster bootstrap on subject, B={n_boot}, {n_sub} clusters"
    else:
        blocks = []
        for c in panel.counts:
            total = round(float(np.sum(c)))
            probs = np.asarray(c, dtype=float)
            probs = probs / probs.sum() if probs.sum() > 0 else np.full(len(c), 1.0 / len(c))
            blocks.append(rng.multinomial(total, probs, size=n_boot).astype(float))
        draws = np.concatenate(blocks, axis=1)
        label = f"nonparametric bootstrap on choices, B={n_boot}"
    lls = draws @ log_sigma.T  # (B, n_grid)
    picks = _argmax_refined(lls, np.log(grid))
    tail = 0.5 * (1.0 - level)
    lo, hi = (float(q) for q in np.quantile(picks, [tail, 1.0 - tail]))
    method = f"{label}; re-maximised on a {len(grid)}-point log-grid with parabolic refinement"
    return lo, hi, method, clustered


def _fit_counts(
    game: DenseTensorGame,
    panel: _Panel,
    *,
    method: Method,
    ci: CIMethod,
    grid: np.ndarray,
    sigmas: list[tuple[Array, ...]],
    log_sigma: np.ndarray,
    n_boot: int,
    level: float,
    rng: np.random.Generator,
) -> tuple[float, float, float, str, float, Literal["point", "unidentified"], list[str], Any]:
    """One group's estimate. Returns lam, lo, hi, ci_method, loglik, kind, warnings, extra."""
    counts: tuple[Array, ...] = tuple(jnp.asarray(c) for c in panel.counts)
    extra: Any = None
    warns: list[str] = []

    if method == "agreement":
        extra = agreement_protocol(game, counts)
        est = extra.estimates["mle"]
        warns.extend(extra.warnings)
    elif method == "bayes":
        extra = refined_posterior(game, counts, grid, points=len(grid))
        est = lambda_mle(game, counts)
        warns.extend(est.warnings)
    else:
        est = lambda_mle(game, counts)
        warns.extend(est.warnings)

    unidentified = any("unidentified" in w for w in est.warnings)
    kind: Literal["point", "unidentified"] = "unidentified" if unidentified else "point"

    # The reported log-likelihood is ALWAYS the maximised one, including on the
    # Bayesian route: the tests below are likelihood-RATIO tests, and quoting them
    # against a posterior-mean likelihood would silently change what is tested.
    lam_hat = float(extra.mean) if method == "bayes" else float(est.lam)
    loglik = float(est.objective)

    if ci == "none":
        lo, hi, ci_method = lam_hat, lam_hat, "none requested"
    elif ci == "posterior" or method == "bayes":
        post = (
            extra if method == "bayes" else refined_posterior(game, counts, grid, points=len(grid))
        )
        lo, hi = post.credible_interval(level)
        ci_method = (
            f"grid posterior (uniform-on-grid prior, {len(grid)} points, refined onto the "
            f"posterior mass), {level:.0%} central credible interval"
        )
        if not post.grid_resolved:
            warns.append(
                "grid-resolution limited: fewer than ~6 effective grid points carry the "
                "posterior, which measurably undercovers -- treat the interval as indicative"
            )
    elif ci == "profile":
        lo, hi = float(est.ci_low), float(est.ci_high)
        cfg = base_config().estimate
        ci_method = (
            f"profile likelihood, chi2(1) drop {cfg.profile_ci_drop}, "
            f"nearest crossing on a {cfg.grid_points}-point grid (conservative)"
        )
    else:
        lo, hi, ci_method, clustered = _bootstrap_ci(
            panel, log_sigma, grid, n_boot=n_boot, level=level, rng=rng
        )
        if not clustered and panel.tidy:
            warns.append(
                "the bootstrap could not cluster (no subject column, or one subject): the "
                "interval assumes independent choices"
            )
        if hi - lo < 1e-9 * max(lam_hat, 1e-9):
            warns.append(
                "the bootstrap interval has collapsed to a point: every resample maximised "
                "at the same lambda. Raise n_boot, or raise n_grid -- do not read this as "
                "zero uncertainty"
            )
    if unidentified:
        lo, hi = float(grid[0]), float(grid[-1])
        ci_method = (
            f"the whole search window [{grid[0]:.4g}, {grid[-1]:.4g}] -- the widest TRUE "
            "statement, because the likelihood is flat on it"
        )
    return lam_hat, float(min(lo, hi)), float(max(lo, hi)), ci_method, loglik, kind, warns, extra


def fit(
    game: DenseTensorGame,
    data: Any,
    *,
    by: str | Sequence[str] | None = None,
    method: Method = "mle",
    ci: CIMethod = "bootstrap",
    n_boot: int = 400,
    seed: int | None = None,
    level: float = 0.95,
    player: str | int | None = None,
    action: str | None = None,
    subject: str | None = None,
    n_grid: int = 200,
) -> LambdaFit:
    """Estimate the logit precision lambda, with an interval that names its method.

    Parameters
    ----------
    game
        The game the choices were made in
        (:class:`~strataq.finite.games.tensor.DenseTensorGame`).
    data
        Either a **tidy long-form table** -- one row per observed choice, as a
        polars/pandas ``DataFrame``, a ``{column: values}`` mapping, or a list of
        record dicts -- or **aggregated counts**, one vector per player (what
        :mod:`strataq.estimate.lam` takes). The tidy route keeps subject, round
        and treatment; the count route cannot, and the report says so.
    by
        Column (or columns) to split on. Each group gets its own lambda, and a
        likelihood-ratio test of homogeneity is reported. Nothing is averaged.
    method
        ``"mle"`` (frequency MLE, gated), ``"agreement"`` (also runs the
        four-estimator redundancy protocol and reports the spread), or
        ``"bayes"`` (grid posterior, uniform-on-grid prior).
    ci
        ``"bootstrap"`` (clustered on subject when one exists), ``"profile"``
        (chi2(1) profile likelihood), ``"posterior"``, or ``"none"``.
    player, action, subject
        Column names, when the automatic match is wrong or absent. ``player`` may
        also be an ``int`` when every row is the same player.

    Examples
    --------
    >>> import strataq
    >>> from strataq.estimate.lam import sample_choices
    >>> import jax
    >>> game = strataq.games.coordination(2, 2, bonus=2.0)
    >>> counts = sample_choices(game, 1.5, 500, jax.random.key(0))
    >>> f = strataq.fit(game, counts, ci="profile")
    >>> f.identified
    True
    """
    if not isinstance(game, DenseTensorGame):
        raise TypeError(
            "game must be a DenseTensorGame; build one with "
            "strataq.DenseTensorGame([u1, u2]) or take one from strataq.games."
        )
    if method not in ("mle", "agreement", "bayes"):
        raise ValueError(f"method must be 'mle', 'agreement' or 'bayes'; got {method!r}")
    if ci not in ("bootstrap", "profile", "posterior", "none"):
        raise ValueError(f"ci must be 'bootstrap', 'profile', 'posterior' or 'none'; got {ci!r}")
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between 0 and 1")
    if n_boot < 1 or n_grid < 8:
        raise ValueError("n_boot must be >= 1 and n_grid >= 8")

    cfg = base_config().estimate
    seed_used = int(base_config().seeds.root if seed is None else seed)
    rng = np.random.default_rng(seed_used)
    warnings: list[str] = []
    refusals: list[str] = []
    provenance: dict[str, Any] = {"library_version": __version__, "seed": seed_used}

    columns = _as_columns(data)
    if columns is None:
        if by is not None:
            raise ValueError(
                "by= needs a tidy table: aggregated counts have already destroyed the "
                "grouping structure. Pass one row per observed choice instead."
            )
        panel = _panel_from_counts(game, data)
        warnings.append(
            "aggregated counts supplied: subject, round and treatment are already gone, so "
            "no clustered interval and no by= split are possible from this input. Pass a "
            "tidy long-form table to keep them."
        )
        provenance["data"] = f"aggregated counts ({panel.n_obs} choices)"
    else:
        panel = _panel_from_tidy(
            game,
            columns,
            player=player,
            action=action,
            subject=subject,
            by=by,
            warnings=warnings,
            provenance=provenance,
        )
        provenance["data"] = f"tidy long-form ({panel.n_obs} rows, {len(columns)} columns)"
        provenance["columns"] = panel.columns_used

    grid = np.geomspace(cfg.lam_min, cfg.lam_max, n_grid)
    sigmas, log_sigma = _grid_sigmas(game, grid)

    lam_hat, lo, hi, ci_method, loglik, kind, warns, extra = _fit_counts(
        game,
        panel,
        method=method,
        ci=ci,
        grid=grid,
        sigmas=sigmas,
        log_sigma=log_sigma,
        n_boot=n_boot,
        level=level,
        rng=rng,
    )
    warnings.extend(warns)

    if kind == "unidentified":
        refusals.append(
            "lambda is NOT IDENTIFIED by these data on this game: the log-likelihood is flat "
            f"across [{grid[0]:.4g}, {grid[-1]:.4g}], so no point estimate is defensible. The "
            "reported interval is the whole search window -- the widest true statement. The "
            "usual cause is a game whose QRE barely moves with lambda (a symmetric game below "
            "its bifurcation reads uniform at every lambda); a different game, or a "
            "perturbation experiment, is what would identify it."
        )

    # -- the two nested boundaries ------------------------------------------------------
    flat = _flat_counts(panel.counts)
    ll_unif = _loglik_uniform(panel.counts, game.num_actions)
    stat_unif = max(2.0 * (loglik - ll_unif), 0.0)
    lr_uniform = LRTest(
        name="vs uniform (lambda = 0)",
        stat=stat_unif,
        df=1,
        p=_chi2_sf(stat_unif, 1),
        p_boundary=0.5 * _chi2_sf(stat_unif, 1),
        note=(
            "lambda = 0 sits on the boundary of the parameter space, so the chi2(1) p-value "
            "is CONSERVATIVE; the correct asymptotic null is the 50:50 mixture "
            "0.5*chi2(0) + 0.5*chi2(1), whose p-value is the bracketed one (Self & Liang 1987)"
        ),
    )

    sigma_nash, lam_reached = _nash_limit(game, max(lam_hat, 1.0))
    nash_probs = np.concatenate([np.asarray(s, dtype=float) for s in sigma_nash])
    zeroed = bool(np.any((flat > 0) & (nash_probs < _ZERO_PROB)))
    if zeroed:
        stat_nash = math.inf
        nash_note = (
            "the limiting logit equilibrium assigns numerically zero probability to choices "
            f"that were observed (lambda -> inf approached at lambda = {lam_reached:.4g}), so "
            "the likelihood ratio is unbounded: Nash is rejected outright, and the statistic "
            "is reported as infinite rather than as a large finite number"
        )
    else:
        ll_nash = float(flat @ np.log(np.maximum(nash_probs, _TINY)))
        stat_nash = max(2.0 * (loglik - ll_nash), 0.0)
        nash_note = (
            "Nash is the lambda -> inf boundary, approached by warm-started continuation to "
            f"lambda = {lam_reached:.4g}; the chi2(1) p-value is conservative for the same "
            "boundary reason as the uniform test"
        )
    lr_nash = LRTest(
        name="vs Nash (lambda -> inf)",
        stat=stat_nash,
        df=1,
        p=_chi2_sf(stat_nash, 1),
        p_boundary=0.5 * _chi2_sf(stat_nash, 1),
        note=nash_note,
    )

    # -- by=: report the spread, do not average it --------------------------------------
    groups: list[GroupFit] = []
    homogeneity: LRTest | None = None
    if panel.group_panels:
        pooled_ll = 0.0
        for key, gpanel in panel.group_panels:
            g_lam, g_lo, g_hi, _m, g_ll, g_kind, g_warns, _x = _fit_counts(
                game,
                gpanel,
                method="mle",
                ci=ci,
                grid=grid,
                sigmas=sigmas,
                log_sigma=log_sigma,
                n_boot=n_boot,
                level=level,
                rng=rng,
            )
            pooled_ll += g_ll
            groups.append(
                GroupFit(
                    key=key,
                    lam_hat=g_lam,
                    ci_low=g_lo,
                    ci_high=g_hi,
                    kind=g_kind,
                    loglik=g_ll,
                    n_obs=gpanel.n_obs,
                    n_subjects=gpanel.n_subjects,
                    warnings=tuple(g_warns),
                )
            )
        df = max(len(groups) - 1, 1)
        stat_hom = max(2.0 * (pooled_ll - loglik), 0.0)
        p_hom = _chi2_sf(stat_hom, df)
        homogeneity = LRTest(
            name=f"homogeneity across {len(groups)} groups",
            stat=stat_hom,
            df=df,
            p=p_hom,
            p_boundary=None,
            note=(
                "H0: one lambda for every group. Rejection means the groups are NOT one "
                "population and a pooled lambda is a fiction -- read the per-group column, "
                "not the headline. lambda is an interior parameter here, so chi2 is exact "
                "asymptotically (no boundary correction)."
            ),
        )
        lams = np.array([g.lam_hat for g in groups if g.kind == "point"])
        if p_hom < 0.05:
            warnings.append(
                f"per-group lambda is heterogeneous (homogeneity LR p {_fmt_p(p_hom)}): the "
                "pooled lambda_hat above is a summary of groups that do not share one, and "
                "should not be quoted as 'the' rationality of this population"
            )
        if lams.size > 1 and float(lams.max() - lams.min()) / max(float(lams.mean()), 1e-12) > (
            cfg.agreement_flag_gap
        ):
            warnings.append(
                "per-group lambda spread exceeds the configured agreement gap "
                f"({cfg.agreement_flag_gap}): treat the groups separately"
            )

    warnings.append(
        "lambda is per payoff unit and is NOT scale-free: rescaling every payoff by s "
        "rescales lambda by 1/s (the scale fold). Compare lambdas only across identically "
        "scaled payoffs, or compare lambda_normalised = lambda * payoff_range instead."
    )
    if panel.n_obs < 100:
        warnings.append(
            f"n = {panel.n_obs} choices: the asymptotics behind these intervals and p-values "
            "are not credible at this sample size"
        )

    prange = float(np.asarray(game.payoff_range))
    provenance |= {
        "method": method,
        "ci": ci,
        "n_boot": n_boot if ci == "bootstrap" else 0,
        "grid": f"{n_grid} points, geomspace [{cfg.lam_min}, {cfg.lam_max}]",
        "payoff_range": prange,
        "lambda_normalised": lam_hat * prange,
        "solver": "damped logit fixed point, float64",
        "n_players": game.n_players,
        "n_actions": list(game.num_actions),
        "clustered_on": panel.columns_used.get("subject", "none"),
        "nash_limit_lambda": lam_reached,
    }
    if by is not None:
        provenance["by"] = panel.columns_used.get("by", str(by))

    freqs = tuple((np.asarray(c, dtype=float) / max(float(np.sum(c)), 1.0)) for c in panel.counts)
    return LambdaFit(
        lam_hat=lam_hat,
        ci_low=lo,
        ci_high=hi,
        ci_method=ci_method,
        ci_level=level,
        kind=kind,
        loglik=loglik,
        n_obs=panel.n_obs,
        n_subjects=panel.n_subjects,
        n_players=game.n_players,
        method=method,
        lr_nash=lr_nash,
        lr_uniform=lr_uniform,
        groups=tuple(groups),
        homogeneity=homogeneity,
        agreement=extra if method == "agreement" else None,
        warnings=tuple(dict.fromkeys(warnings)),
        refusals=tuple(refusals),
        provenance=provenance,
        _game=game,
        _freqs=freqs,
    )
