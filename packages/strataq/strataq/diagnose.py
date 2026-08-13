"""strataq.diagnose -- one call, one verdict, every caveat recoverable.

This is the product spine described in ``docs/product/PRODUCT_v1.md``. It answers the
only question a non-specialist has: *what kind of system am I looking at?*

The design resolves one contradiction: the reading must be **decisive** (a single word a
practitioner acts on) and **non-decisive** (bounded, refusable, publishable). It is
resolved by system level -- the whole is one verdict, every part is a bounded measurement:

* ``repr(d)``      -> the verdict, four numbers, the tier, the refusal count.
* ``d.explain()``  -> every warning, band, null, seed and provenance stanza.
* ``d.snippet()``  -> a runnable reproduction of exactly this reading.
* ``d.plot()``     -> the point in the plane, on the reference cloud.

**Refusals become bounds.** Where a coordinate cannot be identified, ``diagnose`` reports
the widest true statement (an interval, a one-sided bound, or a named pair of quadrants it
cannot separate) rather than raising or returning ``None``. Only a coordinate that is
*unmeasurable in principle* from the given input is reported as absent, and then the
verdict names which quadrants remain live.

Input shapes (exactly one of the first two groups):

1. ``payoffs=..., lam=...``       -- a game you specify. Both coordinates exact.
   ``payoffs`` is anything :func:`strataq.toolkit.game_thermo` accepts (one payoff array
   per player) and, additionally, a :class:`~strataq.finite.games.tensor.DenseTensorGame`.
2. ``chi=..., series=...``        -- readings you already have, in any combination.

References
----------
The irreversibility-plane result (R and EPR share a zero and are otherwise independent);
``strataq.toolkit`` for the underlying instruments. Tier: derived.
"""

from __future__ import annotations

import textwrap
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np

from strataq import __version__
from strataq.toolkit import (
    _require_finite,
    game_thermo,
    irreversibility_test,
    reciprocity_read,
)

__all__ = ["QUADRANTS", "R_BANDS", "Coordinate", "Diagnosis", "Quadrant", "diagnose"]

Quadrant = Literal[
    "landscape", "driven landscape", "stalled whirlpool", "whirlpool", "undetermined"
]

# Calibrated band edges. R edges are toolkit.reciprocity_read's published bands, which are
# anchored on the road network (0.0), Blotto (0.12) and RPS (0.69) readings. The EPR edge is
# a *null* threshold, not an absolute scale: "circulating" means the reading escapes its
# detailed-balance null, never that it exceeds a fixed number.
R_BANDS = (0.02, 0.30)

# On the exact route EPR is computed in float64 from a stationary solve; an exact potential
# game returns O(1e-30) rather than a hard zero. Anything under this is read as zero, and
# the tolerance is stated in ``explain()`` rather than hidden.
EPR_ZERO_TOL = 1e-9

QUADRANTS: dict[Quadrant, dict[str, str]] = {
    "landscape": {
        "roman": "I",
        "physics": "reciprocal response, no persistent flux -- consistent with a potential game",
        "consequence": (
            "Comparative statics are trustworthy and pass-through is symmetric. A static "
            "model of the other agents is adequate. There is no cycle to time."
        ),
    },
    "driven landscape": {
        "roman": "II",
        "physics": "reciprocal response, persistent flux -- circulation driven from outside",
        "consequence": (
            "Something exogenous is cycling the system rather than the strategic "
            "interaction. Timing matters; re-engineering the strategic structure does not."
        ),
    },
    "stalled whirlpool": {
        "roman": "III",
        "physics": (
            "asymmetric response, no persistent flux -- structural asymmetry without circulation"
        ),
        "consequence": (
            "One agent structurally leads. Asymmetric pass-through is the exploitable "
            "object; effort spent timing a cycle is wasted, because there is no cycle."
        ),
    },
    "whirlpool": {
        "roman": "IV",
        "physics": "asymmetric response and persistent flux -- a non-equilibrium steady state",
        "consequence": (
            "Both structure and timing matter. This is the regime where optimising against "
            "a static model of the other agents is worst; model their distribution."
        ),
    },
    "undetermined": {
        "roman": "-",
        "physics": "at least one coordinate is not identified by the supplied data",
        "consequence": (
            "The reading narrows the possibilities but does not pick one. See "
            "`.live_quadrants` for what remains and `.explain()` for what would settle it."
        ),
    },
}


@dataclass(frozen=True)
class Coordinate:
    """One axis of the plane. ``value`` may be None when only a bound is available."""

    name: str
    value: float | None
    lo: float | None
    hi: float | None
    kind: Literal["point", "interval", "upper_bound", "lower_bound", "absent"]
    method: str
    warnings: tuple[str, ...] = ()

    def band(self) -> str:
        if self.kind == "absent":
            return "not identified from the supplied data"
        if self.kind == "upper_bound" and self.hi is not None:
            return f"<= {self.hi:.4g}"
        if self.kind == "lower_bound" and self.lo is not None:
            return f">= {self.lo:.4g}"
        if self.value is None:
            if self.lo is None or self.hi is None:
                return "not identified from the supplied data"
            return f"[{self.lo:.4g}, {self.hi:.4g}]"
        if self.lo is None or self.hi is None:
            return f"{self.value:.4g}"
        return f"{self.value:.4g}  [{self.lo:.4g}, {self.hi:.4g}]"


@dataclass(frozen=True)
class Diagnosis:
    """A located reading: one quadrant, two bounded coordinates, every caveat kept."""

    quadrant: Quadrant
    live_quadrants: tuple[Quadrant, ...]
    response: Coordinate
    dissipation: Coordinate
    alpha: float | None
    lam: float | None
    tier: Literal["instant", "fast", "certified"]
    warnings: tuple[str, ...] = ()
    refusals: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    _repro: dict[str, Any] = field(default_factory=dict, repr=False)

    # -- the decisive layer ------------------------------------------------------------
    def __repr__(self) -> str:
        q = QUADRANTS[self.quadrant]
        head = f"Diagnosis: {self.quadrant.upper()}  (quadrant {q['roman']})"
        lines = [
            head,
            f"  response asymmetry   R = {self.response.band()}",
            f"  dissipation        EPR = {self.dissipation.band()}",
        ]
        if self.alpha is not None:
            lines.append(f"  harmonic fraction    a = {self.alpha:.4g}")
        lam = "n/a" if self.lam is None else f"{self.lam:g}"
        lines.append(
            f"  read at lambda = {lam} - tier: {self.tier}"
            f" - {len(self.refusals)} refusal(s), {len(self.warnings)} warning(s)"
        )
        lines.append("  -> " + textwrap.shorten(q["consequence"], 96, placeholder=" ..."))
        if self.quadrant == "undetermined":
            lines.append("     live: " + ", ".join(self.live_quadrants))
        lines.append("  (call .explain() for the evidence, .snippet() to reproduce)")
        return "\n".join(lines)

    # -- the recoverable layer ---------------------------------------------------------
    def explain(self) -> str:
        """Everything discarded from the headline, recovered in full."""
        q = QUADRANTS[self.quadrant]
        out = [
            "WHY THIS READING",
            "=" * 68,
            f"verdict      : {self.quadrant}  (quadrant {q['roman']})",
            f"physics      : {q['physics']}",
            f"consequence  : {q['consequence']}",
        ]
        if self.quadrant == "undetermined":
            out.append(f"live         : {', '.join(self.live_quadrants)}")
        out += ["", "COORDINATES", "-" * 68]
        for c in (self.response, self.dissipation):
            out += [
                f"{c.name}",
                f"    reading  : {c.band()}   ({c.kind})",
                f"    method   : {c.method}",
            ]
            out += [f"    warning  : {w}" for w in c.warnings]
        out += ["", "BANDS AND ANCHORS", "-" * 68]
        out += [
            f"R band edges          : {R_BANDS[0]} / {R_BANDS[1]} (calibrated)",
            "  road network (exact potential) R = 5.6e-17",
            "  Colonel Blotto                 R = 0.12",
            "  rock-paper-scissors            R = 0.69-0.87",
            "EPR threshold         : escape from a detailed-balance null, not a fixed value",
            f"EPR numerical zero    : {EPR_ZERO_TOL:g} on the exact route (float64 solve)",
            "",
            "SCOPE",
            "-" * 68,
            "R = 0 iff the normalised game is potential, at every lambda.",
            "The MAGNITUDE of R scales with lambda -- never quote it without the lambda it",
            "was read at, and never compare two R magnitudes read at different lambda.",
            "R is a ratio of Frobenius norms, unbounded above; values > 1 occur.",
            "The two coordinates share a zero and are otherwise independent observables",
            "(see the irreversibility-plane result). Do not read one as a proxy for the other.",
        ]
        if self.refusals:
            out += ["", "REFUSALS (reported as bounds, not silences)", "-" * 68]
            out += [f"  - {r}" for r in self.refusals]
        if self.warnings:
            out += ["", "WARNINGS", "-" * 68] + [f"  - {w}" for w in self.warnings]
        out += ["", "PROVENANCE", "-" * 68]
        out += [f"  {k:<20}: {v}" for k, v in self.provenance.items()]
        return "\n".join(out)

    def snippet(self) -> str:
        """A runnable reproduction of exactly this reading."""
        kind = self._repro.get("kind", "unknown")
        if kind == "game":
            return textwrap.dedent(
                f"""\
                import numpy as np, strataq

                payoffs = [np.array(m) for m in {self._repro["payoffs"]!r}]
                d = strataq.diagnose(payoffs=payoffs, lam={self._repro["lam"]!r})
                print(d)
                print(d.explain())
                """
            )
        if kind == "readings":
            return textwrap.dedent(
                f"""\
                import numpy as np, strataq

                d = strataq.diagnose(
                    chi={self._repro.get("chi")!r},
                    chi_se={self._repro.get("chi_se")!r},
                    series=...,                    # length {self._repro.get("n_series")}
                    seed={self._repro.get("seed")!r},
                )
                print(d)
                """
            )
        return "# reproduction snippet unavailable for this input shape\n# import strataq"

    def plot(self, ax: Any = None, reference: bool = True) -> Any:
        """Draw this point in the plane. Requires ``strataq[viz]`` (matplotlib)."""
        from strataq.viz import plot_plane

        return plot_plane([self], ax=ax, reference=reference)

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serialisable view of the whole reading (nothing dropped)."""
        return {
            "quadrant": self.quadrant,
            "live_quadrants": list(self.live_quadrants),
            "response": asdict(self.response),
            "dissipation": asdict(self.dissipation),
            "alpha": self.alpha,
            "lam": self.lam,
            "tier": self.tier,
            "warnings": list(self.warnings),
            "refusals": list(self.refusals),
            "provenance": dict(self.provenance),
        }


# --------------------------------------------------------------------------------------


def _criticality(mats: list[Any], lam: float) -> dict[str, Any] | None:
    """Spectral state of the strategic resolvent at the logit fixed point.

    Returns ``None`` if the spectrum cannot be formed (the caller then proceeds without
    the guard rather than failing). ``near_critical`` is the library's own flag, so this
    guard tracks the solver's notion of criticality rather than inventing a second one.

    References
    ----------
    ``distance_to_criticality = 1 - rho(SB)``; PROGRAMME v3 §8.5 requires refusing to
    report chi near the singularity rather than returning a large number. Tier: exact.
    """
    try:
        from strataq.core.solve.fixedpoint import logit_qre
        from strataq.finite.games.tensor import DenseTensorGame
        from strataq.finite.response.spectral import strategic_spectrum

        game = DenseTensorGame(payoffs=[jnp.asarray(m) for m in mats])
        info = strategic_spectrum(game, logit_qre(game, lam))
        return {
            "rho": float(info.rho),
            "distance": float(info.distance_to_criticality),
            "near_critical": bool(info.near_critical),
            "bifurcation_type": int(info.bifurcation_type),
        }
    except Exception:  # pragma: no cover - guard must never itself break a reading
        return None


def _classify(r: Coordinate, e: Coordinate) -> tuple[Quadrant, tuple[Quadrant, ...], list[str]]:
    """Map two coordinates to a quadrant, degrading to a live set rather than guessing."""
    notes: list[str] = []

    def side_r() -> Literal["low", "high", "unknown"]:
        if r.kind == "absent":
            return "unknown"
        lo = r.lo if r.lo is not None else r.value
        hi = r.hi if r.hi is not None else r.value
        if hi is not None and hi < R_BANDS[0]:
            return "low"
        if lo is not None and lo > R_BANDS[0]:
            return "high"
        notes.append(
            f"R interval straddles the calibrated band edge {R_BANDS[0]}; the reciprocal "
            "and non-reciprocal halves of the plane are not separated by this data."
        )
        return "unknown"

    def side_e() -> Literal["low", "high", "unknown"]:
        if e.kind == "absent":
            return "unknown"
        if e.kind == "lower_bound":
            return "high"
        if e.kind == "upper_bound":
            return "low"
        if e.value is None:
            return "unknown"
        return "high" if e.value > EPR_ZERO_TOL else "low"

    sr, se = side_r(), side_e()
    table: dict[tuple[str, str], Quadrant] = {
        ("low", "low"): "landscape",
        ("low", "high"): "driven landscape",
        ("high", "low"): "stalled whirlpool",
        ("high", "high"): "whirlpool",
    }
    if sr != "unknown" and se != "unknown":
        q = table[(sr, se)]
        return q, (q,), notes

    live = tuple(
        v
        for (a, b), v in table.items()
        if (sr == "unknown" or a == sr) and (se == "unknown" or b == se)
    )
    return "undetermined", live, notes


def _payoff_arrays(payoffs: Any) -> list[np.ndarray]:
    """Accept anything ``game_thermo`` accepts, plus a DenseTensorGame.

    ``game_thermo`` iterates its argument as one payoff array per player; a
    ``DenseTensorGame`` is an equinox Module and is NOT iterable, so unwrap it here.
    """
    raw = getattr(payoffs, "payoffs", payoffs)
    if isinstance(raw, np.ndarray) and raw.ndim > 0 and not isinstance(raw[0], np.ndarray):
        raise TypeError(
            "payoffs= must be one payoff array PER PLAYER (or a DenseTensorGame), not a "
            "single matrix: a two-player game is [u1, u2]."
        )
    try:
        mats = [np.asarray(u, dtype=float) for u in raw]
    except TypeError as exc:  # not iterable at all
        raise TypeError(
            "payoffs= must be a sequence of payoff arrays (one per player) or a "
            f"DenseTensorGame; got {type(payoffs).__name__}."
        ) from exc
    if not mats:
        raise ValueError("payoffs= is empty: a game needs at least one player's payoffs.")
    # toolkit._require_finite takes (name, array) -- not (array, name).
    for i, m in enumerate(mats):
        _require_finite(f"payoff matrix for player {i}", m)
    return mats


def _epr_from_series(
    series: Sequence[float],
    *,
    n_bins: int,
    n_surrogates: int,
    alpha_level: float,
    seed: int,
) -> Coordinate:
    """EPR as a certified one-sided bound from the reversibilized-Markov null.

    ``irreversibility_test`` returns a :class:`~strataq.thermo.nulls.ReversibilizedNullResult`
    with fields ``statistic``, ``null_quantile``, ``null_median``, ``p_value``, ``detected``,
    ``n_surrogates``, ``null_mismatch_low``. It carries no ``warnings`` list of its own, so
    the honesty text is assembled here.
    """
    res = irreversibility_test(
        np.asarray(series, dtype=float),
        n_bins=n_bins,
        n_surrogates=n_surrogates,
        alpha_level=alpha_level,
        seed=seed,
    )
    warn: list[str] = [
        "EPR from a series is a KLD irreversibility rate on a phase embedding "
        f"({n_bins} value bins x direction of last change), not the Schnakenberg EPR of a "
        "known generator; it is comparable across series only at the same embedding",
    ]
    if res.null_mismatch_low:
        warn.append(
            "the observed statistic sits BELOW the null's 5% quantile: the reversible "
            "surrogate model is a poor description of this series, so the bound is "
            "reported but the null itself should be treated as suspect"
        )
    if res.detected:
        # The certified statement is that the reading escapes its null -- a LOWER bound.
        return Coordinate(
            name="dissipation  EPR (nats/step)",
            value=float(res.statistic),
            lo=float(res.null_quantile),
            hi=None,
            kind="lower_bound",
            method=(
                f"reversibilized-Markov null, {res.n_surrogates} surrogates, "
                f"alpha={alpha_level}, p={res.p_value:.4g}; the certified statement is the "
                "escape from the null, not the point value"
            ),
            warnings=tuple(warn),
        )
    return Coordinate(
        name="dissipation  EPR (nats/step)",
        value=float(res.statistic),
        lo=None,
        hi=float(res.null_quantile),
        kind="upper_bound",
        method=(
            f"reversibilized-Markov null, {res.n_surrogates} surrogates; no escape at "
            f"alpha={alpha_level} (p={res.p_value:.4g}). This bounds EPR ABOVE by the null "
            "quantile; it is NOT evidence that EPR is zero."
        ),
        warnings=tuple(warn),
    )


def diagnose(
    payoffs: Any | None = None,
    *,
    lam: float | None = None,
    chi: Any | None = None,
    chi_se: Any | None = None,
    series: Sequence[float] | None = None,
    n_bins: int = 3,
    n_surrogates: int = 200,
    alpha_level: float = 0.01,
    seed: int = 0,
    tier: Literal["instant", "fast", "certified"] = "certified",
) -> Diagnosis:
    """Locate a system in the irreversibility plane.

    Parameters
    ----------
    payoffs
        A game: one payoff array per player (``[u1, u2]``) or a
        :class:`~strataq.finite.games.tensor.DenseTensorGame`. Requires ``lam``.
    lam
        The logit precision the game is read at. R's magnitude scales with it.
    chi, chi_se
        A measured cross-response matrix and (strongly recommended) its elementwise
        standard errors. Without ``chi_se`` the band assignment is refused as indefensible.
    series
        An observed scalar time series of the joint state, for the dissipation coordinate.

    Examples
    --------
    >>> import strataq
    >>> rps = strataq.games.rock_paper_scissors()
    >>> d = strataq.diagnose(rps, lam=1.5)
    >>> d.quadrant
    'whirlpool'
    """
    if payoffs is None and chi is None and series is None:
        raise ValueError(
            "diagnose needs one of: payoffs= (a game), chi=/series= (readings you have). "
            "Nothing was supplied."
        )

    warnings: list[str] = []
    refusals: list[str] = []
    prov: dict[str, Any] = {"library_version": __version__, "seed": seed}

    # -- route 1: a game we can solve exactly -------------------------------------------
    if payoffs is not None:
        if lam is None:
            raise ValueError("payoffs= requires lam= (the logit precision).")
        mats = _payoff_arrays(payoffs)
        read = game_thermo(list(mats), lam=float(lam))

        # R is read through (I - SB)^-1. Near rho(SB) = 1 that operator is singular and the
        # number it returns is not a measurement of anything -- it is rounding. Refuse to
        # report it as a point, and say which quadrants remain live. Found on aarch64 /
        # jax 0.11, where coordination(2, 3, bonus=2.0) at lambda=1.5 sits exactly on the
        # pitchfork (rho = 1 + 4e-16) and a POTENTIAL game read as a whirlpool.
        crit = _criticality(mats, float(lam))
        if crit is not None and crit["near_critical"]:
            r_coord = Coordinate(
                name="response asymmetry  R",
                value=None,
                lo=None,
                hi=None,
                kind="absent",
                method=(
                    "REFUSED at criticality: rho(SB) = "
                    f"{crit['rho']:.12f}, distance to criticality "
                    f"{crit['distance']:.2e}. The resolvent (I - SB)^-1 is singular here, "
                    "so any R computed through it is rounding, not a reading."
                ),
            )
            refusals.append(
                "R not identified: this game sits at (or beyond) the criticality of the "
                f"strategic resolvent, rho(SB) = {crit['rho']:.12f}. R is undefined there. "
                "Move lambda away from the critical value and re-read; "
                "`strataq.critical_lambda(game)` locates it. The refused value would have "
                f"been {float(read.r):.6g}, which is platform-dependent noise."
            )
        else:
            r_coord = Coordinate(
                name="response asymmetry  R",
                value=float(read.r),
                lo=None,
                hi=None,
                kind="point",
                method=(
                    "exact: reciprocity defect of the equilibrium response chi_equilibrium at "
                    "the logit fixed point, on the Helmert tangent space"
                ),
            )
        e_coord = Coordinate(
            name="dissipation  EPR (nats/step)",
            value=float(read.epr),
            lo=None,
            hi=None,
            kind="point",
            method="exact: Schnakenberg EPR of the dense Glauber generator's stationary state",
        )
        alpha_val = float(read.alpha)
        n_states = int(np.prod(mats[0].shape))
        prov |= {
            "route": "exact (game supplied)",
            "solver": "damped logit fixed point, float64",
            "n_players": len(mats),
            "n_actions": [int(x) for x in mats[0].shape],
            "n_joint_states": n_states,
            "game_thermo_verdict": read.verdict,
        }
        if crit is not None:
            prov |= {
                "rho_SB": crit["rho"],
                "distance_to_criticality": crit["distance"],
                "bifurcation_type": crit["bifurcation_type"],
            }
        # game_thermo now carries its own honesty text (lambda scaling, dense-generator
        # cost); forward it rather than restating it.
        warnings += list(read.warnings)
        quadrant, live, notes = _classify(r_coord, e_coord)
        warnings += notes
        return Diagnosis(
            quadrant=quadrant,
            live_quadrants=live,
            response=r_coord,
            dissipation=e_coord,
            alpha=alpha_val,
            lam=float(lam),
            tier=tier,
            warnings=tuple(warnings),
            refusals=tuple(refusals),
            provenance=prov,
            _repro={
                "kind": "game",
                "payoffs": [m.tolist() for m in mats],
                "lam": float(lam),
            },
        )

    # -- route 2: readings supplied, in any combination ---------------------------------
    if chi is not None:
        rr = reciprocity_read(
            np.asarray(chi, dtype=float),
            chi_se=None if chi_se is None else np.asarray(chi_se, dtype=float),
            seed=seed,
        )
        has_ci = rr.ci_low is not None and rr.ci_high is not None
        r_coord = Coordinate(
            name="response asymmetry  R",
            value=float(rr.r),
            lo=float(rr.ci_low) if has_ci and rr.ci_low is not None else None,
            hi=float(rr.ci_high) if has_ci and rr.ci_high is not None else None,
            kind="interval" if has_ci else "point",
            method=(
                "R = ||chi - chi^T|| / ||chi + chi^T|| from a supplied response matrix"
                + (
                    "; Monte-Carlo 95% interval from chi_se"
                    if has_ci
                    else "; NO interval -- supply chi_se to get one, and do not publish a "
                    "classification without it"
                )
            ),
            warnings=tuple(rr.warnings),
        )
        prov["reciprocity_verdict"] = rr.verdict
        if not has_ci:
            refusals.append(
                "No standard errors supplied for chi, so R is a point read with no "
                "interval. The band assignment is not defensible without chi_se."
            )
    else:
        r_coord = Coordinate(
            name="response asymmetry  R",
            value=None,
            lo=None,
            hi=None,
            kind="absent",
            method="no response matrix supplied",
        )
        refusals.append(
            "R not identified: no chi supplied. R needs a perturbation experiment -- a "
            "cost or payoff shock to one agent with the others' responses observed."
        )

    if series is not None:
        e_coord = _epr_from_series(
            series,
            n_bins=n_bins,
            n_surrogates=n_surrogates,
            alpha_level=alpha_level,
            seed=seed,
        )
        n_series: int | None = len(series)
        if n_series is not None and n_series < 300:
            warnings.append(
                f"n={n_series}: the irreversibility test has >=80% power only at n>=300; "
                "a non-detection at this n is underpowered, not evidence of equilibrium."
            )
    else:
        e_coord = Coordinate(
            name="dissipation  EPR (nats/step)",
            value=None,
            lo=None,
            hi=None,
            kind="absent",
            method="no trajectory supplied",
        )
        n_series = None
        refusals.append(
            "EPR not identified: no series supplied. EPR needs an observed sequence of "
            "joint states -- a time series of the profile, not a cross-section."
        )

    prov |= {"route": "observational (readings supplied)", "n_series": n_series}
    quadrant, live, notes = _classify(r_coord, e_coord)
    warnings += notes
    if quadrant == "undetermined":
        warnings.append(
            "Undetermined is a bound, not a failure: the live set below is a true "
            "statement about this system given this data."
        )
    if lam is None and chi is not None:
        warnings.append(
            "R's magnitude scales with the agents' payoff sensitivity (lambda), which is "
            "unknown for a measured chi: only the ZERO test is lambda-free, so compare "
            "this magnitude only against readings taken under matched conditions."
        )
    return Diagnosis(
        quadrant=quadrant,
        live_quadrants=live,
        response=r_coord,
        dissipation=e_coord,
        alpha=None,
        lam=None if lam is None else float(lam),
        tier=tier,
        warnings=tuple(warnings),
        refusals=tuple(refusals),
        provenance=prov,
        _repro={
            "kind": "readings",
            "chi": None if chi is None else np.asarray(chi, dtype=float).tolist(),
            "chi_se": None if chi_se is None else np.asarray(chi_se, dtype=float).tolist(),
            "n_series": n_series,
            "seed": seed,
        },
    )
