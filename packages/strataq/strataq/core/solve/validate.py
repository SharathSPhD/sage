"""Cross-validation against pygambit — the ground-truth oracle for small games.

Gambit's logit solver (Turocy's predictor-corrector homotopy) is authoritative
on small finite games; we inherit its correctness rather than compete with it
(PROGRAMME v3 §7.1). This module converts DenseTensorGame → pygambit and
compares fixed-λ QRE profiles.

Optional dependency: install extra ``[gambit]`` (or dev group). Every consumer
must degrade gracefully when pygambit is absent.

References
----------
Turocy, GEB 2005 (the homotopy method). Tier: engineering validation of
K1-level solvers.
"""

from __future__ import annotations

import itertools

import jax.numpy as jnp

from strataq.finite.games.tensor import DenseTensorGame

try:  # pragma: no cover - trivially environment-dependent
    import pygambit  # type: ignore[import-untyped]

    HAVE_GAMBIT = True
except ImportError:  # pragma: no cover
    pygambit = None
    HAVE_GAMBIT = False


def to_pygambit(game: DenseTensorGame) -> pygambit.Game:
    """Convert to a pygambit strategic-form game (any N, small sizes)."""
    if not HAVE_GAMBIT:
        raise ImportError("pygambit is not installed; install the [gambit] extra")
    shape = game.num_actions
    g = pygambit.Game.new_table(list(shape))
    labels = [p.label for p in g.players]
    for profile in itertools.product(*(range(m) for m in shape)):
        for i in range(game.n_players):
            # pygambit 16.7 outcome access is by player *label*, not object.
            g[profile][labels[i]] = float(game.payoffs[i][profile])
    return g


def gambit_qre_sigma(game: DenseTensorGame, lam: float) -> tuple[jnp.ndarray, ...]:
    """The logit QRE at precision λ, computed by Gambit's homotopy tracer."""
    g = to_pygambit(game)
    result = pygambit.qre.logit_solve_lambda(g, lam=[lam])
    profile = result[0].profile
    out = []
    for player in g.players:
        out.append(jnp.asarray([float(profile[strategy]) for strategy in player.strategies]))
    return tuple(out)


def max_profile_gap(ours: tuple[jnp.ndarray, ...], theirs: tuple[jnp.ndarray, ...]) -> float:
    return max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(ours, theirs, strict=True))
