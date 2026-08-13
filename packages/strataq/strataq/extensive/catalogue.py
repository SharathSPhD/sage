"""Classic trees, built the same way a user would build one.

Every constructor returns a nested dict and hands it to
:meth:`~strataq.extensive.tree.ExtensiveGame.from_dict`, so the catalogue is also
the worked examples for the authoring format — there is no private construction
path these games get and yours does not.

The five here are the ones textbooks and the experimental literature actually
use: entry deterrence (backward induction in three nodes), the centipede
(backward induction against everything anyone observes), two-stage bargaining
(the discrete Rubinstein solution), Selten's horse (subgame perfection is not
enough) and Kuhn poker (the smallest poker with a known value).

References
----------
Selten 1978 (chain store / entry deterrence); Rosenthal 1981 and McKelvey–Palfrey
Econometrica 1992 (centipede); Ståhl 1972, Rubinstein 1982 (alternating offers);
Selten 1975 (the horse); Kuhn 1950 (Kuhn poker, value −1/18 to player 1).
Tier: exact.
"""

from __future__ import annotations

from typing import Any

from strataq.extensive.tree import ExtensiveGame

__all__ = [
    "CATALOGUE",
    "bargaining",
    "build",
    "centipede",
    "entry_deterrence",
    "kuhn_poker",
    "seltens_horse",
]


def entry_deterrence(
    *,
    monopoly: float = 2.0,
    duopoly: float = 1.0,
    fight: float = -1.0,
    outside: float = 0.0,
) -> ExtensiveGame:
    """An entrant decides whether to enter; the incumbent fights or accommodates.

    Backward induction: fighting costs the incumbent more than accommodating
    (``fight < duopoly``), so the threat is not credible, the incumbent
    accommodates and the entrant enters. The Nash equilibrium ("stay out because
    I would fight") survives only because the threat is never tested.
    """
    if fight >= duopoly:
        raise ValueError(
            f"entry deterrence needs fight ({fight}) below duopoly ({duopoly}) for the "
            "incumbent, or accommodation is not the credible reply"
        )
    return ExtensiveGame.from_dict(
        {
            "title": "entry deterrence",
            "players": ["Entrant", "Incumbent"],
            "root": {
                "label": "entry",
                "player": "Entrant",
                "infoset": "enter?",
                "actions": ["in", "out"],
                "children": [
                    {
                        "label": "response",
                        "player": "Incumbent",
                        "infoset": "respond",
                        "actions": ["fight", "accommodate"],
                        "children": [
                            {"label": "war", "payoffs": [fight, fight]},
                            {"label": "share", "payoffs": [duopoly, duopoly]},
                        ],
                    },
                    {"label": "stay out", "payoffs": [outside, monopoly]},
                ],
            },
        }
    )


def centipede(n_moves: int = 6, *, initial: float = 0.4, factor: float = 2.0) -> ExtensiveGame:
    """The centipede: take the bigger share now, or pass and let the pot grow.

    At move ``k`` the mover can take ``large`` and leave ``small``, or pass. The
    pot multiplies by ``factor`` each move, so passing is jointly profitable and
    privately costly — for one move. Backward induction says take at the first
    node; nobody does.

    The payoff convention is McKelvey–Palfrey's: taking at move ``k`` gives the
    taker ``initial · factor^k`` and the other player a quarter of that, and
    passing at the last move pays as if the player who *would* have moved next
    had taken. With the defaults, ``centipede(4)`` is the four-move experiment of
    McKelvey–Palfrey (1992): (0.40, 0.10), (0.20, 0.80), (1.60, 0.40),
    (0.80, 3.20) and (6.40, 1.60).
    """
    if int(n_moves) < 1:
        raise ValueError(f"n_moves must be >= 1, got {n_moves}")
    if float(factor) <= 1.0:
        raise ValueError(f"factor must be > 1 for the pot to grow, got {factor}")

    def terminal(move: int) -> dict[str, Any]:
        """The pot at ``move``, taken by whoever moves there."""
        pot = float(initial) * float(factor) ** move
        mover = move % 2
        payoffs = [0.0, 0.0]
        payoffs[mover] = pot
        payoffs[1 - mover] = pot / 4.0
        return {"label": f"end{move}", "payoffs": payoffs}

    def node(move: int) -> dict[str, Any]:
        if move == int(n_moves):
            return terminal(move)
        return {
            "label": f"m{move}",
            "player": move % 2,
            "infoset": f"move{move}",
            "actions": ["take", "pass"],
            "children": [terminal(move), node(move + 1)],
        }

    return ExtensiveGame.from_dict(
        {"title": f"centipede ({n_moves} moves)", "players": ["One", "Two"], "root": node(0)}
    )


def bargaining(*, discount: float = 0.8, n_offers: int = 5) -> ExtensiveGame:
    """Two-stage alternating offers over a discrete grid of shares of a unit pie.

    Player One proposes a split; Two accepts or rejects. On rejection the pie
    shrinks by ``discount`` and Two proposes; One accepts or rejects, and
    rejection ends it at zero. The grid is ``n_offers`` evenly spaced shares from
    0 to 1, and the subgame-perfect split is ``(1 − discount, discount)`` whenever
    ``1 − discount`` is on the grid.
    """
    if not 0.0 < float(discount) < 1.0:
        raise ValueError(f"discount must be in (0, 1), got {discount}")
    if int(n_offers) < 2:
        raise ValueError(f"n_offers must be >= 2, got {n_offers}")
    grid = [i / (int(n_offers) - 1) for i in range(int(n_offers))]
    delta = float(discount)

    def reply_to_counter(first: float, second: float) -> dict[str, Any]:
        """One accepts or rejects Two's counter of ``second`` for itself."""
        return {
            "label": f"one-replies-{first:.3f}-{second:.3f}",
            "player": "One",
            "infoset": f"one-replies-{first:.3f}-{second:.3f}",
            "actions": ["accept", "reject"],
            "children": [
                {"payoffs": [delta * (1.0 - second), delta * second]},
                {"payoffs": [0.0, 0.0]},
            ],
        }

    def counter(first: float) -> dict[str, Any]:
        """Two counter-offers after rejecting ``first`` — it knows what it turned down."""
        return {
            "label": f"two-counters-after-{first:.3f}",
            "player": "Two",
            "infoset": f"two-counters-after-{first:.3f}",
            "actions": [f"keep {share:.2f}" for share in grid],
            "children": [reply_to_counter(first, share) for share in grid],
        }

    def reply_to_offer(share: float) -> dict[str, Any]:
        """Two accepts or rejects One's offer of ``share`` for One."""
        return {
            "label": f"two-replies-{share:.3f}",
            "player": "Two",
            "infoset": f"two-replies-{share:.3f}",
            "actions": ["accept", "reject"],
            "children": [{"payoffs": [share, 1.0 - share]}, counter(share)],
        }

    return ExtensiveGame.from_dict(
        {
            "title": "two-stage bargaining",
            "players": ["One", "Two"],
            "root": {
                "label": "one-offers",
                "player": "One",
                "infoset": "one-offers",
                "actions": [f"keep {share:.2f}" for share in grid],
                "children": [reply_to_offer(share) for share in grid],
            },
        }
    )


def seltens_horse() -> ExtensiveGame:
    """The horse: three players, one shared information set, no proper subgames.

    Player One plays ``C`` or ``D``. After ``C``, Two plays ``c`` or ``d``. Player
    Three moves at an information set containing *both* the node after ``D`` and
    the node after ``Cd``, so it cannot tell which branch it is on.

    The example exists because ``(D, c, L)`` is a Nash equilibrium and — there
    being no proper subgame — it is subgame perfect, yet Two would deviate at its
    own information set if it were ever reached. Subgame perfection is not enough;
    sequential rationality is the fix, and the λ → ∞ limit of
    :func:`~strataq.extensive.aqre.agent_qre` is how this library gets there.
    """
    three = {
        "label": "three",
        "player": "Three",
        "infoset": "three-moves",
        "actions": ["L", "R"],
        "children": [{"payoffs": [3.0, 2.0, 2.0]}, {"payoffs": [0.0, 0.0, 0.0]}],
    }
    three_after_cd = {
        "label": "three-after-cd",
        "player": "Three",
        "infoset": "three-moves",
        "actions": ["L", "R"],
        "children": [{"payoffs": [4.0, 4.0, 0.0]}, {"payoffs": [0.0, 0.0, 1.0]}],
    }
    return ExtensiveGame.from_dict(
        {
            "title": "Selten's horse",
            "players": ["One", "Two", "Three"],
            "root": {
                "label": "one",
                "player": "One",
                "infoset": "one-moves",
                "actions": ["C", "D"],
                "children": [
                    {
                        "label": "two",
                        "player": "Two",
                        "infoset": "two-moves",
                        "actions": ["c", "d"],
                        "children": [{"payoffs": [1.0, 1.0, 1.0]}, three_after_cd],
                    },
                    three,
                ],
            },
        }
    )


_CARDS = ("J", "Q", "K")
_DEALS = tuple((i, j) for i in range(3) for j in range(3) if i != j)


def kuhn_poker(*, ante: float = 1.0, bet: float = 1.0) -> ExtensiveGame:
    """Kuhn poker: three cards, one card each, one betting round.

    Both players ante. Player One checks or bets; on a check Two may check
    (showdown) or bet, after which One folds or calls; on a bet Two folds or
    calls. The higher card wins the pot at a showdown. 55 nodes, 12 information
    sets, 64 pure strategies per player — small enough for the reduced normal
    form and large enough to be a real test.

    The game value to player One is ``−ante/18`` under optimal play.
    """
    if float(ante) <= 0 or float(bet) <= 0:
        raise ValueError("ante and bet must be positive")
    a = float(ante)
    b = float(bet)

    def showdown(deal: tuple[int, int], pot: float) -> dict[str, Any]:
        winner = 0 if deal[0] > deal[1] else 1
        payoffs = [-pot, -pot]
        payoffs[winner] = pot
        return {"label": f"show-{_CARDS[deal[0]]}{_CARDS[deal[1]]}-{pot:g}", "payoffs": payoffs}

    def fold(loser: int, pot: float) -> dict[str, Any]:
        payoffs = [pot, pot]
        payoffs[loser] = -pot
        return {"label": f"fold{loser}-{pot:g}", "payoffs": payoffs}

    def deal_subtree(deal: tuple[int, int]) -> dict[str, Any]:
        mine, theirs = _CARDS[deal[0]], _CARDS[deal[1]]
        # One checked, Two bet, One decides.
        one_faces_bet = {
            "label": f"one-{mine}-check-bet",
            "player": "One",
            "infoset": f"One:{mine}:checked-then-raised",
            "actions": ["fold", "call"],
            "children": [fold(0, a), showdown(deal, a + b)],
        }
        two_after_check = {
            "label": f"two-{theirs}-after-check",
            "player": "Two",
            "infoset": f"Two:{theirs}:after-check",
            "actions": ["check", "bet"],
            "children": [showdown(deal, a), one_faces_bet],
        }
        two_after_bet = {
            "label": f"two-{theirs}-after-bet",
            "player": "Two",
            "infoset": f"Two:{theirs}:after-bet",
            "actions": ["fold", "call"],
            "children": [fold(1, a), showdown(deal, a + b)],
        }
        return {
            "label": f"one-{mine}",
            "player": "One",
            "infoset": f"One:{mine}:open",
            "actions": ["check", "bet"],
            "children": [two_after_check, two_after_bet],
        }

    return ExtensiveGame.from_dict(
        {
            "title": "Kuhn poker",
            "players": ["One", "Two"],
            "root": {
                "label": "deal",
                "player": "chance",
                "actions": [f"{_CARDS[i]}{_CARDS[j]}" for i, j in _DEALS],
                "probs": [1.0 / len(_DEALS)] * len(_DEALS),
                "children": [deal_subtree(deal) for deal in _DEALS],
            },
        }
    )


CATALOGUE: dict[str, Any] = {
    "entry_deterrence": entry_deterrence,
    "centipede": centipede,
    "bargaining": bargaining,
    "seltens_horse": seltens_horse,
    "kuhn_poker": kuhn_poker,
}
"""Name → constructor, for config- and API-selectable trees."""


def build(name: str, **kwargs: Any) -> ExtensiveGame:
    """Build a catalogue tree by name."""
    if name not in CATALOGUE:
        raise ValueError(f"unknown tree {name!r}; the catalogue is {sorted(CATALOGUE)}")
    game: ExtensiveGame = CATALOGUE[name](**kwargs)
    return game
