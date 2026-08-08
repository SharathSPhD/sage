# 7 · Reciprocity — the same number both ways

Here is a measurement you can do on any strategic system without knowing its
payoffs: **poke player 1, read player 2; poke player 2, read player 1.**
Nudge one player's incentives by a small observable amount — a toll on a road,
a cost shock to one firm — let everyone re-equilibrate, and record how much
everyone *else* moved.

The matrix of all such readings is the equilibrium susceptibility
$\chi^{\mathrm{eq}} = (I - SB)^{-1}S$: the response of the whole system's play
to a payoff perturbation, *after* strategic feedback settles. In physics, the
symmetry of such a matrix is Onsager reciprocity, and it is a signature of
thermodynamic equilibrium.

**The result that makes this a meter** (Result 2, tier: derived): the
strategic feedback loop neither creates nor destroys reciprocity.
$\chi^{\mathrm{eq}}$ is symmetric **exactly when** the game's normalised
payoff structure has zero harmonic (circulating) component — that is, exactly
when the game is a potential game. One would generically expect a resolvent
$(I-SB)^{-1}$ to scramble symmetry; it doesn't. So the *observable* response
matrix inherits the symmetry of the *unobservable* payoff operator, and the
reciprocity defect
$\mathcal{R} = \|\chi - \chi^\top\| / \|\chi + \chi^\top\|$
is an operational test of potentiality requiring no payoff knowledge — and
whether it reads zero doesn't depend on λ, so there is no noise parameter that
can absorb an asymmetry.

**Calibration, measured:** on exact potential games — including logit route
choice on the real Sioux Falls road network, where the potential is known
analytically — ℛ reads $10^{-16}$: zero to machine precision. On
rock–paper–scissors it reads 0.69; on matching pennies 1.2 (yes, above 1 —
ℛ is a norm ratio, and a value above 1 means the circulating part of the
response *dominates* the reciprocal part).

**What ℛ is not** (measured the hard way): its *magnitude* grows with λ, and
in the near-harmonic regime it stops tracking dissipation — we proposed a fix
and refuted it ourselves. ℛ answers one question exactly — *"is this system
potential?"* — at every λ and every α. For *"how hard is it circulating?"*
you need the entropy-production meter. Two instruments, two questions.

*(Interactive controls in the Lab: poke-player selector, poke size, the two
cross-readings side by side, ℛ gauge tracking the α slider.)*
