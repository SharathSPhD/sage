# 5 · Gibbs and potential games — where the physics is exact

Take a congestion game: three resources, each slower the more players use it.
This game has a special property: there is a single function Φ over joint
choices — Rosenthal's *potential* — such that every player's incentive to
switch is exactly the change in Φ. The game is, in a precise sense, all of us
rolling downhill on one shared landscape.

Now let players revise logit-style: at random moments, one player re-picks an
action with probabilities softmax(λ · payoffs). In a potential game this
revision process is **exactly** heat-bath dynamics on Φ, and its stationary
distribution is **exactly** the Gibbs measure $\pi \propto e^{\lambda\Phi}$ —
λ playing inverse temperature, Nash appearing as the zero-temperature limit.
This is not an analogy; strataq verifies it to $2.5\times10^{-16}$ on
congestion games as a permanent regression test.

Everything thermal-equilibrium follows for free: detailed balance,
zero probability current, zero entropy production, symmetric (Onsager)
response. That is why potential games are the programme's **calibration
standard** — every meter must read exactly zero on them, and does, including
on logit route choice over a real road network, where the potential (the
Beckmann integral) is known analytically.

**And where it is not exact.** Break the potential — mix in a harmonic
component like rock–paper–scissors — and there is no landscape any more:
the same revision dynamics still settle into a stationary state, but one that
*circulates* and *dissipates* (explainer 6). The physics language stays honest
by staying tiered: Gibbs statements are `exact` for potential games; for
everything else what survives is the non-equilibrium machinery, which is where
the interesting readings live.
