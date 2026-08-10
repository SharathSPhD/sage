# 2 · The fixed point — everyone's noise responds to everyone's noise

Softmax response (explainer 1) is a one-player idea. The strategic leap is
circular: your rival's *distribution* over prices determines your expected
payoffs; your softmax over those payoffs is *your* distribution; that
determines *their* expected payoffs; and so on:

$$\sigma_B \to EU_A \to \sigma_A \to EU_B \to \sigma_B \to \cdots$$

A **quantal response equilibrium** is where this loop stops moving: every
player's distribution is the softmax of the payoffs induced by everyone
else's distribution. Existence follows from Brouwer; at λ = 0 it is the
uniform centroid; as λ grows it traces a path (a *branch*) that generically
terminates at a Nash equilibrium.

Computing it is its own subject. strataq ships three solvers that must agree
(and are tested against each other and against Gambit's homotopy tracer to
10⁻⁸–10⁻¹³): damped iteration (simple, fine at moderate λ), magnetic mirror
descent (last-iterate convergence even where naive iteration would cycle),
and a pseudo-arclength tracer that follows the whole branch — including
through bifurcations, where the number of equilibria changes and the most
interesting physics of this project begins.
