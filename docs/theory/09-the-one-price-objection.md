# 9 · The one-price objection — the strongest case against all of this

The objection, in its owner's own words: *"In reality you still have to choose
one price. So what's the use of a probability? You then tend to increase λ and
land on the same result as Nash."*

This objection is **correct** — for the question it asks. If your objective is
"what price should I charge tomorrow, given a payoff model I trust," then
compute expected profit for each price and take the argmax. QRE used as a
prescription adds nothing; it *contains* Nash as its λ → ∞ limit, and cranking
λ is just a slow way of taking an argmax. Nothing in this project disputes
that, and no panel in this app will pretend otherwise.

The reframe — and it is a reframe, not a rebuttal — is that the useful output
was never a randomised *own* price. It is two other things:

**1. The distribution over your rivals.** The expected profit you just
maximised — expected over *what*? Over the competitor's behaviour. A point
prediction of the rival ("they will charge £1.74") is empirically terrible;
observed play is dispersed. QRE supplies the strategically consistent
*distribution* of rival behaviour — everyone responding probabilistically to
everyone else's probabilities, at a precision λ estimated from data rather
than assumed. Your argmax is then taken against a credible average instead of
a brittle point. That is why every optimisation endpoint in this system
returns the competitor distribution alongside the recommended price: the
distribution *is* the deliverable.

**2. A positive model of the market you're in.** When £1.74 is estimated
best, firms in scanner data choose it — say — 61% of the time, not 100%. A
model whose predicted frequencies match observed frequencies lets you estimate
λ (how sharply this market responds to incentives), test whether conduct looks
competitive or coordinated, and measure the system-level quantities the rest
of this app is about — reciprocity, dissipation, distance to criticality.
None of those are available to a model that says $P(\text{best}) = 1$.

So: **prescription — the objection wins; prediction and measurement — the
distribution wins.** The honest division of labour, stated once, here.

*(Interactive panel: the same market shown twice — "argmax against a point
rival" vs "argmax against the QRE rival distribution" — with the profit
difference and its sensitivity to the rival assumption.)*
