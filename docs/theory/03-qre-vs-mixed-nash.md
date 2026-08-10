# 3 · QRE vs mixed Nash — sensitivity, not indifference

Mixed-strategy Nash also outputs probabilities, so the two are often
confused. The logic could not be more different.

A mixed Nash requires **indifference**: you randomise 60/40 only if both
actions have *exactly equal* expected payoff — otherwise you'd purify. The
probabilities are pinned down by making your *opponent* indifferent, which
produces famously odd comparative statics (your mix depends on their payoffs,
not yours).

QRE requires **payoff sensitivity**: with EU(A) = 10 and EU(B) = 8, B keeps
positive probability — smaller than A's, by an amount governed by λ. The
probability *ordering follows the payoff ordering*, always (this is one of
the regular-QRE axioms). Nothing needs to be indifferent to anything.

The practical consequence: QRE's predicted frequencies respond smoothly to
payoff changes — which is exactly what makes the susceptibility χ = dσ*/dh a
well-defined instrument (explainer 7 measures its symmetry). Nash's
correspondence is piecewise-constant-then-jumping; there is no meter to build
on it.
