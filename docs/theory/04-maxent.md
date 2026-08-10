# 4 · MaxEnt — the softmax is not an assumption

Where does the exponential form come from? It can be *derived*, twice over.

**As a variational principle.** Ask: which distribution maximises expected
payoff *plus* a bonus for keeping options open,
$$\sigma^* = \arg\max_\sigma \;\mathbb{E}_\sigma[U] + \tfrac{1}{\lambda}H(\sigma),$$
with H the Shannon entropy? The answer is exactly the logit response. Slide
the 1/λ dial in the Lab and watch the optimum morph from uniform (entropy
dominates) to argmax (payoff dominates). Physicists will recognise free
energy F = U − TS with temperature T = 1/λ; the Gibbs variational principle,
verbatim, with payoff as negative energy.

**As a price of information.** Rational inattention (Matějka–McKay 2015)
derives the same logit from first principles: an agent who must *pay* for
information about payoffs, at a per-bit price, optimally ends up choosing
with exactly these probabilities — and λ is the inverse price of information.
"Noise" is then not error but economised attention.

Both derivations matter practically. The variational form gives the
log-partition function ψ = log Σ e^{λU}, whose gradient is the choice
distribution and whose Hessian is the choice covariance C — the exact
identities (verified to 10⁻¹²) on which every instrument in this project is
built: the susceptibility is λC, and everything else follows from there.
