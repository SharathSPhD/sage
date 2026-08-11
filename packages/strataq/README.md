# strataq

**Thermodynamic instruments for strategic systems.** Quantal-response
equilibria, potential/harmonic game decomposition, entropy-regularised
response, and non-equilibrium strategic dynamics — in JAX, with a
plain-numpy toolkit on top.

Some strategic systems settle (landscapes); some circulate forever
(whirlpools). From the outside they can look identical. strataq is a set of
calibrated meters that tell them apart — from a payoff matrix if you have
one, from measured responses or a raw time series if you don't.

```python
import strataq.toolkit as tk
```

## Three questions, one call each

**Is my system a landscape or a whirlpool?** — from any measured
cross-response matrix (e.g. cost pass-through between two firms):

```python
read = tk.reciprocity_read([[1.07, 0.003], [0.0005, 0.97]])
read.r  # 0.0011 — this is the actual Dominick's grocery reading
read.verdict  # "reciprocal (landscape-like): ..."
read.warnings  # the honesty notes travel with the number
```

**Is my time series irreversibly driven?** — the instrument that found the
diurnal loop in day-ahead electricity prices:

```python
verdict = tk.irreversibility_test(weekly_prices)  # any scalar series, ≥ a few hundred points
verdict.detected, verdict.p_value  # vs a reversible null with matched persistence
```

**How payoff-sensitive are my agents?** — a Bayesian posterior over the
logit rationality λ from observed choice frequencies:

```python
est = tk.estimate_rationality([u1, u2], counts=[[412, 95, 493], [301, 402, 297]])
est.mean, (est.ci_low, est.ci_high)  # calibrated 95% interval
est.warnings  # flags flat likelihoods instead of quoting noise
```

And for a game you can write down, the full dashboard:

```python
tk.game_thermo([u1, u2], lam=1.5)  # harmonic fraction α, reciprocity ℛ, dissipation
```

## The full library

Under the facade: three QRE solvers (damped fixed-point, mirror descent,
arclength branch tracer, pygambit-validated), the Hodge potential/harmonic
decomposition, equilibrium response matrices χ = (I − SB)⁻¹S, exact Glauber
dynamics with currents and entropy production, trajectory estimators (KLD,
certified TUR bounds), Hatano–Sasa quench thermodynamics, surrogate nulls,
Bayesian estimation with an EFE experiment-selection loop, and domain
plugins (Colonel Blotto, traffic networks, electricity markets, retail
pricing).

Every claim the library makes is backed by a gated unit with an adversarial
review on record, and every number regenerates from fixed seeds
(`make reproduce` in the [SAGE monorepo](https://github.com/SharathSPhD/sage)).
The interactive companion lives at
[sage-labs.vercel.app](https://sage-labs.vercel.app).

## Install

```bash
pip install strataq
```

Published on PyPI: [pypi.org/project/strataq](https://pypi.org/project/strataq/).
For the unreleased main branch (pip ≥ 21.1):

```bash
pip install "strataq @ git+https://github.com/SharathSPhD/sage.git#subdirectory=packages/strataq"
```

Python ≥ 3.11. CPU JAX by default; float64 is enabled by the library.

`import strataq` — never `import sage` (SageMath owns that name).

## License

Apache-2.0. Dominick's-derived artifacts carry CC-BY-NC-4.0 (see
`strataq.domains.pricing`).
