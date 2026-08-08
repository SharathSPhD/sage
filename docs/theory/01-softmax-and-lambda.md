# 1 · Softmax and λ — where the randomness comes from

Suppose two petrol stations can each charge one of five prices, and given your
rival's behaviour, your expected profits are:

| Price | £1.70 | £1.72 | £1.74 | £1.76 | £1.78 |
|---|---|---|---|---|---|
| Expected profit | 102 | 108 | **115** | 113 | 104 |

Nash says: charge £1.74, with probability one. A 2-point profit difference and
a 13-point difference get the same treatment — total.

Quantal response says instead: choose each price with probability rising in its
payoff,

$$P(p) \;=\; \frac{e^{\lambda\,EU(p)}}{\sum_{p'} e^{\lambda\,EU(p')}},$$

which is exactly the softmax of the payoffs at *precision* λ. At λ = 0 every
price is equally likely; as λ → ∞ the distribution collapses onto the best
response, and Nash reappears as the zero-temperature limit. In between, the
**magnitude of the payoff difference matters**: £1.76 (2 points behind) keeps
real probability; £1.70 (13 behind) nearly none.

**What λ is, and is not.** Mathematically λ converts payoff differences into
log-odds: two prices 5 apart in profit have odds $e^{5\lambda}$. It has a
rational-inattention microfoundation — the inverse shadow price of Shannon
information (Matějka–McKay 2015) — so "low λ" can mean *information is
expensive here*, not *people are dumb here*. And it is **not scale-free**: pay
in pennies instead of pounds and λ must shrink by 100 to describe the same
behaviour. Every strataq output therefore reports both `lam` and
`lambda_normalised = λ × payoff range`.

**Where does the randomness live?** Not necessarily in a coin flip. Write the
payoff as $\tilde U(p) = EU(p) + \varepsilon_p$ with $\varepsilon$ the things
the modeller cannot see — private forecasts, manager idiosyncrasy, stale
competitor data. If $\varepsilon$ is extreme-value distributed, the *observed
frequencies* of a deterministic decision-maker are exactly the softmax above.
The firm picks one price; we, who don't see its ε, predict a distribution.

*(Interactive controls in the Lab: payoff bars → probability bars; λ slider
from 0 to ∞; watch the collapse to Nash.)*
