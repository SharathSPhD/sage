# First data contact — electricity prices through the dissipation meter

The estimators were built synthetic-first (gates `thermo.estimators`,
`estimate.lambda`); `domains.electricity` is where they first touch a real
market. The loader pulls CAISO OASIS locational marginal prices — day-ahead
hourly and real-time 5-minute — with disk caching, a 90% coverage guard
(silent gaps are how bad readings happen), and rate-limit backoff. ERCOT was
the plan; its "public" endpoints reject programmatic clients (HTTP 403), so
CAISO — genuinely open, no key — is the first data domain.

## The question, and the honest answer (F-0008)

*Is a market price series time-reversible (an equilibrium-like fluctuation)
or measurably driven (a cycle that dissipates)?* Three methodological facts
had to be established before the answer could be trusted:

1. **Price-value discretization is structurally blind to loop
   irreversibility.** A periodic drive retraces the same 1-D price path up
   and down; its time-reversal visits identical value transitions. Proven in
   a unit test (a noisy 24h sine reads 6×10⁻⁵ in value space, 0.03 in phase
   space) and then observed on the real data to four decimals: the DAM
   value-space reading sits exactly on its null (0.0121 vs 0.0121).
2. **The embedding that sees loops** is the standard position/velocity
   trick: states = (price bin, Δ-sign). A loop stops retracing itself there.
3. **The null must share the spectrum.** A plain shuffle is wrong twice —
   it destroys persistence, and the Δ-sign embedding of an i.i.d. series is
   structurally asymmetric (0.15 nats of pure artifact). The correct null is
   the phase-randomised (FT) surrogate: identical power spectrum (persistence
   *and* the diurnal peak), time-reversible by construction.

   Even that is not enough for heavy tails: LMPs carry excess kurtosis ≈130,
   plain FT surrogates Gaussianize it, and the red-team rejected them. The
   working null is the **amplitude-adjusted** (AAFT) surrogate — spectrum
   AND marginal preserved — with detection requiring exceedance of *both*
   null classes, and an explicit *below-band flag* when the reading escapes
   the null on the low side.

**The readings** (scoped to this n, 6 bins, and this embedding):
SP15 **DAM hourly is at-null** — inside the AAFT band, consistent with a
linear time-reversible process. **RTM 5-minute is a no-detection with the
below-band flag raised**: the reading sits 5–6× *below* the null median,
meaning no linear process with the data's spectrum and marginal can
reproduce its Δ-sign persistence — neither a detection nor a certified
null, and the persistence anomaly itself is now a chase item (bracketing
surrogates; ramp-constraint mechanism). Finding F-0008, as revised after
adversarial review — the retraction of the original "certified null"
wording is part of the record.

The instruments do not hallucinate irreversibility on real market data,
and they do not manufacture certainty where the null class fails.

Data source: CAISO OASIS (public API, no key; used per its posted terms —
informational reports, paced requests with backoff, local caching).

Artifacts: `electricity_irreversibility_dam.json`,
`electricity_irreversibility_rtm.json` (regenerate: `make reproduce`; the
CAISO cache makes reruns cheap and identical).

## The reciprocity meter's first empirical read (unit domains.pricing, F-0011)

From the Dominick's canned-soup panel (HF `qbz506/dreamprice-dominicks-cso`; single-chain scanner data): own-cost pass-through 1.07 (Campbell) and 0.97 (Progresso); cross terms 0.003 and 0.0005 with the asymmetry CI covering zero; **ℛ = 0.0011 [0.00005, 0.005]** by cluster bootstrap over 86 stores, re-demeaned within every resample (the within transform is part of the estimator, so it resamples too). The ex-ante prediction — stated in config before the run — was exactly this: one retailer pricing both brands toward one category objective must respond symmetrically, so the meter should read ≈ 0. It does. The multi-agent reciprocity question needs cross-chain data; this dataset cannot ask it, and the finding says so. Companion scan: 0/30 stores show Edgeworth irreversibility in weekly category indices vs the reversibilized-Markov null (0.3 false positives expected) — at-null, conservative under the sample's week gaps.
