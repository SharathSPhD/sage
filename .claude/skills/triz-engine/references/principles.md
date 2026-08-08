# The 40 inventive principles, adapted to software / mathematics / this project

Classical name first; then the adaptation that matters here. Principles with no natural software reading are marked *(rarely useful here)* rather than force-fitted.

1. **Segmentation** — decompose into independent components: separable transforms (Kronecker Hodge), per-player blocks, plugin boundaries, worktrees per track.
2. **Taking out** — extract the disturbing part: pull the nonstrategic component out of the game (it cannot affect QRE); isolate impure I/O from the functional core.
3. **Local quality** — different parts under different regimes: per-domain `ConjugateFieldSpec`; per-engine response operators; strict mypy on the library, looser on scripts.
4. **Asymmetry** — exploit asymmetry instead of fighting it: the reciprocity defect *is* measured asymmetry; asymmetric pass-through is signal, not noise.
5. **Merging** — combine identical operations: the resolvent serves susceptibility, implicit diff, and bifurcation detection — implemented once.
6. **Universality** — one object serving multiple functions: `PayoffOracle` wraps any payoff model; `docs/theory` renders as docs and app.
7. **Nested doll** — structures within structures: games within families (α-indexed), configs composed within configs (Hydra groups).
8. **Anti-weight** — counterbalance: regularisation terms offsetting ill-conditioning; entropy term balancing payoff maximisation (that *is* QRE).
9. **Preliminary anti-action** — pre-compensate a known harm: tangent-space projection before any spectral read; payoff normalisation before λ estimation.
10. **Preliminary action** — do it before it's needed: JIT warm-up on service start; golden outputs regenerated before gates run; pre-commit hooks before CI.
11. **Beforehand cushioning** — prepare fallbacks for low-reliability paths: pygambit oracle alongside our solvers; `pure_callback` validation path behind a flag for the DreamPrice port.
12. **Equipotentiality** — remove the need to move between levels: strategically equivalent games mapped to one normalised representative, so all analyses happen at one "level".
13. **The other way round** — invert the action: instead of asking "does the theory predict the number", ask "what does the instrument read and why"; CCP estimation inverts observed probabilities to payoffs instead of solving forward.
14. **Spheroidality / curvature** — move from lines to curves: arclength continuation instead of naive λ-stepping; mirror descent's geometry instead of Euclidean steps.
15. **Dynamics** — make it adaptive: solver switching on `distance_to_criticality`; damping schedules; λ as a covariate-indexed process rather than a constant.
16. **Partial or excessive action** — do slightly less or more when exact is hard: TUR *lower bound* instead of an unattainable exact EPR from partial observation; over-generate synthetic families and filter to target α.
17. **Another dimension** — add a dimension: lift the payoff tensor into the product eigenbasis (where projections are diagonal); phase maps in (λ, α) rather than λ alone.
18. **Mechanical vibration** — probe with oscillation: perturbation probes (poke firm i, read firm j) as the operational reciprocity measurement; quench/anneal protocols.
19. **Periodic action** — act periodically rather than continuously: scheduled benchmark regeneration; merge cadence; dashboard refresh per merge.
20. **Continuity of useful action** — keep the useful thing running: background workers for long jobs; the Lab as an always-on instrument rather than batch scripts.
21. **Skipping** — rush through the harmful stage: fast approximate solve to get near the fixed point, then polish with Newton steps; coarse grids to bracket criticality before fine sweeps.
22. **Blessing in disguise** — turn harm into benefit: anomalies are the product; solver divergence near criticality is a *detector* of criticality.
23. **Feedback** — introduce feedback: gates feeding dashboards feeding priorities; validation reports that refuse bad data loudly; red-team objections cycling back into gate specs.
24. **Intermediary** — use a carrier: the oracle protocol as intermediary between demand models and the engine; the API as the app's only path to compute.
25. **Self-service** — the system serves itself: games generate their own test families (project + re-mix at target α); resolved configs written beside results make experiments self-documenting.
26. **Copying** — use a cheap copy: synthetic trajectories with known EPR as stand-ins for real data; Sioux Falls as the debug-scale copy of a real network.
27. **Cheap short-lived objects** — disposable instead of durable: scratch branches for hook negative-tests; ephemeral worktrees; throwaway small-game cross-checks.
28. **Mechanics substitution** — replace mechanical with field-based: replace generic least-squares projection with the spectral (eigenbasis) transform; replace dense B with matrix-free contractions.
29. **Pneumatics and hydraulics** — *(rarely useful here)* closest reading: streaming/SSE instead of blocking responses.
30. **Flexible shells and thin films** — thin boundary layers: thin client packages; thin domain plugins; API schemas as the membrane between library types and the wire.
31. **Porous materials** — leave deliberate holes: extras (`[gambit]`, `[blp]`) as optional pores rather than hard deps; protocol methods typed as `object` where a hard dependency would leak.
32. **Colour changes** — change visibility: confidence tiers rendered visibly in docs; warnings (`near_criticality`) coloured in the app; dashboard health indicators.
33. **Homogeneity** — interacting objects from the same material: consumer-side and firm-side logit are the same mathematics — reuse the machinery; all engines share one solver stack.
34. **Discarding and recovering** — shed what's spent: prune memory files; supersede claims rather than deleting them; revert broken merges immediately.
35. **Parameter changes** — change the state of a parameter: fold payoff scale into λ; rescale to unit range; work in log-space (log_softmax) instead of probability space.
36. **Phase transitions** — use the transition itself: the bifurcation structure of (I − SB) is the phase locator; criticality is the object of study, not an obstacle.
37. **Thermal expansion** — *(rarely useful here)* closest reading: λ-annealing to traverse the branch.
38. **Strong oxidants** — intensify the environment: adversarial review as the intensified environment; hostile-referee reports at stage boundaries; property-based tests over example tests.
39. **Inert atmosphere** — neutral environment: fixed seeds and frozen configs for reproduction; hermetic CI; sandboxed hooks.
40. **Composite materials** — combine materials with different properties: hybrid estimators (QLk beats pure QRE — use hybrids as baselines); dense-near-criticality + iterative-far composite solver; JAX core + PyTorch-validated port.
