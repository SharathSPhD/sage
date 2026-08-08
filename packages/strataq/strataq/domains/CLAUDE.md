# domains/ — the plugin contract

A domain plugin ships **exactly five things** (see `strataq/core/protocols.py`) and exports one `PLUGIN: DomainPlugin`:

```python
oracle: PayoffOracle  # payoffs; profit/quantity/response_matrix
grid: ActionGridBuilder  # continuous decision -> discrete action grid
field: ConjugateFieldSpec  # WHAT h IS in this domain + where it is in data
loader: DatasetLoader | None  # public data, or None for synthetic-only
learn: LearnPageSpec  # the explainer
ENGINE: Literal["finite", "population", "bayesian"]
```

- **`ConjugateFieldSpec` is load-bearing.** A domain that cannot name its observable payoff perturbation declares `ConjugateFieldSpec.NONE`, and the library/API then refuse `chi_equilibrium` / `reciprocity_defect` for it. (Example: sports has no field — Learn-mode only.)
- **Touch core ⟹ not a plugin.** Any change under `domains/` that edits `core/`, `finite/` or `population/` fails the hook and CI. If the domain needs new machinery, it's an engine: stop, write an ADR.
- Domains do not import from other domains.
- Known fields: congestion → link tolls (exact-linear); blotto → battlefield budgets (experimenter-set); pricing → wholesale cost shocks; electricity → fuel cost shocks (approximate); security → synthetic; sports → NONE.
- Dataset licences ride with loaders: Dominick's-derived artefacts are CC-BY-NC-4.0 everywhere.
