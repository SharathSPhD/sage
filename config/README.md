# config/ — all configuration lives here

Hydra + OmegaConf composition for experiments and library defaults; pydantic-settings for services. Every config is validated into a typed object (`strataq.core.config`) — raw dicts never cross an API boundary.

- `base.yaml` — global invariants: float64, seed policy, the tolerance ladder, λ-normalisation convention, criticality warning threshold.
- `engines/` — per-engine settings (finite, population; bayesian when it exists by decision).
- `domains/` — per-plugin settings (grids, data locations, field columns).
- `solvers/` — solver strategy configs (`damped`, `anderson`, `mirror`, `homotopy`), config-selectable by string via the registry.
- `experiments/` — experiment compositions; every run is `python -m experiments.<name> --config-name=<cfg>` and writes its resolved config next to its results.
- `services/` — env-driven service settings (pydantic-settings); secrets come from the environment only, never from files in this repo.
