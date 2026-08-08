# experiments/ — conventions

- Every experiment is `python -m experiments.<name> --config-name=<cfg>`, Hydra-driven, seeded, and writes its **resolved config into the output directory alongside results**. Reproducible from that config alone.
- Science runs go **through the library** — an experiment is composition + config, never re-implementation. If an experiment needs logic the library lacks, that logic moves into the library with tests first.
- Fixed seeds from `config/base.yaml` seed policy; `make reproduce` regenerates every gate artifact from here.
- The α×λ phase sweep is a Hydra multirun; report the surface, FDR-adjust any per-cell claims (master spec §16).
- EPR estimator work validates on synthetic trajectories with known ground truth before real data, always.
- Anomalies observed while running: straight into `memory/findings.md` with the config that produced them.
