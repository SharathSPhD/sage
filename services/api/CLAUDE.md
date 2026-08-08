# services/api — conventions

- FastAPI + Pydantic v2 + Uvicorn, Docker on Render (`render.yaml` here). Redis-backed `arq` queue + worker for anything over ~5 s (branch continuation, EPR estimation, hierarchical λ). Postgres (Supabase) for jobs/metadata; migrations under `migrations/`, version-controlled.
- Schemas mirror library types; numeric payloads carry explicit `shape`/`dtype`; large arrays returned as signed URLs to Parquet/NPZ, never inline.
- Every response embeds `provenance` (library version, oracle, λ parameterisation, payoff normalisation constant) and `warnings` (`near_criticality`, `low_support`, `weak_instruments`, …).
- **The API refuses what a domain can't support**: if `ConjugateFieldSpec` is `NONE`, response-instrument endpoints return an explicit error, not a number.
- `/v1/optimize/price` always returns the competitor distribution alongside the point recommendation — the distribution over rivals' prices is the useful object.
- API keys (`X-API-Key`) + rate limits from day one; free tier bounded by game size. Config via pydantic-settings, env only; secrets never in the repo.
- Pin `jax[cpu]` in the image; JIT warm-up on startup with a tiny game. Publish OpenAPI; the TS client and `strataq-client` are generated from it.
