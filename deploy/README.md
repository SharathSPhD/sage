# deploy/ — the Oracle VM stack (SAGE_HOSTING.md)

Decision: Oracle Always Free A1.Flex (2 OCPU / 12 GB ARM) replaces Render (ADR-0010).
Compute split: research sweeps stay on the DGX Spark; the VM serves the API only.

- Image: built linux/arm64 by CI (`.github/workflows/api-image.yml`) → GHCR.
- Server prep + stack: docs/ops-hosting.md §4–5 (Docker, swap, iptables 80/443, this compose file at /opt/sage).
- `.env` on the VM only (never committed): JAX_ENABLE_X64=1, JAX_PLATFORMS=cpu, SAGE_API_* limits.
- Deploy: `docker compose pull && docker compose up -d` after each green-gate image push.

## One-time before first pull

The GHCR package is pushed private by default (GITHUB_TOKEN scope). Either:
- GitHub → Packages → sage-api → Package settings → Change visibility → **Public** (recommended, open project), or
- `docker login ghcr.io` on the VM with a classic PAT holding `read:packages`.

## Interim deployment (live) — ADR-0011

`http://150.136.84.2` — VM.Standard.E2.1.Micro (Always Free x86), bare-metal:
repo at `/home/ubuntu/sage`, `uv sync --package sage-api`, systemd unit `sage-api`
(uvicorn on 127.0.0.1:8000, MemoryMax=700M), Caddy on :80. Update:
`ssh ubuntu@150.136.84.2 'cd sage && git pull && systemctl restart sage-api'` (via sudo).
Migrate to the A1 box with the compose stack above when capacity lands.
