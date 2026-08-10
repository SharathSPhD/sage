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
