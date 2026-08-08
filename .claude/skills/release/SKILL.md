---
name: release
description: Cut a release or stage tag. Invoke as /release <version|stage-N>. PyPI publication only after Stage 1 gates are green and the API is stable.
---

# Release

1. Preconditions, verified not assumed: main green (CI + gates + docs + fast `make reproduce`); all tracks merged and pushed; dashboard current; CHANGELOG section for the release complete.
2. Stage completion: tag `v0.<stage>.0`, annotated with the stage's headline artefacts; push tag; verify Pages redeployed.
3. PyPI (`strataq` only, never `sage`): allowed only when Stage 1 gates are fully green and the public API is declared stable in an ADR. Build with `uv build`, check with twine, publish via trusted publishing from CI — never a local token. strataq-client releases follow the API's OpenAPI version.
4. HF artefacts (demo Space, ported DreamPrice weights, derived datasets): dataset cards carry licences; anything Dominick's-derived is CC-BY-NC-4.0, stated in the card, no exceptions.
5. Post-release: SESSION.md entry, memory/decisions.md ADR if any release decision deviated from plan.
