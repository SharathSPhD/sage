---
name: docs-sync
description: Keep docs, dashboard, and ledgers synchronised with reality. Invoke as /docs-sync after closing a unit or before a merge to main.
---

# Docs sync

1. Public API of the touched modules ↔ docs: every public function documented, NumPy style, References + confidence tier present. mkdocs build passes locally (`uv run mkdocs build --strict`).
2. `docs/theory/` and the app Learn content are the same source — check nothing forked; the app renders from docs/theory, never a copy.
3. Regenerate the progress dashboard (`uv run python docs/progress/generate.py`) from gates/status.json + benchmarks/results + memory/claims.md. Both views (technical, non-technical) render; no unexplained Greek letters in the non-technical view (glossary hover-defs present).
4. CHANGELOG has the unit's entry; memory/claims.md tiers match what the docs assert; memory/decisions.md has ADRs for any divergence introduced.
5. Dashboard stale = gate not green. This skill is how it stays fresh.
