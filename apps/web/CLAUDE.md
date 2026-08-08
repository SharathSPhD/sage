# apps/web — SAGE Labs conventions

- Next.js (App Router) + TypeScript on Vercel; Tailwind + shadcn/ui; visx for charts, Plotly for 3-D phase surfaces/heatmaps; KaTeX; TanStack Query against the API.
- Three modes, shipped in order: **Learn** (ten explainers; static/ISR), **Lab** (the research instrument — domain selector above λ and α master sliders, all panels live), **Analyze** (upload → schema mapping → demand → strategic → decision → report).
- Learn content is **not authored here**: `docs/theory/01–10` is the single source; write once, render twice. Explainer 10 ("the same machinery everywhere") ships last.
- Heavy compute always proxies to the Render API; Vercel functions handle only auth and short requests. Signed direct-to-storage uploads — Vercel never holds the file.
- The Lab's export panel includes the "reproduce this in Python" button emitting a runnable `strataq` snippet — the main funnel from app to library.
- Honesty is a feature: the one-price objection gets its own explainer (#9); Analyze reports state once, clearly, that λ absorbs unmodelled heterogeneity and that recommendations are conditional on the demand model. No panel implies more than the analysis supports.
