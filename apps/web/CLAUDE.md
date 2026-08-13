# apps/web — SAGE Labs conventions

- Next.js (App Router) + TypeScript on Vercel; Tailwind + shadcn/ui; visx for charts, Plotly for 3-D phase surfaces/heatmaps; KaTeX; TanStack Query against the API.
- Three modes, shipped in order: **Learn** (ten explainers; static/ISR), **Lab** (the research instrument — domain selector above λ and α master sliders, all panels live), **Analyze** (upload → schema mapping → demand → strategic → decision → report).
- Learn content is **not authored here**: `docs/theory/01–10` is the single source; write once, render twice. Explainer 10 ("the same machinery everywhere") ships last.
- Heavy compute always proxies to the Render API; Vercel functions handle only auth and short requests. Signed direct-to-storage uploads — Vercel never holds the file.
- The Lab's export panel includes the "reproduce this in Python" button emitting a runnable `strataq` snippet — the main funnel from app to library.
- `/demos` is the explorable surface, and its rules are its own: one idea per widget, one degree of freedom on first
  exposure, drag the object rather than a proxy slider, ghost trails instead of redraws, defaults inside the
  interesting regime, and one accent reserved for the mark moving under the reader's hand. Landscape and whirlpool
  keep the `strataq.viz` palette hues site-wide. Every displayed research number carries the `benchmarks/results/`
  filename and key it was read from, in a collapsed "show the maths" layer that also carries the CI and the n; a
  number that cannot be traced that way does not go on the page.
- Honesty is a feature: the one-price objection gets its own explainer (#9); Analyze reports state once, clearly, that λ absorbs unmodelled heterogeneity and that recommendations are conditional on the demand model. No panel implies more than the analysis supports.
