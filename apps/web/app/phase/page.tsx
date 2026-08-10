import fs from "node:fs";
import path from "node:path";
import { PhaseExplorer, type Surface } from "./PhaseExplorer";

export const metadata = { title: "Phase map — SAGE Labs" };

// The surface is a committed, gate-checked artifact regenerated from fixed
// seeds (unit science.phase_map). The app reads it at build time — the
// dashboard reads files, it does not compute.
function loadSurface(): Surface {
  // Monorepo checkout first; the standalone deploy carries a synced copy in
  // data/ (refresh with `make reproduce` + the deploy step).
  const candidates = [
    path.join(process.cwd(), "..", "..", "benchmarks", "results", "phase_map_surface.json"),
    path.join(process.cwd(), "data", "phase_map_surface.json"),
  ];
  for (const p of candidates) {
    if (fs.existsSync(p)) return JSON.parse(fs.readFileSync(p, "utf-8")) as Surface;
  }
  throw new Error("phase_map_surface.json not found in benchmarks/results or data/");
}

export default function PhasePage() {
  const surface = loadSurface();
  return (
    <div className="wrap" style={{ paddingTop: "2.2rem" }}>
      <h1 style={{ marginBottom: "0.3rem" }}>The α × λ phase map</h1>
      <p style={{ color: "var(--text-dim)", maxWidth: "46rem", marginTop: 0 }}>
        {surface.alphas.length * surface.lambdas.length} cells, each the median over 100 randomly
        generated games solved at that (α, λ) — {""}
        {(surface.alphas.length * surface.lambdas.length * 100).toLocaleString()} solves total,
        regenerated from fixed seeds by <code>make reproduce</code>. Hover any cell for its
        readings; switch metrics to see how the same plane looks through each instrument.
      </p>
      <PhaseExplorer surface={surface} />
    </div>
  );
}
