import fs from "node:fs";
import path from "node:path";
import { FindingsGallery, type Charts } from "./FindingsGallery";
import { ResearchCrumb } from "../components/ResearchCrumb";

export const metadata = { title: "Findings — SAGE" };

function loadCharts(): Charts {
  const candidates = [
    path.join(process.cwd(), "data", "findings_charts.json"),
    path.join(process.cwd(), "..", "..", "apps", "web", "data", "findings_charts.json"),
  ];
  for (const p of candidates) {
    if (fs.existsSync(p)) return JSON.parse(fs.readFileSync(p, "utf-8")) as Charts;
  }
  throw new Error("findings_charts.json not found");
}

export default function FindingsPage() {
  return (
    <div className="wrap" style={{ paddingTop: "2.2rem" }}>
      <ResearchCrumb />
      <h1 style={{ marginBottom: "0.3rem" }}>The anomaly log</h1>
      <p style={{ color: "var(--text-dim)", maxWidth: "46rem", marginTop: 0 }}>
        How a claim here earns its confidence. Every unexpected result — discoveries, certified
        nulls, refuted repairs, model rejections, and the retractions in between — with the seed that
        regenerates it. Nothing here is smoothed over: the failed hypotheses are listed with the
        same typography as the wins, because a record you can only read one way is not a record.
      </p>
      <FindingsGallery charts={loadCharts()} />
    </div>
  );
}
