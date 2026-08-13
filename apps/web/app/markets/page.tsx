import fs from "node:fs";
import path from "node:path";
import { MarketReading, type Series } from "./MarketReading";
import { ResearchCrumb } from "../components/ResearchCrumb";

export const metadata = { title: "Market reading — SAGE" };

// Committed, gate-checked artifact (unit domains.electricity) — the app
// draws the data behind F-0008/F-0009; it does not refetch or recompute.
function loadSeries(): Series {
  const candidates = [
    path.join(process.cwd(), "..", "..", "benchmarks", "results", "electricity_series.json"),
    path.join(process.cwd(), "data", "electricity_series.json"),
  ];
  for (const p of candidates) {
    if (fs.existsSync(p)) return JSON.parse(fs.readFileSync(p, "utf-8")) as Series;
  }
  throw new Error("electricity_series.json not found");
}

export default function MarketsPage() {
  const series = loadSeries();
  return (
    <div className="wrap" style={{ paddingTop: "2.2rem" }}>
      <ResearchCrumb />
      <h1 style={{ marginBottom: "0.3rem" }}>First real-data reading: a power market</h1>
      <p style={{ color: "var(--text-dim)", maxWidth: "48rem", marginTop: 0 }}>
        {series.hours.length} hours of real day-ahead prices (CAISO SP15 hub, July 2026) put
        through the dissipation instruments. The finding survived four null classes, two
        adversarial reviews and one retraction — all on the public audit trail. Verdict:{" "}
        <strong>the day-ahead market is a measurably driven cycle</strong>, its irreversibility
        concentrated exactly where summer scarcity ramps drive it hardest.
      </p>
      <MarketReading series={series} />
    </div>
  );
}
