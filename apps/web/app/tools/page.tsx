import type { Metadata } from "next";
import Link from "next/link";
import { ToolsPanel } from "./ToolsPanel";

export const metadata: Metadata = { title: "Your data — SAGE Labs" };

export default function ToolsPage() {
  return (
    <div className="wrap" style={{ paddingTop: "2.2rem" }}>
      <p className="superseded">
        <strong>There is a better way in now.</strong>
        <span>
          <Link href="/diagnose">Diagnose</Link> runs both of these instruments at once and tells
          you what the answer changes. These two panels still work, and the numbers are the same.
        </span>
      </p>
      <h1 style={{ marginBottom: "0.3rem" }}>Point the instruments at your data</h1>
      <p style={{ color: "var(--text-dim)", maxWidth: "46rem", marginTop: 0 }}>
        The same gated machinery behind every finding here, on whatever you paste in. Verdicts
        arrive with their honesty warnings attached — flat likelihoods warn instead of quoting,
        borderline reads say so, and underpowered tests tell you they are. Python users:{" "}
        <code>pip install strataq</code> gives the identical calls offline
        (<code>strataq.toolkit</code>).
      </p>
      <ToolsPanel />
    </div>
  );
}
