import Link from "next/link";

export default function Home() {
  return (
    <div>
      <h1>Instruments for strategic systems</h1>
      <p style={{ fontSize: "1.05rem", lineHeight: 1.6 }}>
        SAGE Labs is a research instrument, not a demo: a susceptibility meter, a reciprocity
        meter, an entropy-production meter and a phase locator for quantal-response equilibria —
        each calibrated on games where the correct reading is known (a real road network reads
        exactly zero; rock–paper–scissors reads loudly), then pointed at systems where nobody
        knows the answer.
      </p>
      <ul style={{ lineHeight: 2 }}>
        <li>
          <Link href="/learn">Learn</Link> — the ideas, honestly, including the strongest
          objection to the whole approach.
        </li>
        <li>
          <Link href="/lab">Lab</Link> — pick a game, slide λ, watch every meter respond live.
        </li>
        <li>
          <a href="https://sharathsphd.github.io/sage/progress/">Progress dashboard</a> — every
          claim&apos;s confidence tier, every gate&apos;s state, every anomaly chased.
        </li>
      </ul>
      <p style={{ fontSize: "0.9rem", color: "#555" }}>
        Every number this app shows regenerates from fixed seeds in the open repository; the
        adversarial reviews that closed each instrument are part of the public record.
      </p>
    </div>
  );
}
