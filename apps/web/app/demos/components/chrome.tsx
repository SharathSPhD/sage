"use client";

import Link from "next/link";
import { createContext, useContext, useEffect, useId, useState } from "react";
import type { Landmark } from "../../../lib/demos/landmarks";
import { PauseButton, PauseProvider } from "./motion";

/* ---------- depth: one page for two audiences ---------- */

const DepthCtx = createContext<{ full: boolean; setFull: (v: boolean) => void }>({ full: false, setFull: () => {} });

export function DemoChrome({ children }: { children: React.ReactNode }) {
  const [full, setFull] = useState(false);
  return (
    <DepthCtx.Provider value={{ full, setFull }}>
      <PauseProvider>{children}</PauseProvider>
    </DepthCtx.Provider>
  );
}

/** The depth control. Not two audience tracks — one page, one dial. */
export function DepthBar() {
  const { full, setFull } = useContext(DepthCtx);
  return (
    <div className="demo-bar">
      <div className="demo-depth" role="group" aria-label="Reading depth">
        <button type="button" className="btn" data-primary={!full ? "true" : undefined} aria-pressed={!full} onClick={() => setFull(false)}>
          Read it
        </button>
        <button type="button" className="btn" data-primary={full ? "true" : undefined} aria-pressed={full} onClick={() => setFull(true)}>
          Show the maths
        </button>
      </div>
      <PauseButton />
    </div>
  );
}

/** The collapsed layer that carries the CI, the n and the artifact filename. */
export function ShowMaths({ title = "Show the maths", children }: { title?: string; children: React.ReactNode }) {
  const { full } = useContext(DepthCtx);
  const [open, setOpen] = useState(false);
  useEffect(() => setOpen(full), [full]);
  return (
    <details className="show-maths" open={open} onToggle={(e) => setOpen((e.currentTarget as HTMLDetailsElement).open)}>
      <summary>{title}</summary>
      <div className="show-maths-body">{children}</div>
    </details>
  );
}

/* ---------- text budget ---------- */

export function Widget({
  hook,
  lede,
  consequence,
  children,
  maths,
}: {
  hook: string;
  lede: React.ReactNode;
  consequence: React.ReactNode;
  children: React.ReactNode;
  maths?: React.ReactNode;
}) {
  const id = useId();
  return (
    <section className="card widget" aria-labelledby={id}>
      <h2 id={id} className="widget-hook">
        {hook}
      </h2>
      <div className="widget-lede">{lede}</div>
      <div className="widget-body">{children}</div>
      <p className="widget-consequence">{consequence}</p>
      {maths ? <ShowMaths>{maths}</ShowMaths> : null}
    </section>
  );
}

/** A number with its label, in the site's reading style. */
export function Readout({
  label,
  value,
  unit,
  tone,
  hand,
  live,
}: {
  label: string;
  value: string;
  unit?: string;
  tone?: "landscape" | "whirlpool" | "neutral";
  hand?: boolean;
  live?: boolean;
}) {
  const color = hand
    ? "var(--accent-strong)"
    : tone === "landscape"
      ? "var(--q-landscape-text)"
      : tone === "whirlpool"
        ? "var(--q-whirlpool-text)"
        : "var(--text)";
  return (
    <div className="demo-readout">
      <div className="panel-label">{label}</div>
      <div className="reading" style={{ color }} aria-live={live ? "polite" : undefined}>
        {value}
        {unit ? <span className="unit">{unit}</span> : null}
      </div>
    </div>
  );
}

/** Provenance for a landmark: what file, what key, what the artifact itself says. */
export function ProvenanceCard({ landmark }: { landmark: Landmark }) {
  return (
    <div className="provenance">
      <p className="provenance-head">
        <strong>{landmark.name}</strong> · {landmark.kind} · quadrant {landmark.quadrant}
      </p>
      <dl>
        <dt>Response asymmetry ℛ</dt>
        <dd>
          {landmark.RLabel}
          {landmark.RCi ? ` [${landmark.RCi[0].toExponential(2)}, ${landmark.RCi[1].toExponential(2)}] 95% CI` : ""}
        </dd>
        <dt>Dissipation σ_EP</dt>
        <dd>{landmark.EPLabel}</dd>
        {landmark.n ? (
          <>
            <dt>n</dt>
            <dd>{landmark.n}</dd>
          </>
        ) : null}
      </dl>
      {landmark.provenance.map((p) => (
        <div key={p.artifact} className="provenance-src">
          <code>{p.artifact}</code>
          <p className="provenance-field">{p.field}</p>
          <p className="provenance-note">{p.note}</p>
        </div>
      ))}
    </div>
  );
}

export function DemoHeader({
  eyebrow,
  title,
  standfirst,
}: {
  eyebrow: string;
  title: string;
  standfirst: React.ReactNode;
}) {
  return (
    <header className="demo-header">
      <p className="research-crumb">
        <Link href="/demos">← All demos</Link>
      </p>
      <p className="eyebrow">{eyebrow}</p>
      <h1>{title}</h1>
      <div className="lede">{standfirst}</div>
      <DepthBar />
    </header>
  );
}
