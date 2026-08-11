"use client";

import { useState } from "react";
import { PanelShell } from "../components/panels/ui";

/* The product surface (unit product.toolkit): paste-your-data instruments,
   served by the live float64 backend's /v1/toolkit endpoints. */

type SeriesVerdict = {
  detected: boolean;
  p_value: number;
  statistic: number;
  null_median: number;
  null_mismatch_low: boolean;
  warnings: string[];
};

type ChiRead = {
  r: number;
  verdict: string;
  ci_low: number | null;
  ci_high: number | null;
  calibration: Record<string, number>;
  warnings: string[];
};

function Warnings({ items }: { items: string[] }) {
  if (!items.length) return null;
  return (
    <ul style={{ margin: "0.5rem 0 0", paddingLeft: "1.1rem", fontSize: "0.78rem", color: "var(--amber)" }}>
      {items.map((w) => (
        <li key={w}>{w}</li>
      ))}
    </ul>
  );
}

function SeriesTool() {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [out, setOut] = useState<SeriesVerdict | null>(null);

  const run = () => {
    const series = text
      .split(/[\s,;]+/)
      .filter(Boolean)
      .map(Number);
    if (series.some((x) => !Number.isFinite(x))) {
      setError("could not parse: every entry must be a number");
      return;
    }
    setBusy(true);
    setError(null);
    fetch("/api/v1/toolkit/irreversibility", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ series, n_surrogates: 150 }),
    })
      .then(async (r) => {
        if (!r.ok) throw new Error((await r.json()).detail ?? `HTTP ${r.status}`);
        return r.json();
      })
      .then(setOut)
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setBusy(false));
  };

  return (
    <PanelShell title="is my time series irreversibly driven?" provenance="live">
      <p style={{ fontSize: "0.85rem", color: "var(--text-dim)", marginTop: 0 }}>
        Paste a scalar series (numbers separated by spaces, commas or newlines — prices, flows,
        anything ordered in time; ≥ 300 points for real power). It is phase-embedded and tested
        against a reversible null with matched persistence — the instrument that found the
        day-ahead market&apos;s diurnal loop (F-0009).
      </p>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={4}
        placeholder="41.2, 38.9, 36.4, 35.1, 37.8, 44.6, 52.3, …"
        style={{ width: "100%", background: "var(--bg-raised)", color: "var(--text)", border: "1px solid var(--border)", borderRadius: "6px", padding: "0.6rem", fontFamily: "var(--mono)", fontSize: "0.8rem" }}
      />
      <div style={{ marginTop: "0.6rem" }}>
        <button data-primary="true" onClick={run} disabled={busy}>
          {busy ? "testing against 150 reversible surrogates…" : "run the test"}
        </button>
      </div>
      {error && <p style={{ color: "var(--red)", fontSize: "0.8rem", marginTop: "0.5rem" }}>{error}</p>}
      {out && (
        <div style={{ marginTop: "0.8rem" }}>
          <div className="reading" data-tone={out.detected ? "warn" : "ok"}>
            {out.detected ? "IRREVERSIBLY DRIVEN" : "at-null (reversible)"}
            <span className="unit">p = {out.p_value.toFixed(3)}</span>
          </div>
          <p style={{ fontSize: "0.78rem", color: "var(--text-faint)", margin: "0.4rem 0 0" }}>
            KLD statistic {out.statistic.toExponential(2)} vs null median{" "}
            {out.null_median.toExponential(2)}
            {out.null_mismatch_low && " · below-null flag raised (model mismatch)"}
          </p>
          <Warnings items={out.warnings} />
        </div>
      )}
    </PanelShell>
  );
}

function ChiTool() {
  const [cells, setCells] = useState(["1.07", "0.003", "0.0005", "0.97"]);
  const [ses, setSes] = useState(["", "", "", ""]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [out, setOut] = useState<ChiRead | null>(null);

  const run = () => {
    const v = cells.map(Number);
    if (v.some((x) => !Number.isFinite(x))) {
      setError("all four χ entries must be numbers");
      return;
    }
    const seVals = ses.map((s) => (s.trim() === "" ? null : Number(s)));
    const anySe = seVals.some((s) => s !== null);
    if (anySe && seVals.some((s) => s === null || !Number.isFinite(s) || s < 0)) {
      setError("standard errors: fill all four (nonnegative) or none");
      return;
    }
    setBusy(true);
    setError(null);
    fetch("/api/v1/toolkit/reciprocity", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chi: [
          [v[0], v[1]],
          [v[2], v[3]],
        ],
        chi_se: anySe
          ? [
              [seVals[0], seVals[1]],
              [seVals[2], seVals[3]],
            ]
          : null,
      }),
    })
      .then(async (r) => {
        if (!r.ok) throw new Error((await r.json()).detail ?? `HTTP ${r.status}`);
        return r.json();
      })
      .then(setOut)
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setBusy(false));
  };

  const cellStyle = {
    width: "6rem",
    background: "var(--bg-raised)",
    color: "var(--text)",
    border: "1px solid var(--border)",
    borderRadius: "6px",
    padding: "0.4rem",
    fontFamily: "var(--mono)",
    fontSize: "0.8rem",
  } as const;

  return (
    <PanelShell title="is my system reciprocal? (χ matrix)" provenance="live">
      <p style={{ fontSize: "0.85rem", color: "var(--text-dim)", marginTop: 0 }}>
        Enter your measured 2×2 cross-response matrix — χ[i][j] = how agent i&apos;s action moves
        when agent j&apos;s incentives shift (e.g. cost pass-through between two firms). The
        defaults are the actual Dominick&apos;s pass-through estimates (F-0011). Add standard
        errors for an uncertainty-aware verdict.
      </p>
      <div style={{ display: "grid", gridTemplateColumns: "auto auto", gap: "0.4rem", width: "fit-content" }}>
        {cells.map((c, i) => (
          <input key={i} value={c} onChange={(e) => setCells(cells.map((x, j) => (j === i ? e.target.value : x)))} style={cellStyle} aria-label={`chi ${i}`} />
        ))}
      </div>
      <p style={{ fontSize: "0.75rem", color: "var(--text-faint)", margin: "0.6rem 0 0.3rem" }}>
        standard errors (optional):
      </p>
      <div style={{ display: "grid", gridTemplateColumns: "auto auto", gap: "0.4rem", width: "fit-content" }}>
        {ses.map((c, i) => (
          <input key={i} value={c} onChange={(e) => setSes(ses.map((x, j) => (j === i ? e.target.value : x)))} style={cellStyle} placeholder="—" aria-label={`se ${i}`} />
        ))}
      </div>
      <div style={{ marginTop: "0.6rem" }}>
        <button data-primary="true" onClick={run} disabled={busy}>
          {busy ? "reading…" : "read ℛ"}
        </button>
      </div>
      {error && <p style={{ color: "var(--red)", fontSize: "0.8rem", marginTop: "0.5rem" }}>{error}</p>}
      {out && (
        <div style={{ marginTop: "0.8rem" }}>
          <div className="reading">
            ℛ = {out.r < 0.01 ? out.r.toExponential(2) : out.r.toFixed(3)}
            {out.ci_low !== null && out.ci_high !== null && (
              <span className="unit">
                95% CI [{out.ci_low.toFixed(4)}, {out.ci_high.toFixed(4)}]
              </span>
            )}
          </div>
          <p style={{ fontSize: "0.82rem", color: "var(--text-dim)", margin: "0.4rem 0 0" }}>{out.verdict}</p>
          <p style={{ fontSize: "0.75rem", color: "var(--text-faint)", margin: "0.3rem 0 0" }}>
            calibration: {Object.entries(out.calibration).map(([k, v]) => `${k} ${v}`).join(" · ")}
          </p>
          <Warnings items={out.warnings} />
        </div>
      )}
    </PanelShell>
  );
}

export function ToolsPanel() {
  return (
    <div className="panel-cols" style={{ marginTop: "1.6rem" }}>
      <SeriesTool />
      <ChiTool />
    </div>
  );
}
