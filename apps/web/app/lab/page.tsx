"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Gauge } from "../components/Gauge";
import { GuessLambda } from "../components/panels/GuessLambda";

/* The Lab is a research instrument: pick a game, slide λ, every meter updates
   from the live API (heavy compute never runs in the browser). All requests go
   through the same-origin /api proxy. */

type Payoffs = number[][][];

interface Example {
  payoffs: Payoffs;
}

interface Readings {
  sigma?: number[][];
  reciprocity?: number;
  distance?: number;
  rhoSb?: number;
  epr?: number;
  detailedBalance?: boolean;
  warnings: string[];
}

interface GameFacts {
  alpha?: number;
  branch?: { lambdas: number[]; rhos: number[] };
}

const LAM_MIN = 0.1;
const LAM_MAX = 8;
const fromSlider = (t: number) => LAM_MIN * Math.pow(LAM_MAX / LAM_MIN, t);
const toSlider = (lam: number) => Math.log(lam / LAM_MIN) / Math.log(LAM_MAX / LAM_MIN);

async function post(path: string, body: unknown, signal?: AbortSignal) {
  const r = await fetch(`/api/v1/${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!r.ok) {
    const detail = await r.json().catch(() => ({}));
    throw new Error(detail?.detail ?? `${path}: HTTP ${r.status}`);
  }
  return r.json();
}

function BranchTrace({
  branch,
  lam,
}: {
  branch: { lambdas: number[]; rhos: number[] } | undefined;
  lam: number;
}) {
  if (!branch || branch.lambdas.length === 0) {
    return <div style={{ color: "var(--text-faint)", fontSize: "0.82rem" }}>tracing branch…</div>;
  }
  const W = 560;
  const H = 120;
  const maxRho = Math.max(1.15, ...branch.rhos);
  const x = (l: number) => (toSlider(Math.min(Math.max(l, LAM_MIN), LAM_MAX)) * (W - 20)) + 10;
  const y = (r: number) => H - 14 - (r / maxRho) * (H - 28);
  const pts = branch.lambdas.map((l, i) => `${x(l).toFixed(1)},${y(branch.rhos[i]).toFixed(1)}`);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%" }} aria-label="spectral radius along the λ branch">
      {/* criticality line at rho = 1 */}
      <line x1="10" x2={W - 10} y1={y(1)} y2={y(1)} stroke="var(--amber)" strokeDasharray="4 4" strokeWidth="1" />
      <text x={W - 12} y={y(1) - 4} fill="var(--amber)" fontSize="9" textAnchor="end" fontFamily="var(--mono)">
        ρ = 1 · criticality
      </text>
      <polyline points={pts.join(" ")} fill="none" stroke="var(--accent)" strokeWidth="1.8" />
      {/* current λ marker */}
      <line x1={x(lam)} x2={x(lam)} y1={12} y2={H - 12} stroke="var(--text-dim)" strokeWidth="1" strokeDasharray="2 3" />
      <text x={x(lam) + 4} y={20} fill="var(--text-dim)" fontSize="9" fontFamily="var(--mono)">
        λ = {lam.toFixed(2)}
      </text>
      <text x="10" y={H - 2} fill="var(--text-faint)" fontSize="9" fontFamily="var(--mono)">
        λ → (log scale)
      </text>
      <text x="10" y="12" fill="var(--text-faint)" fontSize="9" fontFamily="var(--mono)">
        ρ(SB): strategic feedback gain
      </text>
    </svg>
  );
}

function SigmaBars({ sigma }: { sigma: number[][] | undefined }) {
  if (!sigma) return <div style={{ color: "var(--text-faint)" }}>—</div>;
  return (
    <div className="sigma-bars">
      {sigma.map((probs, p) => (
        <div className="sigma-row" key={p}>
          <span className="lbl">P{p + 1}</span>
          <div className="sigma-track">
            {probs.map((v, a) => (
              <div className="sigma-cell" key={a} title={`action ${a + 1}: ${v.toFixed(4)}`}>
                <div className="sigma-fill" style={{ height: `${Math.max(3, v * 100)}%` }} />
                <span className="sigma-val">{v.toFixed(2)}</span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function PayoffMatrix({ payoffs }: { payoffs: Payoffs }) {
  if (payoffs.length !== 2) {
    return (
      <div style={{ color: "var(--text-faint)", fontSize: "0.82rem" }}>
        {payoffs.length}-player tensor ({payoffs.map((p) => JSON.stringify(p).length).length} blocks) — matrix view is 2-player only
      </div>
    );
  }
  const [u1, u2] = payoffs;
  const rows = u1.length;
  const cols = u1[0]?.length ?? 0;
  return (
    <table className="payoff-table">
      <thead>
        <tr>
          <th />
          {Array.from({ length: cols }, (_, j) => (
            <th key={j}>b{j + 1}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {Array.from({ length: rows }, (_, i) => (
          <tr key={i}>
            <th>a{i + 1}</th>
            {Array.from({ length: cols }, (_, j) => (
              <td key={j}>
                {u1[i][j]}, {u2[i][j]}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function Lab() {
  const [examples, setExamples] = useState<Record<string, Example>>({});
  const [selected, setSelected] = useState("");
  const [lam, setLam] = useState(1.2);
  const [readings, setReadings] = useState<Readings | null>(null);
  const [facts, setFacts] = useState<GameFacts>({});
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    fetch("/api/v1/examples")
      .then((r) => r.json())
      .then((d: Record<string, Example>) => {
        setExamples(d);
        const first = Object.keys(d)[0];
        if (first) setSelected(first);
      })
      .catch(() => setError("could not reach the backend — the Lab needs the live API"));
  }, []);

  const payoffs = useMemo(() => examples[selected]?.payoffs, [examples, selected]);

  // Per-game facts (α, branch trace) — once per game selection.
  useEffect(() => {
    if (!payoffs) return;
    setFacts({});
    const ctl = new AbortController();
    post("decompose", { payoffs, lam: 1.0 }, ctl.signal)
      .then((d) => setFacts((f) => ({ ...f, alpha: d.alpha })))
      .catch(() => undefined);
    post("solve/branch", { payoffs, lam_max: LAM_MAX, n_points: 120 }, ctl.signal)
      .then((d) => setFacts((f) => ({ ...f, branch: { lambdas: d.lambdas, rhos: d.rhos } })))
      .catch(() => undefined);
    return () => ctl.abort();
  }, [payoffs]);

  // Per-(game, λ) readings — debounced, stale requests aborted.
  const measure = useCallback(() => {
    if (!payoffs) return;
    abortRef.current?.abort();
    const ctl = new AbortController();
    abortRef.current = ctl;
    setBusy(true);
    setError("");
    const body = { payoffs, lam };
    Promise.all([
      post("solve/qre", body, ctl.signal),
      post("response", body, ctl.signal),
      post("dynamics/stationary", body, ctl.signal).catch((e) => ({ unavailable: e.message })),
    ])
      .then(([qre, resp, dyn]) => {
        if (ctl.signal.aborted) return;
        setReadings({
          sigma: qre.sigma,
          reciprocity: resp.reciprocity_defect,
          distance: resp.distance_to_criticality,
          rhoSb: resp.rho_sb,
          epr: "unavailable" in dyn ? undefined : dyn.epr,
          detailedBalance: "unavailable" in dyn ? undefined : dyn.detailed_balance,
          warnings: [...resp.warnings, ...("unavailable" in dyn ? [dyn.unavailable] : [])],
        });
        setBusy(false);
      })
      .catch((e: Error) => {
        if (ctl.signal.aborted) return;
        setError(e.message);
        setBusy(false);
      });
  }, [payoffs, lam]);

  useEffect(() => {
    const t = setTimeout(measure, 250);
    return () => clearTimeout(t);
  }, [measure]);

  const alphaPct = facts.alpha === undefined ? undefined : Math.round(facts.alpha * 100);

  return (
    <div className="wrap" style={{ paddingTop: "2.2rem" }}>
      <h1 style={{ marginBottom: "0.3rem" }}>Lab</h1>
      <p style={{ color: "var(--text-dim)", maxWidth: "44rem", marginTop: 0 }}>
        Every number below is computed on request by the deployed solver — float64 JAX on the
        backend, nothing simulated in the browser. Slide λ and watch the meters.
      </p>

      <div className="lab-grid" style={{ marginTop: "1.6rem" }}>
        {/* ---------- controls ---------- */}
        <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
          <div className="card">
            <div className="panel-label">Game</div>
            <select
              value={selected}
              onChange={(e) => setSelected(e.target.value)}
              style={{ width: "100%" }}
              aria-label="select game"
            >
              {Object.keys(examples).map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
            <div style={{ marginTop: "0.9rem" }}>
              <div className="panel-label">
                harmonic fraction α{" "}
                {alphaPct === undefined ? "" : `= ${facts.alpha?.toFixed(2)}`}
              </div>
              <div
                style={{
                  height: 8,
                  borderRadius: 4,
                  background: "var(--bg-raised)",
                  overflow: "hidden",
                  display: "flex",
                }}
                title="potential (teal) vs harmonic (amber) content"
              >
                <div
                  style={{
                    width: `${100 - (alphaPct ?? 0)}%`,
                    background: "var(--accent-dim)",
                    transition: "width 400ms",
                  }}
                />
                <div
                  style={{
                    width: `${alphaPct ?? 0}%`,
                    background: "var(--amber)",
                    transition: "width 400ms",
                  }}
                />
              </div>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  fontSize: "0.68rem",
                  color: "var(--text-faint)",
                  fontFamily: "var(--mono)",
                  marginTop: "0.25rem",
                }}
              >
                <span>potential → relaxes</span>
                <span>harmonic → circulates</span>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="panel-label">
              rationality λ = <span style={{ color: "var(--accent)" }}>{lam.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min={0}
              max={1}
              step={0.005}
              value={toSlider(lam)}
              onChange={(e) => setLam(fromSlider(Number(e.target.value)))}
              aria-label="lambda slider (log scale)"
            />
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                fontSize: "0.68rem",
                color: "var(--text-faint)",
                fontFamily: "var(--mono)",
              }}
            >
              <span>{LAM_MIN} · noise</span>
              <span>{LAM_MAX} · sharp</span>
            </div>
          </div>

          {payoffs && (
            <div className="card">
              <div className="panel-label">payoff matrix (u₁, u₂)</div>
              <PayoffMatrix payoffs={payoffs} />
            </div>
          )}

          {readings?.warnings.length ? (
            <div className="warnings">
              {readings.warnings.map((w) => (
                <div className="w" key={w}>
                  ⚠ {w}
                </div>
              ))}
            </div>
          ) : null}
          {error && (
            <div className="warnings">
              <div className="w" style={{ borderColor: "#6e2c38", color: "var(--red)" }}>
                ✕ {error}
              </div>
            </div>
          )}
        </div>

        {/* ---------- readings ---------- */}
        <div style={{ display: "flex", flexDirection: "column", gap: "1rem", opacity: busy ? 0.65 : 1, transition: "opacity 150ms" }}>
          <div className="card">
            <div className="panel-label">equilibrium mix σ*(λ)</div>
            <SigmaBars sigma={readings?.sigma} />
          </div>

          <div className="meter-grid">
            <div className="card">
              <Gauge
                value={readings?.reciprocity}
                min={0}
                max={1.3}
                danger={0.05}
                label="reciprocity defect ℛ"
                format={(v) => v.toExponential(2)}
              />
              <p style={{ fontSize: "0.75rem", color: "var(--text-faint)", textAlign: "center", margin: "0.4rem 0 0" }}>
                0 ⟺ potential game — exactly
              </p>
            </div>
            <div className="card">
              <Gauge
                value={readings?.rhoSb}
                min={0}
                max={1.3}
                danger={1.0}
                label="feedback gain ρ(SB)"
                format={(v) => v.toFixed(3)}
              />
              <p style={{ fontSize: "0.75rem", color: "var(--text-faint)", textAlign: "center", margin: "0.4rem 0 0" }}>
                ρ ≥ 1: supercritical, χ unreliable
              </p>
            </div>
            <div className="card">
              <div className="panel-label">entropy production σ</div>
              <div className="reading" data-tone={readings?.detailedBalance === false ? "warn" : undefined}>
                {readings?.epr === undefined ? "—" : readings.epr.toExponential(2)}
              </div>
              <div style={{ marginTop: "0.6rem" }}>
                {readings?.detailedBalance === undefined ? null : readings.detailedBalance ? (
                  <span className="badge" data-tone="ok">
                    detailed balance · equilibrium
                  </span>
                ) : (
                  <span className="badge" data-tone="warn">
                    driven · circulating current
                  </span>
                )}
              </div>
            </div>
          </div>

          <div className="card">
            <div className="panel-label">the whole branch — ρ(SB) along λ</div>
            <BranchTrace branch={facts.branch} lam={lam} />
            <p style={{ fontSize: "0.78rem", color: "var(--text-faint)", margin: "0.4rem 0 0" }}>
              On potential games this curve can be non-monotone — rise toward criticality, then
              escape as λ sharpens the equilibrium (finding F-0006). Harmonic games cross ρ = 1
              and stay there.
            </p>
          </div>
        </div>
      </div>

      <div style={{ marginTop: "1.4rem" }}>
        <GuessLambda />
      </div>
    </div>
  );
}
