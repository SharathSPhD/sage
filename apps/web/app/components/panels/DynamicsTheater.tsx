"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Bars, LambdaSlider, PanelShell } from "./ui";

/* Doc 06 made visible: the joint profile space drawn as a lattice, stationary
   probability as node mass, the exact probability current J* as animated
   directed flow. Potential game: still water. Harmonic game: the water turns.
   Everything here is the deployed float64 solver's exact reading. */

const GAMES: Record<string, { payoffs: number[][][]; note: string }> = {
  "rock–paper–scissors": {
    payoffs: [
      [
        [0, -1, 1],
        [1, 0, -1],
        [-1, 1, 0],
      ],
      [
        [0, 1, -1],
        [-1, 0, 1],
        [1, -1, 0],
      ],
    ],
    note: "stationary but TURNING: probability flows in closed loops (rock→paper→scissors…), dissipating forever",
  },
  coordination: {
    payoffs: [
      [
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2],
      ],
      [
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2],
      ],
    ],
    note: "detailed balance: every edge carries equal flow both ways — the current is zero, the water is still",
  },
};

interface Stationary {
  pi: number[];
  currents: number[][];
  states: number[][];
  epr: number;
  detailed_balance: boolean;
}

interface SampleRead {
  exact_epr: number;
  kld_epr: number;
  tur_point: number;
  tur_ci_low: number;
}

const W = 460;
const H = 420;

function nodeXY(state: number[], n1: number, n2: number): [number, number] {
  const pad = 70;
  const x = pad + (state[0] / Math.max(n1 - 1, 1)) * (W - 2 * pad);
  const y = pad + (state[1] / Math.max(n2 - 1, 1)) * (H - 2 * pad);
  return [x, y];
}

export function DynamicsTheater() {
  const [gameName, setGameName] = useState<keyof typeof GAMES>("rock–paper–scissors");
  const [lam, setLam] = useState(1.5);
  const [data, setData] = useState<Stationary | null>(null);
  const [sample, setSample] = useState<SampleRead | null>(null);
  const [busy, setBusy] = useState(false);
  const [sampling, setSampling] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const load = useCallback(() => {
    abortRef.current?.abort();
    const ctl = new AbortController();
    abortRef.current = ctl;
    setBusy(true);
    fetch("/api/v1/dynamics/stationary", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ payoffs: GAMES[gameName].payoffs, lam }),
      signal: ctl.signal,
    })
      .then((r) => r.json())
      .then((d: Stationary) => {
        if (!ctl.signal.aborted) {
          setData(d);
          setBusy(false);
        }
      })
      .catch(() => !ctl.signal.aborted && setBusy(false));
  }, [gameName, lam]);

  useEffect(() => {
    const t = setTimeout(load, 300);
    return () => clearTimeout(t);
  }, [load]);

  const drawSample = useCallback(() => {
    setSampling(true);
    setSample(null);
    fetch("/api/v1/dynamics/sample", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        payoffs: GAMES[gameName].payoffs,
        lam,
        n_steps: 15000,
        n_trajectories: 8,
        seed: 42,
      }),
    })
      .then((r) => r.json())
      .then((d: SampleRead) => {
        setSample(d);
        setSampling(false);
      })
      .catch(() => setSampling(false));
  }, [gameName, lam]);

  const n1 = GAMES[gameName].payoffs[0].length;
  const n2 = GAMES[gameName].payoffs[0][0].length;
  const maxJ = data ? Math.max(...data.currents.flat().map(Math.abs), 1e-12) : 1;
  // absolute floor: on a detailed-balance chain currents are ~1e-16 numerical
  // noise — never draw them scaled up to "flow"
  const showFlow = data ? maxJ > 1e-8 && !data.detailed_balance : false;
  const maxPi = data ? Math.max(...data.pi) : 1;

  return (
    <PanelShell title="the joint profile space — where the water turns" provenance="live">
      <div className="panel-cols" style={{ gridTemplateColumns: "auto 1fr" }}>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxWidth: 480, opacity: busy ? 0.6 : 1, transition: "opacity 150ms" }} aria-label="stationary distribution and probability currents on the joint profile lattice">
          <defs>
            <marker id="flow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
              <path d="M0,0 L6,3 L0,6 z" fill="var(--amber)" />
            </marker>
          </defs>
          {data &&
            data.currents.map((row, a) =>
              row.map((j, b) => {
                if (!showFlow || b <= a || Math.abs(j) < 0.02 * maxJ) return null;
                // net flow direction: j = pi_a w_ab - pi_b w_ba > 0 means a -> b
                const from = j > 0 ? a : b;
                const to = j > 0 ? b : a;
                const [x1, y1] = nodeXY(data.states[from], n1, n2);
                const [x2, y2] = nodeXY(data.states[to], n1, n2);
                const mag = Math.abs(j) / maxJ;
                const dur = 3.5 - 2.8 * mag;
                return (
                  <line
                    key={`${a}-${b}`}
                    x1={x1}
                    y1={y1}
                    x2={x2}
                    y2={y2}
                    stroke="var(--amber)"
                    strokeWidth={1 + 3.5 * mag}
                    strokeDasharray="6 7"
                    markerEnd="url(#flow)"
                    opacity={0.35 + 0.6 * mag}
                  >
                    <animate attributeName="stroke-dashoffset" from="13" to="0" dur={`${dur.toFixed(2)}s`} repeatCount="indefinite" />
                  </line>
                );
              }),
            )}
          {data &&
            data.states.map((s, k) => {
              const [x, y] = nodeXY(s, n1, n2);
              const r = 6 + 22 * Math.sqrt(data.pi[k] / maxPi);
              return (
                <g key={k}>
                  <circle cx={x} cy={y} r={r} fill="var(--accent-dim)" opacity="0.55" />
                  <circle cx={x} cy={y} r={r} fill="none" stroke="var(--accent)" strokeWidth="1" />
                  <text x={x} y={y + 3} textAnchor="middle" fontSize="9" fill="var(--text)" fontFamily="var(--mono)">
                    {s.map((v) => v + 1).join(",")}
                  </text>
                </g>
              );
            })}
          <text x={W / 2} y={H - 8} textAnchor="middle" fontSize="10" fill="var(--text-faint)" fontFamily="var(--mono)">
            P1 action →
          </text>
          <text x={14} y={H / 2} textAnchor="middle" fontSize="10" fill="var(--text-faint)" fontFamily="var(--mono)" transform={`rotate(-90 14 ${H / 2})`}>
            P2 action →
          </text>
        </svg>

        <div>
          <select value={gameName} onChange={(e) => setGameName(e.target.value as keyof typeof GAMES)} aria-label="game" style={{ marginBottom: "0.9rem" }}>
            {Object.keys(GAMES).map((g) => (
              <option key={g}>{g}</option>
            ))}
          </select>
          <LambdaSlider lam={lam} setLam={setLam} min={0.3} max={6} />
          {data && (
            <div style={{ marginTop: "0.9rem" }}>
              <div className="reading" data-tone={data.detailed_balance ? undefined : "warn"} style={{ fontSize: "1.25rem" }}>
                σ_EP = {data.epr.toExponential(2)}
              </div>
              <span className="badge" data-tone={data.detailed_balance ? "ok" : "warn"} style={{ marginTop: "0.4rem" }}>
                {data.detailed_balance ? "detailed balance · still water" : "circulating · driven steady state"}
              </span>
              <p style={{ fontSize: "0.78rem", color: "var(--text-faint)", marginTop: "0.7rem" }}>
                {GAMES[gameName].note}. Node size = stationary probability π; animated amber
                edges = the exact net current J*, thicker and faster where the flow is stronger.
              </p>
            </div>
          )}
          <div style={{ marginTop: "1rem" }}>
            <button data-primary="true" onClick={drawSample} disabled={sampling}>
              {sampling ? "sampling trajectories…" : "estimate σ_EP from trajectories alone"}
            </button>
            {sample && (
              <div style={{ marginTop: "0.7rem" }}>
                <Bars
                  values={[sample.exact_epr, sample.kld_epr, sample.tur_ci_low]}
                  labels={["exact", "KLD est.", "TUR cert."]}
                  format={(v) => v.toExponential(2)}
                />
                <p style={{ fontSize: "0.74rem", color: "var(--text-faint)", marginTop: "0.4rem" }}>
                  8×15k sampled revision steps, seeded. The KLD estimator recovers the exact
                  meter; the TUR value is the certified lower bound (the point estimate can
                  legitimately straddle exact near equilibrium).
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </PanelShell>
  );
}
