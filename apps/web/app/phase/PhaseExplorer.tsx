"use client";

import { useMemo, useState } from "react";

export interface Surface {
  lambdas: number[];
  alphas: number[];
  grid: {
    rho: number[][];
    reciprocity: number[][];
    epr: number[][];
    supercritical_frac: number[][];
  };
}

const METRICS = {
  epr: {
    label: "entropy production",
    desc: "median exact EPR — where the dissipation lives. Zero along α = 0 (potential: equilibrium), growing with both α and λ.",
    log: true,
  },
  reciprocity: {
    label: "reciprocity defect ℛ",
    desc: "median ℛ — the observable proxy for harmonic content. Tracks α at every λ.",
    log: false,
  },
  rho: {
    label: "feedback gain ρ(SB)",
    desc: "median spectral radius of SB. The amber region is supercritical: the resolvent is near-singular and χ magnitudes are unreliable.",
    log: false,
  },
  supercritical_frac: {
    label: "supercritical fraction",
    desc: "fraction of games in each cell with ρ(SB) ≥ 1 — the wedge (finding F-0006): onset at median α = 0.5, escape at high λ on potential-leaning games.",
    log: false,
  },
} as const;

type MetricKey = keyof typeof METRICS;

/* graphite → teal → amber → red colour ramp */
function ramp(t: number): string {
  const stops = [
    [16, 22, 30],
    [23, 84, 72],
    [53, 224, 178],
    [232, 176, 75],
    [224, 89, 110],
  ];
  const x = Math.min(0.999, Math.max(0, t)) * (stops.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  const c = stops[i].map((a, k) => Math.round(a + f * (stops[i + 1][k] - a)));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

export function PhaseExplorer({ surface }: { surface: Surface }) {
  const [metric, setMetric] = useState<MetricKey>("epr");
  const [hover, setHover] = useState<{ ai: number; li: number } | null>(null);

  const { grid } = surface;
  const values = grid[metric];

  const { lo, hi } = useMemo(() => {
    const flat = values.flat().filter((v) => Number.isFinite(v));
    const cfg = METRICS[metric];
    const xs = cfg.log ? flat.map((v) => Math.log10(Math.max(v, 1e-12))) : flat;
    return { lo: Math.min(...xs), hi: Math.max(...xs) };
  }, [values, metric]);

  const norm = (v: number) => {
    const cfg = METRICS[metric];
    const x = cfg.log ? Math.log10(Math.max(v, 1e-12)) : v;
    return hi === lo ? 0.5 : (x - lo) / (hi - lo);
  };

  const nA = surface.alphas.length;
  const nL = surface.lambdas.length;
  const cell = 46;
  const padL = 46;
  const padB = 36;
  const W = padL + nL * cell + 8;
  const H = nA * cell + padB + 8;

  const hovered =
    hover && {
      alpha: surface.alphas[hover.ai],
      lam: surface.lambdas[hover.li],
      epr: grid.epr[hover.ai][hover.li],
      r: grid.reciprocity[hover.ai][hover.li],
      rho: grid.rho[hover.ai][hover.li],
      sup: grid.supercritical_frac[hover.ai][hover.li],
    };

  return (
    <div className="phase-layout" style={{ marginTop: "1.4rem" }}>
      <div className="card">
        <div className="metric-tabs" role="tablist" aria-label="phase-map metric">
          {(Object.keys(METRICS) as MetricKey[]).map((k) => (
            <button key={k} data-on={k === metric} onClick={() => setMetric(k)} role="tab">
              {METRICS[k].label}
            </button>
          ))}
        </div>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%" }} onMouseLeave={() => setHover(null)}>
          {surface.alphas.map((a, ai) =>
            surface.lambdas.map((l, li) => {
              const v = values[ai][li];
              const isSup = grid.supercritical_frac[ai][li] > 0;
              return (
                <g key={`${ai}-${li}`}>
                  <rect
                    x={padL + li * cell}
                    y={(nA - 1 - ai) * cell + 4}
                    width={cell - 2}
                    height={cell - 2}
                    rx={3}
                    fill={ramp(norm(v))}
                    stroke={
                      hover?.ai === ai && hover?.li === li
                        ? "var(--text)"
                        : isSup && metric !== "supercritical_frac"
                          ? "rgba(232,176,75,0.55)"
                          : "transparent"
                    }
                    strokeWidth={hover?.ai === ai && hover?.li === li ? 2 : 1}
                    strokeDasharray={isSup && !(hover?.ai === ai && hover?.li === li) ? "3 2" : undefined}
                    onMouseEnter={() => setHover({ ai, li })}
                  />
                </g>
              );
            }),
          )}
          {surface.alphas.map((a, ai) => (
            <text
              key={a}
              x={padL - 8}
              y={(nA - 1 - ai) * cell + cell / 2 + 8}
              textAnchor="end"
              fontSize="10"
              fill="var(--text-faint)"
              fontFamily="var(--mono)"
            >
              {a.toFixed(2)}
            </text>
          ))}
          {surface.lambdas.map((l, li) => (
            <text
              key={l}
              x={padL + li * cell + cell / 2}
              y={nA * cell + 18}
              textAnchor="middle"
              fontSize="10"
              fill="var(--text-faint)"
              fontFamily="var(--mono)"
            >
              {l}
            </text>
          ))}
          <text x={padL - 36} y={12} fontSize="10" fill="var(--text-dim)" fontFamily="var(--mono)">
            α ↑
          </text>
          <text x={W - 10} y={nA * cell + 32} fontSize="10" fill="var(--text-dim)" textAnchor="end" fontFamily="var(--mono)">
            λ →
          </text>
        </svg>
        <p style={{ fontSize: "0.8rem", color: "var(--text-faint)", margin: "0.6rem 0 0" }}>
          {METRICS[metric].desc} Dashed amber outlines mark cells containing supercritical games.
        </p>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
        <div className="card">
          <div className="panel-label">cell readout</div>
          {hovered ? (
            <table style={{ width: "100%", fontFamily: "var(--mono)", fontSize: "0.85rem" }}>
              <tbody>
                <tr>
                  <td style={{ color: "var(--text-faint)", padding: "0.2rem 0" }}>α</td>
                  <td style={{ textAlign: "right" }}>{hovered.alpha.toFixed(2)}</td>
                </tr>
                <tr>
                  <td style={{ color: "var(--text-faint)", padding: "0.2rem 0" }}>λ</td>
                  <td style={{ textAlign: "right" }}>{hovered.lam}</td>
                </tr>
                <tr>
                  <td style={{ color: "var(--text-faint)", padding: "0.2rem 0" }}>EPR</td>
                  <td style={{ textAlign: "right", color: "var(--accent)" }}>{hovered.epr.toExponential(2)}</td>
                </tr>
                <tr>
                  <td style={{ color: "var(--text-faint)", padding: "0.2rem 0" }}>ℛ</td>
                  <td style={{ textAlign: "right", color: "var(--accent)" }}>{hovered.r.toFixed(3)}</td>
                </tr>
                <tr>
                  <td style={{ color: "var(--text-faint)", padding: "0.2rem 0" }}>ρ(SB)</td>
                  <td style={{ textAlign: "right", color: hovered.rho >= 1 ? "var(--red)" : "var(--accent)" }}>
                    {hovered.rho.toFixed(3)}
                  </td>
                </tr>
                <tr>
                  <td style={{ color: "var(--text-faint)", padding: "0.2rem 0" }}>supercritical</td>
                  <td style={{ textAlign: "right" }}>{Math.round(hovered.sup * 100)}%</td>
                </tr>
              </tbody>
            </table>
          ) : (
            <p style={{ color: "var(--text-faint)", fontSize: "0.85rem", margin: 0 }}>
              hover a cell
            </p>
          )}
        </div>

        <div className="card">
          <div className="panel-label">how to read this</div>
          <p style={{ fontSize: "0.85rem", color: "var(--text-dim)", margin: 0 }}>
            Bottom edge (α = 0): potential games — every meter quiet, dynamics relax to a Gibbs
            equilibrium. Top edge (α = 1): harmonic games — broken reciprocity, circulating
            probability current, positive dissipation. The interesting physics is in between:
            inside the supercritical wedge the meters <em>decouple</em> — response asymmetry and
            dissipation stop co-moving (findings F-0004 and F-0007).
          </p>
        </div>
      </div>
    </div>
  );
}
