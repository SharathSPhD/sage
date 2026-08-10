"use client";

import { useMemo, useState } from "react";
import { logitFlow, logitTrajectory, simplexToXY, type Matrix } from "../../../lib/qre";
import { LambdaSlider, PanelShell } from "./ui";

/* The fixed point made visible (doc 02): single-population logit dynamics on
   the 2-simplex. Click anywhere inside the triangle to seed a trajectory;
   the flow field shows where the loop settles — or circulates. */

const GAMES: Record<string, { u: Matrix; note: string }> = {
  "rock–paper–scissors": {
    u: [
      [0, -1, 1],
      [1, 0, -1],
      [-1, 1, 0],
    ],
    note: "harmonic: trajectories spiral around the centre — the circulation the dissipation meter measures",
  },
  coordination: {
    u: [
      [2, 0, 0],
      [0, 2, 0],
      [0, 0, 2],
    ],
    note: "potential: trajectories roll downhill to a corner (or the centre at low λ) and stop",
  },
};

const W = 420;
const H = 380;
const PAD = 30;
const SCALE = W - 2 * PAD;

function toSvg(p: number[]): [number, number] {
  const [x, y] = simplexToXY(p);
  return [PAD + x * SCALE, H - PAD - y * SCALE];
}

function fromSvg(sx: number, sy: number): number[] | null {
  const x = (sx - PAD) / SCALE;
  const y = (H - PAD - sy) / SCALE;
  const p2 = (2 * y) / Math.sqrt(3);
  const p1 = x - 0.5 * p2;
  const p0 = 1 - p1 - p2;
  if (p0 < 0.01 || p1 < 0.01 || p2 < 0.01) return null;
  return [p0, p1, p2];
}

function fieldSeeds(): number[][] {
  const pts: number[][] = [];
  for (let a = 1; a <= 5; a++)
    for (let b = 1; b <= 6 - a; b++) {
      const c = 7 - a - b;
      pts.push([a / 7, b / 7, c / 7]);
    }
  return pts;
}

export function SimplexPortrait() {
  const [gameName, setGameName] = useState<keyof typeof GAMES>("rock–paper–scissors");
  const [lam, setLam] = useState(2.0);
  const [seeds, setSeeds] = useState<number[][]>([[0.7, 0.2, 0.1]]);
  const { u, note } = GAMES[gameName];

  const arrows = useMemo(
    () =>
      fieldSeeds().map((p) => {
        const v = logitFlow(u, p, lam);
        const [x1, y1] = toSvg(p);
        const [x2, y2] = toSvg(p.map((c, i) => c + 0.12 * v[i]));
        return { x1, y1, x2, y2 };
      }),
    [u, lam],
  );

  const paths = useMemo(
    () =>
      seeds.map((s) =>
        logitTrajectory(u, s, lam, { dt: 0.06, steps: 500 })
          .map((p) => toSvg(p).map((c) => c.toFixed(1)).join(","))
          .join(" "),
      ),
    [seeds, u, lam],
  );

  return (
    <PanelShell title="the loop, settling — click inside the triangle" provenance="client">
      <div className="panel-cols" style={{ gridTemplateColumns: "auto 1fr" }}>
        <svg
          viewBox={`0 0 ${W} ${H}`}
          style={{ width: "100%", maxWidth: 440, cursor: "crosshair" }}
          tabIndex={0}
          role="button"
          aria-description="Press Enter or Space to seed a trajectory at a fresh interior point"
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              const n = seeds.length;
              const p = [0.2 + 0.5 * ((n * 7) % 10) / 10, 0.15 + 0.4 * ((n * 3) % 10) / 10, 0];
              p[2] = Math.max(0.05, 1 - p[0] - p[1]);
              const z = p[0] + p[1] + p[2];
              setSeeds((prev) => [...prev.slice(-4), p.map((c) => c / z)]);
            }
          }}
          onClick={(e) => {
            const rect = (e.target as SVGElement).closest("svg")!.getBoundingClientRect();
            const sx = ((e.clientX - rect.left) / rect.width) * W;
            const sy = ((e.clientY - rect.top) / rect.height) * H;
            const p = fromSvg(sx, sy);
            if (p) setSeeds((prev) => [...prev.slice(-4), p]);
          }}
          aria-label="simplex phase portrait"
        >
          <polygon
            points={`${toSvg([1, 0, 0])} ${toSvg([0, 1, 0])} ${toSvg([0, 0, 1])}`}
            fill="var(--panel-2)"
            stroke="var(--border-bright)"
          />
          {arrows.map((a, i) => (
            <line key={i} x1={a.x1} y1={a.y1} x2={a.x2} y2={a.y2} stroke="var(--text-faint)" strokeWidth="1" markerEnd="url(#arr)" />
          ))}
          <defs>
            <marker id="arr" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto">
              <path d="M0,0 L5,2.5 L0,5 z" fill="var(--text-faint)" />
            </marker>
          </defs>
          {paths.map((d, i) => (
            <polyline key={i} points={d} fill="none" stroke="var(--accent)" strokeWidth="1.6" opacity={0.5 + 0.5 * ((i + 1) / paths.length)} />
          ))}
          {seeds.map((s, i) => {
            const [x, y] = toSvg(s);
            return <circle key={i} cx={x} cy={y} r="3.5" fill="var(--amber)" />;
          })}
          {(["rock", "paper", "scissors"] as const).map((lbl, i) => {
            const corner = [
              [1, 0, 0],
              [0, 1, 0],
              [0, 0, 1],
            ][i];
            const [x, y] = toSvg(corner);
            return (
              <text key={lbl} x={x} y={y + (i === 2 ? -8 : 16)} textAnchor="middle" fontSize="10" fill="var(--text-dim)" fontFamily="var(--mono)">
                {gameName.startsWith("rock") ? lbl : `action ${i + 1}`}
              </text>
            );
          })}
        </svg>
        <div>
          <select value={gameName} onChange={(e) => { setGameName(e.target.value as keyof typeof GAMES); setSeeds([[0.7, 0.2, 0.1]]); }} aria-label="game" style={{ marginBottom: "1rem" }}>
            {Object.keys(GAMES).map((g) => (
              <option key={g}>{g}</option>
            ))}
          </select>
          <LambdaSlider lam={lam} setLam={setLam} min={0.2} max={20} />
          <p style={{ fontSize: "0.78rem", color: "var(--text-faint)", marginTop: "1rem" }}>
            {note}. The quantal response equilibrium is where the flow stops moving — watch it
            drift from the uniform centre (λ small) toward a Nash equilibrium (λ large).
          </p>
          <button onClick={() => setSeeds([])} style={{ marginTop: "0.6rem" }}>
            clear trajectories
          </button>
        </div>
      </div>
    </PanelShell>
  );
}
