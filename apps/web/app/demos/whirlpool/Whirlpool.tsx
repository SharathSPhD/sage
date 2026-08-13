"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { glauber, mixGame } from "../../../lib/demos/gametheory";
import { Readout, Widget } from "../components/chrome";
import { useDragNumber } from "../components/drag";
import { useAnimationFrame, useInView, useReducedMotion } from "../components/motion";

const LABELS = ["R", "P", "S"];
const VIEW_W = 440;
const VIEW_H = 420;

/** Grid position of joint profile (i, j): row player picks the row, column player the column. */
const nodeXY = (i: number, j: number) => ({ x: 90 + j * 130, y: 80 + i * 130 });

interface Edge {
  a: number;
  b: number;
  /** Quadratic control point, or null for a straight edge. */
  cx: number | null;
  cy: number | null;
}

const EDGES: Edge[] = (() => {
  const out: Edge[] = [];
  const id = (i: number, j: number) => i * 3 + j;
  for (let i = 0; i < 3; i++) {
    out.push({ a: id(i, 0), b: id(i, 1), cx: null, cy: null });
    out.push({ a: id(i, 1), b: id(i, 2), cx: null, cy: null });
    const y = nodeXY(i, 0).y;
    out.push({ a: id(i, 0), b: id(i, 2), cx: 220, cy: i === 0 ? y - 54 : y + 54 });
  }
  for (let j = 0; j < 3; j++) {
    out.push({ a: id(0, j), b: id(1, j), cx: null, cy: null });
    out.push({ a: id(1, j), b: id(2, j), cx: null, cy: null });
    const x = nodeXY(0, j).x;
    out.push({ a: id(0, j), b: id(2, j), cx: j === 0 ? x - 54 : x + 54, cy: 210 });
  }
  return out;
})();

function pointOn(e: Edge, t: number) {
  const A = nodeXY(Math.floor(e.a / 3), e.a % 3);
  const B = nodeXY(Math.floor(e.b / 3), e.b % 3);
  if (e.cx === null || e.cy === null) {
    return { x: A.x + (B.x - A.x) * t, y: A.y + (B.y - A.y) * t };
  }
  const u = 1 - t;
  return {
    x: u * u * A.x + 2 * u * t * e.cx + t * t * B.x,
    y: u * u * A.y + 2 * u * t * e.cy + t * t * B.y,
  };
}

function pathFor(e: Edge) {
  const A = nodeXY(Math.floor(e.a / 3), e.a % 3);
  const B = nodeXY(Math.floor(e.b / 3), e.b % 3);
  if (e.cx === null || e.cy === null) return `M${A.x} ${A.y}L${B.x} ${B.y}`;
  return `M${A.x} ${A.y}Q${e.cx} ${e.cy} ${B.x} ${B.y}`;
}

export function Whirlpool() {
  const [alpha, setAlpha] = useState(0.78);
  const [lam, setLam] = useState(2);
  const [hostRef, inView] = useInView<HTMLDivElement>();
  const reduced = useReducedMotion();

  const reading = useMemo(() => {
    const [u1, u2] = mixGame(alpha);
    return glauber(u1, u2, lam);
  }, [alpha, lam]);

  const isZero = reading.ep < 1e-12;
  const verdict = isZero ? "landscape" : reading.ep < 0.02 ? "near-landscape" : "whirlpool";

  const dial = useDragNumber({
    value: alpha,
    min: 0,
    max: 1,
    step: 0.01,
    onChange: setAlpha,
    axis: "x",
    travelPx: 320,
    label: "Game character: 0 is an exact potential game, 1 is rock–paper–scissors",
    valueText: (v) =>
      `${v === 0 ? "exact potential game" : v === 1 ? "rock\u2013paper\u2013scissors" : `${(v * 100).toFixed(0)} percent toward rock\u2013paper\u2013scissors`}; entropy production ${isZero ? "exactly zero" : reading.ep.toFixed(3)} nats per step`,
  });

  return (
    <>
      <Widget
        hook="Drag the game from a landscape to a whirlpool"
        lede={
          <p>
            These nine circles are the nine things that can be true at once when two players each pick rock, paper or
            scissors. Probability moves between them one player at a time. Grab the knob and change what kind of game it
            is.
          </p>
        }
        consequence={
          isZero ? (
            <>
              At the potential end every edge cancels: the flow one way exactly equals the flow the other, entropy
              production is zero, and no particle has anywhere to go. That zero is a theorem about this chain, not a
              rounding convention.
            </>
          ) : (
            <>
              The edges no longer cancel. Probability circulates round a loop for as long as the game exists, and the
              entropy-production counter is the price of keeping it turning.
            </>
          )
        }
        maths={
          <>
            <p>
              The chain is the Glauber (logit revision) dynamic on the nine joint profiles: at each step one player is
              drawn at random and redraws its action from softmax(λ · u<sub>i</sub>(other&apos;s current action)). π is
              its stationary distribution by power iteration to 10⁻¹⁵; the net current on an edge is J<sub>xy</sub> =
              π<sub>x</sub>W<sub>xy</sub> − π<sub>y</sub>W<sub>yx</sub>; entropy production is Schnakenberg&apos;s
              σ = Σ<sub>x&lt;y</sub> (π<sub>x</sub>W<sub>xy</sub> − π<sub>y</sub>W<sub>yx</sub>) log(π<sub>x</sub>W
              <sub>xy</sub> / π<sub>y</sub>W<sub>yx</sub>), in nats per step.
            </p>
            <p>
              At α = 0 the game is an exact potential game, the chain satisfies detailed balance with respect to
              exp(λ·potential), and σ vanishes identically. In float64 it reads{" "}
              <code>{reading.ep.toExponential(3)}</code> and the largest net current reads{" "}
              <code>{reading.maxCurrent.toExponential(3)}</code>. The committed benchmark makes the same statement on
              the library&apos;s exact solver: <code>benchmarks/results/equilibrium_reads_zero.json</code> records{" "}
              <code>max_epr = 4.19e-30</code>, <code>max_current = 9.71e-17</code> over its potential-game suite, and{" "}
              <code>benchmarks/results/ness_reads_positive.json</code> records <code>min_epr = 1.568</code> over its
              harmonic suite.
            </p>
            <p>
              α here is the convex mixing weight between the coordination game and rock–paper–scissors, not the Hodge
              harmonic fraction of the normalised game — those differ, which is why Colonel Blotto reads α = 0.694
              rather than 1 (<code>benchmarks/results/blotto_readings.json</code>).
            </p>
          </>
        }
      >
        <div className="whirl-layout">
          <div className="whirl-figure" ref={hostRef}>
            <Lattice reading={reading} inView={inView && !reduced} />
          </div>
          <div className="whirl-controls">
            <div className="morph-dial">
              <div className="panel-label" id="morph-label">
                Game character
              </div>
              <div className="morph-track">
                <div className="morph-fill" style={{ width: `${alpha * 100}%` }} />
                <div
                  className="morph-knob"
                  {...dial.handleProps}
                  data-dragging={dial.dragging ? "true" : undefined}
                  style={{ ...dial.handleProps.style, left: `calc(${alpha * 100}% - 14px)` }}
                />
              </div>
              <div className="morph-ends">
                <span style={{ color: "var(--q-landscape-text)" }}>landscape</span>
                <span style={{ color: "var(--q-whirlpool-text)" }}>whirlpool</span>
              </div>
            </div>

            <div className="demo-control">
              <label htmlFor="whirl-lam" className="panel-label">
                Precision λ = <span style={{ color: "var(--accent-strong)" }}>{lam.toFixed(1)}</span>
              </label>
              <input
                id="whirl-lam"
                type="range"
                min={0.2}
                max={8}
                step={0.1}
                value={lam}
                onChange={(e) => setLam(Number(e.target.value))}
              />
            </div>

            <div className="demo-readouts demo-readouts-col">
              <Readout
                label="entropy production"
                value={isZero ? "0.000" : reading.ep.toFixed(3)}
                unit="nats/step"
                tone={isZero ? "landscape" : "whirlpool"}
                live
              />
              <Readout
                label="largest net current"
                value={reading.maxCurrent < 1e-12 ? "0.000" : reading.maxCurrent.toFixed(3)}
                tone={isZero ? "landscape" : "whirlpool"}
              />
              <div className="demo-readout">
                <div className="panel-label">reading</div>
                <p className="verdict-line" data-tone={isZero ? "landscape" : "whirlpool"} aria-live="polite">
                  <span aria-hidden>{isZero ? "◇" : "◉"}</span> {verdict}
                </p>
              </div>
            </div>

            <button type="button" className="btn" onClick={() => setAlpha(0)}>
              Snap to the potential end
            </button>
          </div>
        </div>
      </Widget>
    </>
  );
}

function Lattice({ reading, inView }: { reading: ReturnType<typeof glauber>; inView: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const particles = useRef<{ e: number; dir: 1 | -1; t: number }[]>([]);
  const [, force] = useState(0);

  const maxJ = Math.max(1e-12, reading.maxCurrent);

  useEffect(() => {
    const next: { e: number; dir: 1 | -1; t: number }[] = [];
    EDGES.forEach((e, k) => {
      const j = reading.J[e.a][e.b];
      if (Math.abs(j) < 1e-11) return;
      const n = Math.min(12, Math.max(1, Math.round((Math.abs(j) / maxJ) * 12)));
      for (let p = 0; p < n; p++) next.push({ e: k, dir: j > 0 ? 1 : -1, t: p / n });
    });
    particles.current = next;
    force((v) => v + 1);
  }, [reading, maxJ]);

  useAnimationFrame((dt) => {
    const cv = canvasRef.current;
    if (!cv) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = cv.clientWidth;
    const h = cv.clientHeight;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(h * dpr);
    }
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.setTransform((cv.width / VIEW_W), 0, 0, cv.height / VIEW_H, 0, 0);
    ctx.clearRect(0, 0, VIEW_W, VIEW_H);
    const style = getComputedStyle(document.documentElement).getPropertyValue("--q-whirlpool").trim() || "#c44e52";
    ctx.fillStyle = style;
    for (const p of particles.current) {
      p.t += (dt / 1000) * 0.42;
      if (p.t > 1) p.t -= 1;
      const e = EDGES[p.e];
      const pos = pointOn(e, p.dir === 1 ? p.t : 1 - p.t);
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }, inView);

  return (
    <div className="lattice">
      <canvas ref={canvasRef} className="lattice-canvas" aria-hidden />
      <svg viewBox={`0 0 ${VIEW_W} ${VIEW_H}`} className="lattice-svg" role="img" aria-label="Nine joint outcomes of a two-player three-action game. Edges carry probability between them; an arrow marks an edge where the two directions do not cancel.">
        <defs>
          <marker id="whirl-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M0 0 L10 5 L0 10 z" fill="var(--q-whirlpool)" />
          </marker>
        </defs>
        {EDGES.map((e, k) => {
          const j = reading.J[e.a][e.b];
          const mag = Math.abs(j);
          const dead = mag < 1e-11;
          const d = pathFor(e);
          const flip = j < 0;
          return (
            <path
              key={k}
              d={flip ? reversePath(e) : d}
              fill="none"
              stroke={dead ? "var(--text-3)" : "var(--q-whirlpool)"}
              strokeWidth={dead ? 1.2 : Math.min(5, 1.4 + (mag / maxJ) * 3.6)}
              strokeDasharray={dead ? "3 4" : undefined}
              opacity={dead ? 0.7 : Math.min(1, 0.4 + (mag / maxJ) * 0.6)}
              markerEnd={dead ? undefined : "url(#whirl-arrow)"}
            />
          );
        })}
        {reading.pi.map((p, k) => {
          const i = Math.floor(k / 3);
          const j = k % 3;
          const { x, y } = nodeXY(i, j);
          const r = 13 + Math.sqrt(p) * 26;
          return (
            <g key={k}>
              <circle className="morph" cx={x} cy={y} r={r} fill="var(--surface)" stroke="var(--q-landscape)" strokeWidth={2} />
              <text x={x} y={y + 4} textAnchor="middle" fontSize={12} fontFamily="var(--mono)" fill="var(--text)">
                {LABELS[i]}
                {LABELS[j]}
              </text>
            </g>
          );
        })}
      </svg>
      <p className="lattice-legend">
        Circle size is how often that pair of moves occurs. An <span style={{ color: "var(--q-whirlpool-text)" }}>arrowed
        edge</span> carries net flow in the arrow&apos;s direction; a <span style={{ color: "var(--text-2)" }}>thin dashed
        edge</span> is one where the two directions cancel exactly.
      </p>
    </div>
  );
}

function reversePath(e: Edge) {
  const A = nodeXY(Math.floor(e.a / 3), e.a % 3);
  const B = nodeXY(Math.floor(e.b / 3), e.b % 3);
  if (e.cx === null || e.cy === null) return `M${B.x} ${B.y}L${A.x} ${A.y}`;
  return `M${B.x} ${B.y}Q${e.cx} ${e.cy} ${A.x} ${A.y}`;
}
