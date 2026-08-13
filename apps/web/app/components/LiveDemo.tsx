"use client";

/* The demonstration on the landing page: one input moves, and the answer
 * moves with it.
 *
 * A two-firm price grid is solved in the browser with lib/qre.ts — the same
 * fixed-point iteration as the library, held to the library's own goldens by
 * scripts/test-qre.mjs — so the panel answers instantly and keeps answering
 * if the network is slow. The studio at /solve calls the deployed solver.
 *
 * The rival's cost sweeps back and forth on its own; grabbing the slider
 * takes over. Under prefers-reduced-motion the sweep never starts and the
 * slider is the only way it moves.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { solveQRE } from "../../lib/qre";
import { usePrefersReducedMotion } from "../../lib/anim";
import { CurveChart } from "./charts/CurveChart";
import { TweenNumber } from "./charts/Tween";

const LEVELS = [1.19, 1.29, 1.39, 1.49, 1.59, 1.69, 1.79, 1.89];
const OWN_COST = 1.0;
const SENSITIVITY = 3.6;
const OUTSIDE = 1.65;
const UNITS = 400;
const PRECISION = 1.6;

const money = (v: number) => `$${v.toFixed(2)}`;
const money0 = (v: number) => `$${Math.round(v).toLocaleString()}`;

/** Logit shares over the two prices and an outside option, then profit. */
function payoffTable(rivalCost: number) {
  const anchor = SENSITIVITY * OUTSIDE;
  const u1: number[][] = [];
  const u2: number[][] = [];
  for (let i = 0; i < LEVELS.length; i++) {
    u1.push([]);
    u2.push([]);
    for (let j = 0; j < LEVELS.length; j++) {
      const a = Math.exp(anchor - SENSITIVITY * LEVELS[i]);
      const b = Math.exp(anchor - SENSITIVITY * LEVELS[j]);
      const z = 1 + a + b;
      u1[i].push(UNITS * (a / z) * (LEVELS[i] - OWN_COST));
      u2[i].push(UNITS * (b / z) * (LEVELS[j] - rivalCost));
    }
  }
  return { u1, u2 };
}

export function LiveDemo() {
  const reduced = usePrefersReducedMotion();
  const [rivalCost, setRivalCost] = useState(1.05);
  const [running, setRunning] = useState(true);
  const dir = useRef(1);

  useEffect(() => {
    if (reduced || !running) return;
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const dt = Math.min(64, now - last) / 1000;
      last = now;
      setRivalCost((c) => {
        let next = c + dir.current * 0.11 * dt;
        if (next > 1.35) {
          next = 1.35;
          dir.current = -1;
        }
        if (next < 0.72) {
          next = 0.72;
          dir.current = 1;
        }
        return next;
      });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [reduced, running]);

  const solved = useMemo(() => {
    const { u1, u2 } = payoffTable(rivalCost);
    const qre = solveQRE({ u1, u2 }, PRECISION);
    const profits = u1.map((row) => row.reduce((acc, v, j) => acc + v * qre.sigma2[j], 0));
    const best = profits.indexOf(Math.max(...profits));
    const rivalMean = LEVELS.reduce((acc, p, j) => acc + p * qre.sigma2[j], 0);
    return { profits, best, rivalMean };
  }, [rivalCost]);

  const price = LEVELS[solved.best];
  const profit = solved.profits[solved.best];

  return (
    <div className="demo-panel">
      <div className="demo-head">
        <h2>Your price, against a rival setting one too</h2>
        <span className="badge" data-tone="accent">
          live
        </span>
      </div>

      <div className="demo-figure">
        <span className="reading">
          <TweenNumber value={price} format={money} ms={260} />
        </span>
        <span className="demo-delta">
          earns <TweenNumber value={profit} format={money0} ms={260} /> · they will sit near{" "}
          <TweenNumber value={solved.rivalMean} format={money} ms={260} />
        </span>
      </div>

      <CurveChart
        x={LEVELS}
        y={solved.profits}
        markIndex={solved.best}
        formatX={money}
        formatY={money0}
        xLabel="your price"
        yLabel="expected profit"
        markLabel="set this"
        tableCaption="The profit at every price on this grid"
      />

      <div className="demo-controls">
        <label htmlFor="demo-rival-cost">Their unit cost</label>
        <input
          id="demo-rival-cost"
          type="range"
          min={0.7}
          max={1.4}
          step={0.01}
          value={Number(rivalCost.toFixed(2))}
          onChange={(e) => {
            setRunning(false);
            setRivalCost(Number(e.target.value));
          }}
        />
        <output className="mono" style={{ minWidth: "3.4rem", textAlign: "right", fontSize: "var(--text-sm)" }}>
          {money(rivalCost)}
        </output>
        {!reduced && (
          <button type="button" data-quiet="true" onClick={() => setRunning((r) => !r)} aria-pressed={running}>
            {running ? "Pause" : "Play"}
          </button>
        )}
      </div>

      <p className="demo-caption">
        Cheaper rival, softer price and a thinner margin; dearer rival, and the whole curve lifts. The recommendation
        never chases their price one-for-one, because they are choosing too.{" "}
        <Link href="/solve">Open the studio</Link> to do this with your own numbers.
      </p>
    </div>
  );
}
