"use client";

/* Splitting a fixed budget across three accounts against a rival doing the
 * same. Every allocation is enumerated and solved exactly by the deployed
 * solver — there is no sampling here and nothing fitted. Previously /blotto.
 */

import { useEffect, useState } from "react";
import { Term } from "../../components/Term";

type Read = {
  allocations_a: number[][];
  allocations_b: number[][];
  sigma_a: number[];
  sigma_b: number[];
  alpha: number;
  r: number;
  epr: number | null;
  n_joint_states: number;
  warnings: string[];
};

function split(a: number[]): string {
  return a.join(" · ");
}

function Plan({ allocations, sigma, who }: { allocations: number[][]; sigma: number[]; who: string }) {
  const order = sigma.map((_, i) => i).sort((a, b) => sigma[b] - sigma[a]);
  const max = Math.max(...sigma, 1e-9);
  const top = order[0];
  return (
    <div>
      <h3>{who}</h3>
      <p className="lead-note">
        Most likely split: <strong>{split(allocations[top])}</strong> across the three accounts, in{" "}
        {(100 * sigma[top]).toFixed(0)}% of rounds.
      </p>
      <ul className="rival-bars">
        {order.slice(0, 8).map((i) => (
          <li key={i}>
            <span className="rb-label">{split(allocations[i])}</span>
            <span className="rb-track">
              <span className="rb-fill" style={{ width: `${Math.max(0.6, (100 * sigma[i]) / max)}%` }} />
            </span>
            <span className="rb-val">{(100 * sigma[i]).toFixed(1)}%</span>
          </li>
        ))}
      </ul>
      {sigma.length > 8 && <p className="figure-note">{sigma.length - 8} rarer splits not shown.</p>}
    </div>
  );
}

export function AllocationLab() {
  const [yours, setYours] = useState(3);
  const [theirs, setTheirs] = useState(3);
  const [sharpness, setSharpness] = useState(1.5);
  const [read, setRead] = useState<Read | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const t = setTimeout(() => {
      setBusy(true);
      fetch("/api/v1/domains/blotto/read", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ budget_a: yours, budget_b: theirs, lam: sharpness }),
      })
        .then(async (r) => {
          if (!r.ok) throw new Error((await r.json()).detail ?? `HTTP ${r.status}`);
          return r.json();
        })
        .then((b: Read) => {
          setRead(b);
          setError(null);
        })
        .catch((e) => setError(String((e as Error).message ?? e)))
        .finally(() => setBusy(false));
    }, 250);
    return () => clearTimeout(t);
  }, [yours, theirs, sharpness]);

  const spread = read ? 1 - Math.max(...read.sigma_a) : 0;

  return (
    <div className="studio">
      <section className="answer" aria-labelledby="alloc-answer">
        <div className="answer-head">
          <h2 id="alloc-answer">Do this</h2>
          <span className="badge" data-tone="ok">
            solved by the deployed solver
          </span>
        </div>
        {error && (
          <p className="studio-error" role="alert">
            The solver could not return an answer for these budgets: {error}
          </p>
        )}
        {read && (
          <>
            <p className="answer-verb">
              Split your {yours} units as {split(read.allocations_a[read.sigma_a.indexOf(Math.max(...read.sigma_a))])} —
              and vary it.
            </p>
            <p className="lead-note">
              {spread > 0.6
                ? "No single split is safe. Any plan you commit to every quarter is one the rival can cover for free; the value is in being unpredictable."
                : "One split does most of the work here, because the budgets are far enough apart that the stronger side can cover and the weaker side has to gamble."}
            </p>
          </>
        )}
      </section>

      <section className="card controls" aria-labelledby="alloc-controls">
        <h3 id="alloc-controls">Your numbers</h3>
        <div className="knob-grid">
          {[
            { label: "Your budget", value: yours, set: setYours, min: 1, max: 8, step: 1, fmt: (v: number) => `${v} units` },
            { label: "Their budget", value: theirs, set: setTheirs, min: 1, max: 8, step: 1, fmt: (v: number) => `${v} units` },
            {
              label: "How closely they chase their best split",
              value: sharpness,
              set: setSharpness,
              min: 0.2,
              max: 6,
              step: 0.1,
              fmt: (v: number) => v.toFixed(1),
            },
          ].map((s) => (
            <label key={s.label} className="knob">
              <span className="knob-label">
                {s.label}
                <output aria-hidden="true">{s.fmt(s.value)}</output>
              </span>
              <input
                type="range"
                aria-label={s.label}
                min={s.min}
                max={s.max}
                step={s.step}
                value={s.value}
                onChange={(e) => s.set(Number(e.target.value))}
              />
            </label>
          ))}
        </div>
      </section>

      {read && (
        <div className="answer-cols" style={{ opacity: busy ? 0.6 : 1, transition: "opacity 150ms" }}>
          <section className="card">
            <Plan allocations={read.allocations_a} sigma={read.sigma_a} who="Your plan" />
          </section>
          <section className="card">
            <Plan allocations={read.allocations_b} sigma={read.sigma_b} who="What they are likely to do" />
          </section>
        </div>
      )}

      {read?.warnings?.length ? <p className="studio-warn">{read.warnings.join(" · ")}</p> : null}

      {read && (
        <details className="under-the-hood">
          <summary>The numbers behind this</summary>
          <ul>
            <li>
              <Term term="Harmonic fraction α" explain="The share of the game's structure that circulates rather than climbing a gradient. 0 is a pure landscape, 1 is pure cycling." />{" "}
              = {read.alpha.toFixed(3)}
            </li>
            <li>
              <Term term="Reciprocity defect ℛ" explain="How far the two sides' cross-responses fail to mirror each other. Zero means a change you make moves them exactly as much as the same change by them would move you." />{" "}
              = {read.r.toFixed(3)}
            </li>
            <li>
              <Term term="Entropy production σ_EP" explain="The rate at which the strategy dynamics dissipate — positive means probability circulates forever rather than settling." />{" "}
              = {read.epr === null ? "not computed at this size" : read.epr.toExponential(2)}
            </li>
            <li>{read.n_joint_states.toLocaleString()} joint splits enumerated exactly.</li>
          </ul>
        </details>
      )}
    </div>
  );
}
