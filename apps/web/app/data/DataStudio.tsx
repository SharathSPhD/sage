"use client";

/* Bring your own data.
 *
 * Load a table of observed choices, say which column is what, name the
 * problem, and the page does two calls: POST /v1/fit to estimate the
 * precision the choices imply, then POST /v1/solve/* on the levels the data
 * actually contains, at that precision. Both calls, and what came back, are
 * downloadable, and the same run is printed as a Python script.
 *
 * Nothing is invented. The levels are the levels in the file; the counts are
 * the counts in the file; the only numbers you supply are the ones the
 * payoffs need and no table can give (costs, demand, capacity).
 */

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { CopyButton } from "../components/CopyButton";
import { DistributionBars } from "../components/charts/DistributionBars";
import { CurveChart } from "../components/charts/CurveChart";
import { TweenNumber } from "../components/charts/Tween";
import { inferMapping, numericColumn, parseTable, toLevels, type Table } from "../../lib/table";

type Kind = "pricing" | "auction" | "electricity";

interface Example {
  id: string;
  file: string;
  name: string;
  detail: string;
  provenance: "real" | "synthetic";
  provenanceNote: string;
  kind: Kind;
}

const EXAMPLES: Example[] = [
  {
    id: "prices",
    file: "/examples/synthetic-price-points.csv",
    name: "Shelf prices, two chains",
    detail: "60 store-weeks, both chains' chosen price off the same eight-step ladder.",
    provenance: "synthetic",
    provenanceNote: "Generated from a seeded logit draw so the fit has something to recover. Not a real retailer.",
    kind: "pricing",
  },
  {
    id: "bids",
    file: "/examples/synthetic-tender-bids.csv",
    name: "Sealed tender bids",
    detail: "90 rounds, the bid level chosen off an eight-step ladder, with the cost.",
    provenance: "synthetic",
    provenanceNote: "Generated from a seeded logit draw. Not a real procurement record.",
    kind: "auction",
  },
  {
    id: "caiso",
    file: "/examples/caiso.csv",
    name: "CAISO day-ahead prices",
    detail: "840 hourly SP15 day-ahead prices, July 2026, binned into eight levels.",
    provenance: "real",
    provenanceNote: "Copied from the committed, gate-checked artifact behind the market reading. No resampling.",
    kind: "electricity",
  },
];

/** The field names strataq.fit actually serialises. */
interface FitResult {
  lam_hat: number | null;
  ci_low?: number | null;
  ci_high?: number | null;
  ci_method?: string;
  ci_level?: number;
  identified?: boolean;
  kind?: string;
  loglik?: number;
  n_obs?: number;
  summary?: string;
  warnings?: string[];
  refusals?: string[];
  lr_nash?: { p?: number; p_boundary?: number } | null;
  lr_uniform?: { p?: number; p_boundary?: number } | null;
}

const money = (v: number) => `$${v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
const money0 = (v: number) => `$${Math.round(v).toLocaleString()}`;
const sig = (v: number) => (Math.abs(v) >= 1e5 || (v !== 0 && Math.abs(v) < 1e-3) ? v.toExponential(2) : Number(v.toPrecision(3)).toString());

export function DataStudio() {
  const [table, setTable] = useState<Table | null>(null);
  const [source, setSource] = useState<{ label: string; provenance: "real" | "synthetic" | "yours"; note: string } | null>(null);
  const [pasted, setPasted] = useState("");
  const [dragging, setDragging] = useState(false);
  const [mapping, setMapping] = useState({ level: -1, rival: -1, count: -1, group: -1 });
  const [kind, setKind] = useState<Kind>("pricing");
  const [ownCost, setOwnCost] = useState(1.0);
  const [rivalCost, setRivalCost] = useState(1.05);
  const [marketSize, setMarketSize] = useState(400);
  const [sensitivity, setSensitivity] = useState(3.6);
  const [demandMW, setDemandMW] = useState(80);
  const [capacity, setCapacity] = useState(100);
  const [fit, setFit] = useState<FitResult | null>(null);
  const [fitting, setFitting] = useState(false);
  const [solved, setSolved] = useState<Record<string, unknown> | null>(null);
  const [solving, setSolving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement | null>(null);

  const load = (text: string, label: string, provenance: "real" | "synthetic" | "yours", note: string, k?: Kind) => {
    const parsed = parseTable(text);
    if (parsed.columns.length === 0 || parsed.rows.length === 0) {
      setError("That file had no rows this page could read. A header row and at least one numeric column are needed.");
      return;
    }
    setError(null);
    setTable(parsed);
    setSource({ label, provenance, note });
    setMapping(inferMapping(parsed));
    if (k) setKind(k);
    setFit(null);
    setSolved(null);
  };

  const openExample = async (ex: Example) => {
    try {
      const r = await fetch(ex.file);
      load(await r.text(), ex.name, ex.provenance, ex.provenanceNote, ex.kind);
    } catch {
      setError("The bundled example could not be loaded.");
    }
  };

  const readFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = () => load(String(reader.result ?? ""), file.name, "yours", "Your file. It never leaves the browser except as the counts sent to the solver.", undefined);
    reader.readAsText(file);
  };

  // ---- what the mapping produces ----
  const observed = useMemo(() => {
    if (!table || mapping.level < 0) return null;
    const own = numericColumn(table, mapping.level);
    if (own.length < 4) return null;
    const rival = mapping.rival >= 0 ? numericColumn(table, mapping.rival) : [];
    const combined = rival.length ? [...own, ...rival] : own;
    const { levels, binned } = toLevels(combined, 8);
    const countOn = (values: number[]) => {
      const counts = new Array(levels.length).fill(0);
      for (const v of values) {
        let best = 0;
        for (let i = 1; i < levels.length; i++) {
          if (Math.abs(v - levels[i]) < Math.abs(v - levels[best])) best = i;
        }
        counts[best] += 1;
      }
      return counts;
    };
    const ownCounts = countOn(own);
    const rivalCounts = rival.length ? countOn(rival) : ownCounts;
    return { levels, ownCounts, rivalCounts, binned, symmetric: rival.length === 0, n: own.length + rival.length };
  }, [table, mapping]);

  // ---- the payoff table the fit is against ----
  const payoffs = useMemo(() => {
    if (!observed) return null;
    const L = observed.levels;
    const u1: number[][] = [];
    const u2: number[][] = [];
    for (let i = 0; i < L.length; i++) {
      u1.push([]);
      u2.push([]);
      for (let j = 0; j < L.length; j++) {
        if (kind === "pricing") {
          const anchor = sensitivity * (Math.max(...L) + 0.1);
          const a = Math.exp(anchor - sensitivity * L[i]);
          const b = Math.exp(anchor - sensitivity * L[j]);
          const z = 1 + a + b;
          u1[i].push(marketSize * (a / z) * (L[i] - ownCost));
          u2[i].push(marketSize * (b / z) * (L[j] - rivalCost));
        } else if (kind === "auction") {
          const win = (mine: number, theirs: number) => (mine < theirs ? 1 : mine > theirs ? 0 : 0.5);
          u1[i].push((L[i] - ownCost) * win(L[i], L[j]));
          u2[i].push((L[j] - rivalCost) * win(L[j], L[i]));
        } else {
          const clear = Math.max(L[i], L[j]);
          const bothNeeded = demandMW > capacity;
          const mine = bothNeeded ? Math.min(capacity, demandMW - capacity) : L[i] <= L[j] ? Math.min(capacity, demandMW) : 0;
          const theirs = bothNeeded ? Math.min(capacity, demandMW - capacity) : L[j] < L[i] ? Math.min(capacity, demandMW) : 0;
          u1[i].push(mine * (clear - ownCost));
          u2[i].push(theirs * (clear - rivalCost));
        }
      }
    }
    return { u1, u2 };
  }, [observed, kind, ownCost, rivalCost, marketSize, sensitivity, demandMW, capacity]);

  const runFit = async () => {
    if (!observed || !payoffs) return;
    setFitting(true);
    setError(null);
    try {
      const r = await fetch("/api/v1/fit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          payoffs: [payoffs.u1, payoffs.u2],
          counts: [observed.ownCounts, observed.rivalCounts],
          method: "mle",
          ci: "bootstrap",
          n_boot: 200,
        }),
      });
      const payload = await r.json();
      if (!r.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail));
      setFit(payload as FitResult);
    } catch (e) {
      setError(`The fit could not be run: ${(e as Error).message}`);
    } finally {
      setFitting(false);
    }
  };

  const solveBody = useMemo(() => {
    if (!observed) return null;
    const precision = fit?.lam_hat ?? null;
    if (precision === null) return null;
    if (kind === "pricing") {
      const anchor = sensitivity * (Math.max(...observed.levels) + 0.1);
      return {
        costs: [ownCost, rivalCost],
        grid: observed.levels,
        demand: { kind: "logit", price_sensitivity: sensitivity, quality: [anchor, anchor], market_size: marketSize },
        precision,
      };
    }
    if (kind === "auction") {
      return { costs: [ownCost, rivalCost], grid: observed.levels, precision };
    }
    return {
      costs: [ownCost, rivalCost],
      offers: observed.levels,
      capacities: [capacity, capacity],
      demand: demandMW,
      precision,
    };
  }, [observed, fit, kind, ownCost, rivalCost, marketSize, sensitivity, capacity, demandMW]);

  const runSolve = async () => {
    if (!solveBody) return;
    setSolving(true);
    setError(null);
    try {
      const r = await fetch(`/api/v1/solve/${kind}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(solveBody),
      });
      const payload = await r.json();
      if (!r.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail));
      setSolved(payload as Record<string, unknown>);
    } catch (e) {
      setError(`The solve could not be run: ${(e as Error).message}`);
    } finally {
      setSolving(false);
    }
  };

  useEffect(() => {
    setSolved(null);
  }, [kind, fit]);

  const download = (name: string, text: string, type: string) => {
    const url = URL.createObjectURL(new Blob([text], { type }));
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.click();
    URL.revokeObjectURL(url);
  };

  const script = useMemo(() => buildScript(kind, observed, payoffs, solveBody), [kind, observed, payoffs, solveBody]);

  const headline = readHeadline(kind, solved);

  return (
    <div className="wrap page">
      <h1 className="surface-title">Bring your own data</h1>
      <p className="surface-lede">
        Load a table of choices somebody actually made — prices set, bids submitted, offers made — and this page fits
        the precision those choices imply, then solves the problem on the levels your file contains. Five steps, two
        API calls, and a script at the end that reproduces both.
      </p>

      {error && (
        <p className="studio-error" role="alert" style={{ marginBottom: "1.2rem" }}>
          {error}
        </p>
      )}

      <div className="steps">
        {/* 1 ---------------------------------------------------------------- */}
        <section className="card step-card" data-done={table ? "true" : "false"}>
          <div className="step-head">
            <span className="step-num">1</span>
            <h2>Load a table</h2>
          </div>
          <button
            type="button"
            className="dropzone"
            data-dragging={dragging}
            onClick={() => fileInput.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              const file = e.dataTransfer.files?.[0];
              if (file) readFile(file);
            }}
          >
            <strong>Drop a CSV here, or choose a file</strong>
            <span>Header row, one numeric column of chosen levels. Nothing is uploaded until you run the fit.</span>
          </button>
          <input
            ref={fileInput}
            type="file"
            accept=".csv,.tsv,.txt,text/csv"
            className="visually-hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) readFile(file);
            }}
          />

          <p className="panel-label" style={{ marginTop: "1.2rem" }}>
            Or start from one of these
          </p>
          <div className="example-grid">
            {EXAMPLES.map((ex) => (
              <button key={ex.id} type="button" onClick={() => openExample(ex)}>
                <span>{ex.name}</span>
                <span className="ex-detail">{ex.detail}</span>
                <span className="badge" data-provenance={ex.provenance}>
                  {ex.provenance === "real" ? "real data" : "generated"}
                </span>
              </button>
            ))}
          </div>

          <details style={{ marginTop: "1rem" }}>
            <summary style={{ cursor: "pointer", color: "var(--text-2)", fontSize: "var(--text-base)" }}>
              Paste instead
            </summary>
            <textarea
              value={pasted}
              onChange={(e) => setPasted(e.target.value)}
              rows={6}
              style={{ width: "100%", marginTop: "0.6rem", fontFamily: "var(--mono)", fontSize: "var(--text-sm)" }}
              placeholder={"price,rival_price\n1.49,1.59\n1.39,1.49"}
              aria-label="Paste comma-separated values"
            />
            <button
              type="button"
              style={{ marginTop: "0.6rem" }}
              onClick={() => load(pasted, "pasted table", "yours", "Pasted into this page.")}
              disabled={pasted.trim() === ""}
            >
              Use this
            </button>
          </details>

          {source && (
            <p className="callout" style={{ marginTop: "1rem" }}>
              <strong>{source.label}</strong> · {table?.rows.length.toLocaleString()} rows ·{" "}
              {table?.columns.length} columns. {source.note}
            </p>
          )}
        </section>

        {/* 2 ---------------------------------------------------------------- */}
        <section className="card step-card" data-disabled={table ? "false" : "true"} data-done={observed ? "true" : "false"}>
          <div className="step-head">
            <span className="step-num">2</span>
            <h2>Say which column is what</h2>
          </div>
          {!table ? (
            <p className="figure-note">Load a table first.</p>
          ) : (
            <>
              <p className="lead-note">
                These are guesses from the column names and their contents. Change any of them — inference on somebody
                else&apos;s file is a guess however good it is.
              </p>
              <div className="mapping-grid">
                <label>
                  Chosen level (yours)
                  <select
                    value={mapping.level}
                    onChange={(e) => setMapping({ ...mapping, level: Number(e.target.value) })}
                  >
                    {table.columns.map((c, i) => (
                      <option key={i} value={i}>
                        {c}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Chosen level (the other side)
                  <select
                    value={mapping.rival}
                    onChange={(e) => setMapping({ ...mapping, rival: Number(e.target.value) })}
                  >
                    <option value={-1}>not in this file</option>
                    {table.columns.map((c, i) => (
                      <option key={i} value={i}>
                        {c}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Row label
                  <select
                    value={mapping.group}
                    onChange={(e) => setMapping({ ...mapping, group: Number(e.target.value) })}
                  >
                    <option value={-1}>none</option>
                    {table.columns.map((c, i) => (
                      <option key={i} value={i}>
                        {c}
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              <div className="preview-table" style={{ marginTop: "1rem" }}>
                <table>
                  <thead>
                    <tr>
                      {table.columns.map((c, i) => (
                        <th key={i} scope="col">
                          {c}
                          {i === mapping.level ? " ◂ yours" : i === mapping.rival ? " ◂ theirs" : ""}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {table.rows.slice(0, 8).map((r, i) => (
                      <tr key={i}>
                        {r.map((cell, j) => (
                          <td key={j}>{cell}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {observed && (
                <>
                  <h3 style={{ marginTop: "1.2rem" }}>What the file actually contains</h3>
                  <DistributionBars
                    rows={observed.levels.map((l, i) => ({
                      label: kind === "pricing" ? money(l) : sig(l),
                      p: observed.ownCounts[i] / Math.max(1, observed.ownCounts.reduce((a, b) => a + b, 0)),
                    }))}
                    limit={8}
                  />
                  <p className="figure-note">
                    {observed.n.toLocaleString()} observations over {observed.levels.length} levels
                    {observed.binned ? ", binned from a continuous column into equal-width levels" : ", taken exactly as they appear"}.
                    {observed.symmetric
                      ? " Only one side is mapped, so both players are fitted from the same observed distribution — a symmetric read. Map a second column to fit the two sides separately."
                      : ""}
                  </p>
                </>
              )}
            </>
          )}
        </section>

        {/* 3 ---------------------------------------------------------------- */}
        <section className="card step-card" data-disabled={observed ? "false" : "true"}>
          <div className="step-head">
            <span className="step-num">3</span>
            <h2>Name the problem</h2>
          </div>
          <div className="situation-picker" role="group" aria-label="Problem type">
            {(["pricing", "auction", "electricity"] as Kind[]).map((k) => (
              <button key={k} type="button" data-on={k === kind} aria-pressed={k === kind} onClick={() => setKind(k)}>
                {k === "pricing" ? "Pricing" : k === "auction" ? "Auction and tender" : "Electricity offers"}
              </button>
            ))}
          </div>
          <p className="lead-note" style={{ marginTop: "0.9rem" }}>
            The levels come from your file. These are the numbers no table can supply, and they are what turns a column
            of choices into a payoff a precision can be fitted against.
          </p>
          <div className="knob-grid">
            <NumberField label="Your unit cost" value={ownCost} onChange={setOwnCost} step={kind === "auction" ? 1000 : 0.01} />
            <NumberField label="The other side's unit cost" value={rivalCost} onChange={setRivalCost} step={kind === "auction" ? 1000 : 0.01} />
            {kind === "pricing" && (
              <>
                <NumberField label="Category units per period" value={marketSize} onChange={setMarketSize} step={10} />
                <NumberField label="Price sensitivity" value={sensitivity} onChange={setSensitivity} step={0.1} />
              </>
            )}
            {kind === "electricity" && (
              <>
                <NumberField label="Demand (MW)" value={demandMW} onChange={setDemandMW} step={5} />
                <NumberField label="Capacity each (MW)" value={capacity} onChange={setCapacity} step={5} />
              </>
            )}
          </div>
        </section>

        {/* 4 ---------------------------------------------------------------- */}
        <section className="card step-card" data-disabled={observed ? "false" : "true"} data-done={fit?.lam_hat != null ? "true" : "false"}>
          <div className="step-head">
            <span className="step-num">4</span>
            <h2>Estimate what the data can pin down</h2>
          </div>
          <p className="lead-note">
            <code>POST /v1/fit</code> takes the payoff table above and your observed counts and returns the precision
            that best explains those choices, with an interval that names the method that produced it. Where the
            likelihood is flat it returns no number at all, because a refusal is a bound and not an estimate.
          </p>
          <button type="button" data-primary="true" onClick={runFit} disabled={!observed || fitting}>
            {fitting ? "Fitting…" : "Fit the precision"}
          </button>

          {fit && (
            <div className="fit-figures" style={{ marginTop: "1.2rem" }}>
              <div>
                <div className="panel-label">Precision</div>
                <div className="reading" data-tone="accent">
                  {fit.lam_hat === null ? "not identified" : <TweenNumber value={fit.lam_hat} format={sig} />}
                </div>
                <p className="figure-note">
                  {fit.ci_low != null && fit.ci_high != null
                    ? `${((fit.ci_level ?? 0.95) * 100).toFixed(0)}% interval ${sig(fit.ci_low)} to ${sig(fit.ci_high)}`
                    : "no interval returned"}
                </p>
              </div>
              <div>
                <div className="panel-label">Observations</div>
                <div className="reading">{(fit.n_obs ?? observed?.n ?? 0).toLocaleString()}</div>
                <p className="figure-note">choices in the likelihood</p>
              </div>
              <div>
                <div className="panel-label">Against exact optimising</div>
                <div className="reading">{fit.lr_nash?.p != null ? `p = ${sig(fit.lr_nash.p)}` : "—"}</div>
                <p className="figure-note">likelihood ratio test against the Nash limit</p>
              </div>
              <div>
                <div className="panel-label">Against picking at random</div>
                <div className="reading">{fit.lr_uniform?.p != null ? `p = ${sig(fit.lr_uniform.p)}` : "—"}</div>
                <p className="figure-note">likelihood ratio test against uniform choice</p>
              </div>
            </div>
          )}

          {fit?.refusals?.length ? (
            <div className="callout" data-tone="warn" style={{ marginTop: "1rem" }}>
              <strong>What these data cannot settle</strong>
              <ul style={{ margin: "0.4rem 0 0", paddingLeft: "1.1rem" }}>
                {fit.refusals.map((r, i) => (
                  <li key={i} style={{ marginBottom: "0.3rem" }}>
                    {r}
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
          {fit?.ci_method && (
            <p className="figure-note" style={{ marginTop: "0.8rem" }}>
              Interval method: {fit.ci_method}
            </p>
          )}
          {fit?.summary && (
            <details className="under-the-hood" style={{ marginTop: "1rem" }}>
              <summary>The estimator&apos;s own printout</summary>
              <pre className="code-block" style={{ marginBottom: "0.9rem" }}>
                {fit.summary}
              </pre>
            </details>
          )}
          {fit?.warnings?.length ? (
            <div className="warnings" style={{ marginTop: "0.8rem" }}>
              {fit.warnings.map((w, i) => (
                <p key={i} className="w">
                  {w}
                </p>
              ))}
            </div>
          ) : null}
        </section>

        {/* 5 ---------------------------------------------------------------- */}
        <section className="card step-card" data-disabled={solveBody ? "false" : "true"} data-done={solved ? "true" : "false"}>
          <div className="step-head">
            <span className="step-num">5</span>
            <h2>Solve on your levels</h2>
          </div>
          <p className="lead-note">
            <code>POST /v1/solve/{kind}</code> with the grid taken from your file and the precision from step 4.
          </p>
          <button type="button" data-primary="true" onClick={runSolve} disabled={!solveBody || solving}>
            {solving ? "Solving…" : "Solve"}
          </button>

          {solved && headline && (
            <>
              <section className="answer" style={{ marginTop: "1.2rem" }}>
                <div className="answer-head">
                  <h2 style={{ margin: 0 }}>Result</h2>
                  <span className="badge" data-tone="accent">
                    {source?.label}
                  </span>
                </div>
                <p className="answer-verb">{headline.text}</p>
                <div className="answer-figures">
                  {headline.figures.map((f) => (
                    <div key={f.label}>
                      <div className="panel-label">{f.label}</div>
                      <div className="reading">{f.value}</div>
                      {f.note && <p className="figure-note">{f.note}</p>}
                    </div>
                  ))}
                </div>
              </section>
              {headline.curve && (
                <div style={{ marginTop: "1.2rem" }}>
                  <CurveChart
                    x={headline.curve.x}
                    y={headline.curve.y}
                    markIndex={headline.curve.mark}
                    formatX={headline.curve.formatX}
                    formatY={money0}
                    xLabel={headline.curve.xLabel}
                    yLabel={headline.curve.yLabel}
                  />
                </div>
              )}
            </>
          )}
        </section>

        {/* 6 ---------------------------------------------------------------- */}
        <section className="card step-card" data-disabled={solved ? "false" : "true"}>
          <div className="step-head">
            <span className="step-num">6</span>
            <h2>Take it with you</h2>
          </div>
          <div className="controls-actions" style={{ marginTop: 0, paddingTop: 0, borderTop: "none" }}>
            <button
              type="button"
              disabled={!solved}
              onClick={() =>
                download(
                  "sage-result.json",
                  JSON.stringify({ source: source?.label, fit, request: solveBody, solution: solved }, null, 2),
                  "application/json",
                )
              }
            >
              Download the result as JSON
            </button>
            <button
              type="button"
              disabled={!solved || !headline?.curve}
              onClick={() => {
                if (!headline?.curve) return;
                const rows = headline.curve.x.map((v, i) => `${v},${headline.curve!.y[i]}`);
                download("sage-curve.csv", `level,value\n${rows.join("\n")}\n`, "text/csv");
              }}
            >
              Download the curve as CSV
            </button>
          </div>
          <div className="code-head" style={{ marginTop: "1.2rem" }}>
            <span className="panel-label">The same run, in Python</span>
            <CopyButton text={script} />
          </div>
          <pre className="code-block">{script}</pre>
          <p className="figure-note">
            <code>pip install strataq</code>. This script does the fit and the solve locally, with no service in the
            middle, and returns the same numbers. The <Link href="/api">API console</Link> has the HTTP form of both
            calls.
          </p>
        </section>
      </div>
    </div>
  );
}

function NumberField({
  label,
  value,
  onChange,
  step,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  step: number;
}) {
  return (
    <label className="knob">
      <span className="knob-label">{label}</span>
      <input
        type="number"
        step={step}
        value={value}
        onChange={(e) => {
          const v = Number(e.target.value);
          if (Number.isFinite(v)) onChange(v);
        }}
      />
    </label>
  );
}

interface Headline {
  text: string;
  figures: { label: string; value: string; note?: string }[];
  curve?: { x: number[]; y: number[]; mark: number; formatX: (v: number) => string; xLabel: string; yLabel: string };
}

function readHeadline(kind: Kind, solved: Record<string, unknown> | null): Headline | null {
  if (!solved) return null;
  const g = <T,>(key: string): T | undefined => solved[key] as T | undefined;

  if (kind === "pricing") {
    const price = g<number>("price");
    const grid = g<number[]>("price_grid") ?? [];
    const curve = g<number[]>("profit_curve") ?? [];
    if (price === undefined) return null;
    return {
      text: `Price at ${money(price)}.`,
      figures: [
        { label: "Expected profit", value: money0(g<number>("profit") ?? 0), note: "per period at this price" },
        { label: "Margin", value: money(g<number>("margin") ?? 0), note: "price minus your unit cost" },
        {
          label: "Fitted precision",
          value: sig(g<number>("precision") ?? 0),
          note: "from your observed choices",
        },
      ],
      curve: {
        x: grid,
        y: curve,
        mark: grid.findIndex((p) => Math.abs(p - price) < 1e-9),
        formatX: money,
        xLabel: "your price",
        yLabel: "expected profit",
      },
    };
  }

  if (kind === "auction") {
    const bid = g<number>("bid");
    const grid = g<number[]>("bid_grid") ?? [];
    const curve = g<number[]>("surplus_curve") ?? [];
    if (bid === undefined) return null;
    return {
      text: `Bid ${money0(bid)}.`,
      figures: [
        { label: "Expected surplus", value: money0(g<number>("surplus") ?? 0), note: "bid minus cost, times the chance it wins" },
        { label: "Win probability", value: `${(100 * (g<number>("win_probability") ?? 0)).toFixed(0)}%` },
        { label: "Fitted precision", value: sig(g<number>("precision") ?? 0), note: "from your observed choices" },
      ],
      curve: {
        x: grid,
        y: curve,
        mark: grid.findIndex((p) => Math.abs(p - bid) < 1e-9),
        formatX: money0,
        xLabel: "your bid",
        yLabel: "expected surplus",
      },
    };
  }

  const offer = g<number>("offer");
  const grid = g<number[]>("offers") ?? [];
  const curve = g<number[]>("profit_curve") ?? [];
  if (offer === undefined) return null;
  return {
    text: `Offer at ${money(offer)}/MWh.`,
    figures: [
      { label: "Expected clearing price", value: `${money(g<number>("clearing_price") ?? 0)}/MWh` },
      { label: "Expected revenue", value: money0(g<number>("revenue") ?? 0) },
      { label: "Fitted precision", value: sig(g<number>("precision") ?? 0), note: "from your observed choices" },
    ],
    curve: {
      x: grid,
      y: curve,
      mark: grid.findIndex((p) => Math.abs(p - offer) < 1e-9),
      formatX: (v: number) => money(v),
      xLabel: "your offer",
      yLabel: "expected profit",
    },
  };
}

function buildScript(
  kind: Kind,
  observed: { levels: number[]; ownCounts: number[]; rivalCounts: number[] } | null,
  payoffs: { u1: number[][]; u2: number[][] } | null,
  solveBody: Record<string, unknown> | null,
): string {
  if (!observed || !payoffs) return "# Load a table and map its columns to see the script.";
  const py = (v: unknown) => JSON.stringify(v);
  const constructor =
    kind === "pricing"
      ? `sq.PricingProblem(\n    costs=${py((solveBody?.costs as number[]) ?? [])},\n    grid=${py(observed.levels)},\n    demand=sq.LogitDemand(${
          (solveBody?.demand as { price_sensitivity?: number })?.price_sensitivity ?? 3.6
        }, ${py((solveBody?.demand as { quality?: number[] })?.quality ?? [])}, market_size=${
          (solveBody?.demand as { market_size?: number })?.market_size ?? 1
        }),\n    precision=lam,\n)`
      : kind === "auction"
        ? `sq.AuctionProblem(\n    costs=${py((solveBody?.costs as number[]) ?? [])},\n    grid=${py(observed.levels)},\n    precision=lam,\n)`
        : `sq.ElectricityProblem(\n    costs=${py((solveBody?.costs as number[]) ?? [])},\n    offers=${py(observed.levels)},\n    capacities=${py(
            (solveBody?.capacities as number[]) ?? [],
          )},\n    demand=${solveBody?.demand ?? 1},\n    precision=lam,\n)`;

  return `"""Fit the precision your observations imply, then solve at it.

Reproduces exactly what the page did: the levels and counts below are read
off the table you loaded; the payoffs are the ones step 3 built.
"""

import strataq as sq
from strataq.finite.games.tensor import DenseTensorGame

levels = ${py(observed.levels)}
counts = [${py(observed.ownCounts)}, ${py(observed.rivalCounts)}]

payoffs = [
    ${py(payoffs.u1)},
    ${py(payoffs.u2)},
]
game = DenseTensorGame(payoffs)

estimate = sq.fit(game, counts, method="mle", ci="bootstrap", n_boot=200)
print(estimate.summary())

lam = estimate.lam_hat
solution = ${constructor}.solve()
print(solution.as_dict())
`;
}
