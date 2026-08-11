"use client";

import { useMemo, useState } from "react";
import { PanelShell } from "../components/panels/ui";

/* F-0008/F-0009 made visible: the actual July-2026 SP15 series, its weekly
   irreversibility ratios, and the null-band verdict — all from the committed
   gate artifact, no recomputation. */

export interface Series {
  node: string;
  start: string;
  hours: string[];
  prices: number[];
  verdict: {
    kld_embed_per_hour: number;
    null_markov_median: number;
    null_markov_q99: number;
    markov_detected: number;
    weekly_ratios_f0009: number[];
  };
}

const W = 900;
const H = 240;

export function MarketReading({ series }: { series: Series }) {
  const [hoverWeek, setHoverWeek] = useState<number | null>(null);
  const { prices, verdict } = series;
  const ratios = verdict.weekly_ratios_f0009;

  const pts = useMemo(() => {
    const max = Math.max(...prices);
    const min = Math.min(...prices);
    return prices
      .map((p, i) => {
        const x = 40 + (i / (prices.length - 1)) * (W - 60);
        const y = H - 30 - ((p - min) / (max - min)) * (H - 55);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  }, [prices]);

  const maxRatio = Math.max(...ratios);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.2rem", marginTop: "1.4rem" }}>
      <PanelShell title={`the raw data · ${series.node} · hourly DAM LMP`} provenance="client">
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%" }} aria-label="hourly day-ahead prices with weekly irreversibility shading">
          {ratios.map((r, w) => {
            const x0 = 40 + ((w * 168) / (prices.length - 1)) * (W - 60);
            const x1 = 40 + (Math.min((w + 1) * 168, prices.length - 1) / (prices.length - 1)) * (W - 60);
            return (
              <rect
                key={w}
                x={x0}
                y={12}
                width={x1 - x0}
                height={H - 42}
                fill="var(--amber)"
                opacity={0.04 + 0.28 * (r / maxRatio)}
                stroke={hoverWeek === w ? "var(--amber)" : "transparent"}
                onMouseEnter={() => setHoverWeek(w)}
                onMouseLeave={() => setHoverWeek(null)}
              />
            );
          })}
          <polyline points={pts} fill="none" stroke="var(--accent)" strokeWidth="1.1" />
          {ratios.map((r, w) => {
            const x0 = 40 + ((w * 168 + 84) / (prices.length - 1)) * (W - 60);
            return (
              <text key={w} x={x0} y={H - 14} textAnchor="middle" fontSize="10" fill={r > 3 ? "var(--amber)" : "var(--text-faint)"} fontFamily="var(--mono)">
                wk{w + 1} · {r.toFixed(1)}×
              </text>
            );
          })}
          <text x={40} y={12} fontSize="10" fill="var(--text-faint)" fontFamily="var(--mono)">
            $/MWh — shading = weekly irreversibility ratio vs its reversible null (F-0009 stratification)
          </text>
        </svg>
        <p style={{ fontSize: "0.78rem", color: "var(--text-faint)", margin: "0.5rem 0 0" }}>
          Every week reads above its null median (the signal never changes sign); the statistical
          weight concentrates in the high-ramp second half — week 4 at 9.1× — exactly where the
          summer scarcity drive is strongest.
        </p>
      </PanelShell>

      <div className="panel-cols">
        <PanelShell title="the verdict, against the right null" provenance="client">
          <table style={{ width: "100%", fontFamily: "var(--mono)", fontSize: "0.88rem" }}>
            <tbody>
              <tr>
                <td style={{ color: "var(--text-faint)", padding: "0.25rem 0" }}>observed pair-flux KLD</td>
                <td style={{ textAlign: "right", color: "var(--amber)" }}>
                  {verdict.kld_embed_per_hour.toFixed(4)} nats/h
                </td>
              </tr>
              <tr>
                <td style={{ color: "var(--text-faint)", padding: "0.25rem 0" }}>reversible-null median</td>
                <td style={{ textAlign: "right" }}>{verdict.null_markov_median.toFixed(4)}</td>
              </tr>
              <tr>
                <td style={{ color: "var(--text-faint)", padding: "0.25rem 0" }}>reversible-null q99</td>
                <td style={{ textAlign: "right" }}>{verdict.null_markov_q99.toFixed(4)}</td>
              </tr>
              <tr style={{ borderTop: "1px solid var(--border)" }}>
                <td style={{ padding: "0.35rem 0" }}>verdict</td>
                <td style={{ textAlign: "right" }}>
                  <span className="badge" data-tone={verdict.markov_detected ? "warn" : "ok"}>
                    {verdict.markov_detected ? "driven cycle — detailed balance violated (p < 0.01)" : "at null"}
                  </span>
                </td>
              </tr>
            </tbody>
          </table>
          <p style={{ fontSize: "0.78rem", color: "var(--text-faint)", marginTop: "0.7rem" }}>
            The null that matters: a reversible Markov chain with the data&apos;s own persistence
            (symmetrized pair flux). Spectral surrogates (FT, AAFT) are also computed and
            reported — the path through their failures, including one retraction, is finding
            F-0008/F-0009 in the open repository.
          </p>
        </PanelShell>

        <PanelShell title="what the meter measures here" provenance="client">
          <p style={{ fontSize: "0.85rem", color: "var(--text-dim)", margin: 0 }}>
            Prices are binned; each hour becomes a state (price level, direction of last move).
            A time-reversible market would cross each pair of states equally often in both
            directions. It doesn&apos;t: the daily loop — night valley, morning ramp, evening
            peak, decline — runs one way around, at about <strong>1.1 nats of entropy
            production per day</strong> at hourly resolution. The same instrument, calibrated on
            games where the answer is provable (a road network reads exactly zero;
            rock–paper–scissors reads loudly), read this off a real market. Real-time 5-minute
            prices, by contrast, are genuinely at-null at this statistic — dwell dominates —
            and that contrast is part of the finding.
          </p>
        </PanelShell>
      </div>
    </div>
  );
}
