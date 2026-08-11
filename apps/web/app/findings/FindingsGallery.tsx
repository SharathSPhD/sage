"use client";

import { useState } from "react";
import Link from "next/link";
import { PanelShell } from "../components/panels/ui";

export interface Charts {
  decoupling: { levels: number[]; rho: number[] };
  frontier: { alphas: number[]; lambda_c: number[] };
  estimators: { alphas: number[]; exact: number[]; kld: number[]; tur_ci_low: number[] };
  crossover_signs: { lambdas: number[]; corr_a095: number[]; m4_a095: number };
}

type Tone = "ok" | "warn" | "hot" | undefined;

const FINDINGS: {
  id: string;
  kind: string;
  tone: Tone;
  title: string;
  body: string;
  playable?: { href: string; label: string };
}[] = [
  { id: "F-0001", kind: "calibration", tone: undefined, title: "ℛ exceeds 1 on matching pennies", body: "ℛ is a norm ratio, unbounded above: values > 1 mean the circulating response dominates the reciprocal part. Docs corrected; the meter was right." },
  { id: "F-0002", kind: "correction", tone: "warn", title: "Only ℛ's zero test is λ-free", body: "The magnitude scales with λ (red-team probe). The λ-free property is the symmetry statement: ℛ = 0 ⟺ potential, at every λ." },
  { id: "F-0004", kind: "discovery", tone: "ok", title: "The meters decouple at high α", body: "Marginal ρ(EPR, ℛ) = 0.993 is α-confounding; within-level coupling collapses in the near-harmonic regime. Found by red-team stratification of our own headline number." },
  { id: "F-0005", kind: "calibration", tone: undefined, title: "Blotto is mixed, not pure harmonic", body: "α = 0.69 on the budget-3 game: zero-sum ≠ harmonic-pure. The realistic high-α anchor sits at ~0.7, not 1.0." },
  { id: "F-0006", kind: "discovery", tone: "ok", title: "Criticality escape + the supercritical wedge", body: "Potential games escape criticality at high λ (concentration kills the choice covariance — now proven a pure λ×scale fold, identity error 0.0); between the anchors a supercritical wedge opens.", playable: { href: "/phase", label: "phase map" } },
  { id: "F-0007", kind: "refuted repair", tone: "hot", title: "Our repair hypothesis, refuted by its own test", body: "H1 predicted the numerator of ℛ would keep tracking dissipation at high α. It doesn't. Response and dissipation are structurally distinct observables — the honest negative that made the instrument-scope table mandatory." },
  { id: "F-0008", kind: "certified null + retraction", tone: "warn", title: "Value-space blindness, and a retracted claim", body: "Price-value discretization is provably blind to loop irreversibility; an intermediate 'certified null' claim was retracted on adversarial review when the null class failed to bracket the data. The retraction is part of the record.", playable: { href: "/markets", label: "markets" } },
  { id: "F-0009", kind: "detection", tone: "ok", title: "The day-ahead market is a driven cycle", body: "Against a persistence-matched reversible null: pair-level detailed balance violated at ~1.1 nats/day (p < 0.01), concentrated in scarcity weeks; verified for null validity, FPR, seeds, bins, ties, multiple testing, and order-2 leakage.", playable: { href: "/markets", label: "markets" } },
  { id: "F-0010", kind: "failed criterion → finding", tone: "warn", title: "Universal collapse, λ-dependent sign", body: "The initial criterion (reversal in ≥3/4 conditions) failed 2/4 — and the failure is the finding: what's universal at α = 0.95 is decorrelation; anti-alignment is a λ-amplified second effect within ~2 null-SD per point." },
];

function LineChart({
  x,
  series,
  yLabel,
  height = 180,
  yZeroLine = false,
}: {
  x: number[];
  series: { ys: number[]; color: string; label: string; dashed?: boolean }[];
  yLabel: string;
  height?: number;
  yZeroLine?: boolean;
}) {
  const W = 460;
  const H = height;
  const all = series.flatMap((s) => s.ys);
  const lo = Math.min(...all, yZeroLine ? 0 : Infinity);
  const hi = Math.max(...all);
  const X = (v: number) => 44 + ((v - x[0]) / (x[x.length - 1] - x[0])) * (W - 60);
  const Y = (v: number) => H - 26 - ((v - lo) / (hi - lo || 1)) * (H - 46);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%" }} aria-label={yLabel}>
      {yZeroLine && <line x1={44} x2={W - 16} y1={Y(0)} y2={Y(0)} stroke="var(--text-faint)" strokeDasharray="3 4" strokeWidth="1" />}
      {series.map((s) => (
        <g key={s.label}>
          <polyline
            points={x.map((xv, i) => `${X(xv).toFixed(1)},${Y(s.ys[i]).toFixed(1)}`).join(" ")}
            fill="none"
            stroke={s.color}
            strokeWidth="1.8"
            strokeDasharray={s.dashed ? "5 4" : undefined}
          />
          {x.map((xv, i) => (
            <circle key={i} cx={X(xv)} cy={Y(s.ys[i])} r="2.4" fill={s.color} />
          ))}
        </g>
      ))}
      <text x={44} y={12} fontSize="9" fill="var(--text-faint)" fontFamily="var(--mono)">
        {yLabel}
      </text>
      {series.map((s, k) => (
        <text key={s.label} x={W - 16} y={16 + 12 * k} fontSize="9" textAnchor="end" fill={s.color} fontFamily="var(--mono)">
          {s.label}
        </text>
      ))}
      <text x={W - 16} y={H - 6} fontSize="9" textAnchor="end" fill="var(--text-faint)" fontFamily="var(--mono)">
        {x[0]} … {x[x.length - 1]}
      </text>
    </svg>
  );
}

export function FindingsGallery({ charts }: { charts: Charts }) {
  const [revealed, setRevealed] = useState(false);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.4rem", marginTop: "1.6rem" }}>
      <div className="feature-grid" style={{ margin: 0 }}>
        {FINDINGS.map((f) => (
          <div key={f.id} className="card">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: "0.6rem" }}>
              <span className="mono" style={{ color: "var(--accent-dim)", fontSize: "0.8rem" }}>{f.id}</span>
              <span className="badge" data-tone={f.tone}>{f.kind}</span>
            </div>
            <h3 style={{ fontSize: "0.98rem", margin: "0.5rem 0 0.35rem" }}>{f.title}</h3>
            <p style={{ color: "var(--text-dim)", fontSize: "0.85rem", margin: 0 }}>{f.body}</p>
            {f.playable && (
              <p style={{ margin: "0.5rem 0 0", fontSize: "0.8rem" }}>
                <Link href={f.playable.href}>▶ play with it — {f.playable.label}</Link>
              </p>
            )}
          </div>
        ))}
      </div>

      <div className="panel-cols">
        <PanelShell title="F-0004/F-0010 · you predict it" provenance="client">
          <p style={{ fontSize: "0.85rem", color: "var(--text-dim)", marginTop: 0 }}>
            Within each α level, how does the correlation between dissipation (EPR) and response
            asymmetry (ℛ) behave as games get more harmonic? Commit to a guess — rises, flat, or
            collapses — then reveal the measured curve.
          </p>
          {revealed ? (
            <>
              <LineChart
                x={charts.decoupling.levels}
                series={[{ ys: charts.decoupling.rho, color: "var(--red)", label: "within-level ρ(EPR, ℛ)" }]}
                yLabel="Spearman ρ within α level"
                yZeroLine
              />
              <p style={{ fontSize: "0.78rem", color: "var(--text-faint)", margin: "0.4rem 0 0" }}>
                It collapses — and at this seed set even reverses. The λ-sweep (F-0010) showed
                the collapse is the universal part; the negative sign grows with λ.
              </p>
            </>
          ) : (
            <button data-primary="true" onClick={() => setRevealed(true)}>
              I&apos;ve committed to a guess — reveal the measurement
            </button>
          )}
        </PanelShell>

        <PanelShell title="F-0006 · the supercritical frontier λ_c(α)" provenance="client">
          <LineChart
            x={charts.frontier.alphas}
            series={[{ ys: charts.frontier.lambda_c, color: "var(--amber)", label: "λ_c (median game)" }]}
            yLabel="λ_c — first crossing of ρ = 1"
          />
          <p style={{ fontSize: "0.78rem", color: "var(--text-faint)", margin: "0.4rem 0 0" }}>
            The wedge boundary, measured by verified-single-crossing bisection: more harmonic
            content ⇒ criticality arrives at lower λ. Below α ≈ 0.5 the median game never
            crosses by λ = 15.
          </p>
        </PanelShell>
      </div>

      <PanelShell title="thermo.estimators · data-side meters vs the exact one" provenance="client">
        <LineChart
          x={charts.estimators.alphas}
          series={[
            { ys: charts.estimators.exact, color: "var(--text)", label: "exact EPR" },
            { ys: charts.estimators.kld, color: "var(--accent)", label: "KLD estimate", dashed: true },
            { ys: charts.estimators.tur_ci_low, color: "var(--amber)", label: "TUR certified", dashed: true },
          ]}
          yLabel="nats / unit time across the α family"
          height={200}
        />
        <p style={{ fontSize: "0.78rem", color: "var(--text-faint)", margin: "0.4rem 0 0" }}>
          The KLD estimator sits on the exact meter (ρ = 1.0); the TUR certified bound stays
          below it everywhere, tightest near equilibrium — the calibration that made the
          market reading of F-0009 trustworthy.
        </p>
      </PanelShell>
    </div>
  );
}
