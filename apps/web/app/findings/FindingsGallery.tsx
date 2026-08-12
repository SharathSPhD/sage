"use client";

import { useState } from "react";
import Link from "next/link";
import { PanelShell } from "../components/panels/ui";

export interface Charts {
  decoupling: { levels: number[]; rho: number[] };
  frontier: { alphas: number[]; lambda_c: number[] };
  estimators: { alphas: number[]; exact: number[]; kld: number[]; tur_ci_low: number[] };
  crossover_signs: { lambdas: number[]; corr_a095: number[]; m4_a095: number };
  quench: {
    alphas: number[];
    excess_log10: number[];
    housekeeping: number[];
    burn_rate_a095: number;
  };
  passthrough: {
    chi: number[][];
    R: number;
    R_ci: number[];
    asym_ci: number[];
    n_stores: number;
    n_store_weeks: number;
    edgeworth_detections: number;
    edgeworth_tested: number;
  };
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
  { id: "F-0011", kind: "empirical read", tone: "ok", title: "The reciprocity meter's first real-data read: ℛ ≈ 0.001", body: "Cross-brand wholesale-cost pass-through (Campbell ↔ Progresso, Dominick's scanner panel, 86 stores): the prediction stated in config before the run — one retailer pricing both brands must respond symmetrically — confirmed. Own pass-through 1.07/0.97, asymmetry CI covering zero. A χ row-ordering bug caught pre-review is on the record." },
  { id: "F-0012", kind: "discovery · mechanism open", tone: "ok", title: "The driving cost inverts: pay per change vs pay rent", body: "Quenching λ across the α family: potential games pay only for change (excess ∝ 1/steps, quasi-static driving is free) and nothing to exist; near-harmonic games pay almost nothing for change but burn constant housekeeping rent just to hold their steady state. The first-pass mechanism was refuted by the red-team's own probe — the collapse is real, its cause is open." },
  { id: "F-0016", kind: "withheld → certified", tone: "warn", title: "The estimator that had to earn it", body: "The data-facing quench meter was refused by adversarial review: its natural self-check (the fluctuation theorem) proved structurally insufficient — a 45% bias hid behind an IFT of 1.01. The chase refuted two hypotheses, found a real missing-window bug, vindicated the statistics (20/20 calibration), replaced the check with a physical relaxation gate — and a fresh review granted certification with every escalation on the record." },
  { id: "F-0019", kind: "refusal with a mechanism", tone: "warn", title: "The number survives a month of data; the verdict doesn't", body: "Can a month of market data (~30 trajectories) be quoted against a certified floor of 200? No — but not for the expected reason. Interval coverage holds all the way down to n = 20; what fails is the instrument's own decision machinery, which needs roughly ten times more data than the estimate does. A permutation of the trajectories — which cannot change any physical property — was flipping the verdict, and that localised the blame to one implementation choice rather than to the physics." },
  { id: "F-0020", kind: "fixed → partly retracted", tone: "hot", title: "Two failure modes that looked like one", body: "The suspect was an arbitrary 4-way split of the trajectories, used to estimate an error bar. Replacing it with order-invariant estimators kills the instability completely — zero flips at every n, and on real price data the flip rate falls 0.214 → 0.050 with one month collapsing 17/20 → 0/20. It even flips a real month from refused to admitted while leaving the estimate identical to six decimals. But the small-n floor does not move, which retracts half of what the previous finding predicted: the relaxation-time estimate itself varies 35–40% across seeds at n = 30, so an error bar that is accurate must report that variance — and must therefore refuse holds that are genuinely settled. Stability was a bug; accuracy is a limit. The same permutation trick then exposed a second, independent order-dependence hiding in the interval bootstrap, where the obstruction is structural: a yes/no flag thresholded on a Monte-Carlo interval sitting at its threshold cannot be stabilised by any choice of seed." },
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

function PassthroughMatrix({ pt }: { pt: Charts["passthrough"] }) {
  const [revealed, setRevealed] = useState(false);
  const cell = (v: number, own: boolean) => (
    <td
      className="mono"
      style={{
        padding: "0.55rem 0.9rem",
        textAlign: "right",
        fontSize: "0.95rem",
        color: own ? "var(--text)" : "var(--accent)",
        borderTop: "1px solid var(--border)",
      }}
    >
      {revealed ? v.toFixed(4) : "?"}
    </td>
  );
  return (
    <PanelShell title="F-0011 · the empirical pass-through matrix" provenance="artifact">
      <p style={{ fontSize: "0.85rem", color: "var(--text-dim)", marginTop: 0 }}>
        Poke Campbell&apos;s wholesale cost, read both shelf prices; poke Progresso&apos;s, read
        both again. That is the poke panel&apos;s procedure run on {pt.n_store_weeks.toLocaleString()}{" "}
        real store-weeks. One retailer prices both brands — so before revealing: should the two{" "}
        <em>cross</em>-readings agree (a landscape) or disagree (a whirlpool)?
      </p>
      <div style={{ overflowX: "auto" }}>
        <table style={{ borderCollapse: "collapse", margin: "0.3rem 0" }}>
          <thead>
            <tr>
              {["∂ log p / ∂ log c", "Campbell cost", "Progresso cost"].map((h) => (
                <th key={h} className="mono" style={{ padding: "0.35rem 0.9rem", fontSize: "0.72rem", color: "var(--text-faint)", textAlign: "right", fontWeight: 400 }}>
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="mono" style={{ padding: "0.55rem 0.9rem", fontSize: "0.78rem", color: "var(--text-faint)", borderTop: "1px solid var(--border)" }}>Campbell price</td>
              {cell(pt.chi[0][0], true)}
              {cell(pt.chi[0][1], false)}
            </tr>
            <tr>
              <td className="mono" style={{ padding: "0.55rem 0.9rem", fontSize: "0.78rem", color: "var(--text-faint)", borderTop: "1px solid var(--border)" }}>Progresso price</td>
              {cell(pt.chi[1][0], false)}
              {cell(pt.chi[1][1], true)}
            </tr>
          </tbody>
        </table>
      </div>
      {revealed ? (
        <p style={{ fontSize: "0.78rem", color: "var(--text-faint)", margin: "0.4rem 0 0" }}>
          Own pass-through is textbook (≈ 1); the cross terms are tiny and their{" "}
          <em>difference</em> has CI [{pt.asym_ci[0]}, {pt.asym_ci[1]}] ∋ 0 — so ℛ ={" "}
          <strong>{pt.R}</strong> [{pt.R_ci[0]}, {pt.R_ci[1]}], cluster-bootstrapped over{" "}
          {pt.n_stores} stores. A landscape, exactly where a single-retailer category objective
          demands one. Companion scan: {pt.edgeworth_detections}/{pt.edgeworth_tested} stores
          show Edgeworth-cycle irreversibility in weekly category indices — at-null.
        </p>
      ) : (
        <button data-primary="true" onClick={() => setRevealed(true)}>
          I&apos;ve committed to a guess — reveal the matrix
        </button>
      )}
    </PanelShell>
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

      <PassthroughMatrix pt={charts.passthrough} />

      <div className="panel-cols">
        <PanelShell title="F-0012 · pay per change: excess dissipation vs α" provenance="artifact">
          <LineChart
            x={charts.quench.alphas}
            series={[{ ys: charts.quench.excess_log10, color: "var(--accent)", label: "log₁₀ excess ⟨Y⟩" }]}
            yLabel="log₁₀ nats per ramp (λ 0.5 → 3.0)"
          />
          <p style={{ fontSize: "0.78rem", color: "var(--text-faint)", margin: "0.4rem 0 0" }}>
            The cost of *changing* λ collapses three orders of magnitude toward the harmonic
            end — and honesty note: the first-pass explanation for why was refuted by the
            red-team&apos;s own probe. The collapse is measured; its mechanism is an open chase
            item, on the record.
          </p>
        </PanelShell>

        <PanelShell title="F-0012 · pay rent: housekeeping vs α" provenance="artifact">
          <LineChart
            x={charts.quench.alphas}
            series={[{ ys: charts.quench.housekeeping, color: "var(--amber)", label: "∫σ_hk dt (nats)" }]}
            yLabel="housekeeping over the same ramp"
            yZeroLine
          />
          <p style={{ fontSize: "0.78rem", color: "var(--text-faint)", margin: "0.4rem 0 0" }}>
            The fuel burned just to *hold* the steady state: exactly zero for potential games,
            {" "}{charts.quench.housekeeping[charts.quench.housekeeping.length - 1].toFixed(1)} nats
            at α = 0.95 — at a constant {charts.quench.burn_rate_a095} nats per unit time,
            charged whether you drive or not. Quasi-static driving is free only on a landscape.
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
