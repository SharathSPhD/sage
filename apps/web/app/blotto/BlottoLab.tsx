"use client";

import { useEffect, useState } from "react";
import { PanelShell } from "../components/panels/ui";

/* A3 second half: the Blotto allocation lab. Budgets are the conjugate
   field; the QRE allocation mix is drawn as a simplex-style bar list. */

type BlottoRead = {
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

function AllocationBars({
  allocations,
  sigma,
  label,
}: {
  allocations: number[][];
  sigma: number[];
  label: string;
}) {
  const max = Math.max(...sigma);
  const order = sigma.map((_, i) => i).sort((a, b) => sigma[b] - sigma[a]);
  return (
    <div>
      <div className="panel-label">{label}</div>
      <div style={{ display: "flex", flexDirection: "column", gap: "2px", marginTop: "0.4rem" }}>
        {order.slice(0, 12).map((i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <span className="mono" style={{ fontSize: "0.72rem", color: "var(--text-faint)", width: "4.2rem" }}>
              ({allocations[i].join(",")})
            </span>
            <div style={{ flex: 1, background: "var(--bg-raised)", borderRadius: "3px", height: "0.9rem", overflow: "hidden" }}>
              <div
                style={{
                  width: `${(100 * sigma[i]) / max}%`,
                  height: "100%",
                  background: "var(--accent)",
                  opacity: 0.85,
                  transition: "width 0.3s",
                }}
              />
            </div>
            <span className="mono" style={{ fontSize: "0.72rem", color: "var(--text-dim)", width: "3.4rem", textAlign: "right" }}>
              {(100 * sigma[i]).toFixed(1)}%
            </span>
          </div>
        ))}
        {sigma.length > 12 && (
          <span style={{ fontSize: "0.7rem", color: "var(--text-faint)" }}>
            … {sigma.length - 12} more allocations
          </span>
        )}
      </div>
    </div>
  );
}

export function BlottoLab() {
  const [budgetA, setBudgetA] = useState(3);
  const [budgetB, setBudgetB] = useState(3);
  const [lam, setLam] = useState(1.5);
  const [read, setRead] = useState<BlottoRead | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const t = setTimeout(() => {
      setBusy(true);
      fetch("/api/v1/domains/blotto/read", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ budget_a: budgetA, budget_b: budgetB, lam }),
      })
        .then(async (r) => {
          if (!r.ok) throw new Error((await r.json()).detail ?? `HTTP ${r.status}`);
          return r.json();
        })
        .then((b) => {
          setRead(b);
          setError(null);
        })
        .catch((e) => setError(String(e.message ?? e)))
        .finally(() => setBusy(false));
    }, 250);
    return () => clearTimeout(t);
  }, [budgetA, budgetB, lam]);

  return (
    <div style={{ marginTop: "1.6rem", display: "flex", flexDirection: "column", gap: "1.4rem" }}>
      <PanelShell title="budgets and rationality" provenance="live">
        <div style={{ display: "flex", gap: "2rem", flexWrap: "wrap" }}>
          {[
            { label: `Colonel A budget: ${budgetA}`, value: budgetA, set: setBudgetA, min: 1, max: 8, step: 1 },
            { label: `Colonel B budget: ${budgetB}`, value: budgetB, set: setBudgetB, min: 1, max: 8, step: 1 },
            { label: `λ (payoff sensitivity): ${lam.toFixed(1)}`, value: lam, set: setLam, min: 0.2, max: 6, step: 0.1 },
          ].map((s) => (
            <label key={s.label} style={{ fontSize: "0.8rem", color: "var(--text-dim)", display: "flex", flexDirection: "column", gap: "0.3rem", minWidth: "14rem" }}>
              {s.label}
              <input
                type="range"
                min={s.min}
                max={s.max}
                step={s.step}
                value={s.value}
                onChange={(e) => s.set(Number(e.target.value))}
              />
            </label>
          ))}
        </div>
        {error && <p style={{ color: "var(--red)", fontSize: "0.8rem" }}>{error}</p>}
        {read && (
          <div style={{ display: "flex", gap: "2.2rem", flexWrap: "wrap", marginTop: "1rem", opacity: busy ? 0.5 : 1 }}>
            <div>
              <div className="panel-label">harmonic fraction α</div>
              <div className="reading">{read.alpha.toFixed(3)}</div>
            </div>
            <div>
              <div className="panel-label">reciprocity defect ℛ</div>
              <div className="reading" data-tone={read.r > 0.3 ? "warn" : "neutral"}>
                {read.r.toFixed(3)}
              </div>
            </div>
            <div>
              <div className="panel-label">dissipation (nats/time)</div>
              <div className="reading" data-tone={read.epr && read.epr > 1e-6 ? "warn" : "ok"}>
                {read.epr === null ? "—" : read.epr.toExponential(2)}
              </div>
            </div>
          </div>
        )}
        {read?.warnings?.length ? (
          <p style={{ fontSize: "0.75rem", color: "var(--amber)", marginTop: "0.5rem" }}>
            {read.warnings.join(" · ")}
          </p>
        ) : null}
      </PanelShell>

      {read && (
        <div className="panel-cols">
          <PanelShell title="Colonel A — QRE allocation mix" provenance="live">
            <AllocationBars allocations={read.allocations_a} sigma={read.sigma_a} label={`over ${read.sigma_a.length} allocations (f1, f2, f3)`} />
          </PanelShell>
          <PanelShell title="Colonel B — QRE allocation mix" provenance="live">
            <AllocationBars allocations={read.allocations_b} sigma={read.sigma_b} label={`over ${read.sigma_b.length} allocations (f1, f2, f3)`} />
          </PanelShell>
        </div>
      )}

      <p style={{ fontSize: "0.85rem", color: "var(--text-dim)", maxWidth: "46rem" }}>
        Things to try: equal budgets at high λ (the mix stays spread — there is no pure
        equilibrium to concentrate on); a one-troop advantage (watch the favourite&apos;s mass
        shift toward covering, the underdog&apos;s toward gambling on fewer fields); and the α
        reading as asymmetry grows — dominance relations add gradient structure to a game that
        pure symmetry keeps circulating.
      </p>
    </div>
  );
}
