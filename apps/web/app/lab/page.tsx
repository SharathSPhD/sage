"use client";

import { useCallback, useEffect, useState } from "react";

// The Lab is a research instrument: pick a game, slide λ, every meter updates
// from the live API (heavy compute never runs in the browser).
const API = process.env.NEXT_PUBLIC_SAGE_API_BASE ?? "http://150.136.84.2";

interface Readings {
  sigma?: number[][];
  reciprocity_defect?: number;
  distance_to_criticality?: number;
  rho_sb?: number;
  epr?: number;
  detailed_balance?: boolean;
  warnings: string[];
}

export default function Lab() {
  const [examples, setExamples] = useState<Record<string, { payoffs: unknown }>>({});
  const [selected, setSelected] = useState<string>("");
  const [lam, setLam] = useState(1.2);
  const [readings, setReadings] = useState<Readings | null>(null);
  const [error, setError] = useState<string>("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    fetch(`${API}/v1/examples`)
      .then((r) => r.json())
      .then((body) => {
        setExamples(body);
        setSelected(Object.keys(body)[0] ?? "");
      })
      .catch(() => setError(`API unreachable at ${API} — start it with: uvicorn sage_api.main:app`));
  }, []);

  const measure = useCallback(async () => {
    if (!selected || !examples[selected]) return;
    setBusy(true);
    setError("");
    try {
      const payload = { payoffs: examples[selected].payoffs, lam };
      const [solve, response, dynamics] = await Promise.all(
        ["/v1/solve/qre", "/v1/response", "/v1/dynamics/stationary"].map((ep) =>
          fetch(`${API}${ep}`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify(payload),
          }).then((r) => r.json()),
        ),
      );
      setReadings({
        sigma: solve.sigma,
        reciprocity_defect: response.reciprocity_defect,
        distance_to_criticality: response.distance_to_criticality,
        rho_sb: response.rho_sb,
        epr: dynamics.epr,
        detailed_balance: dynamics.detailed_balance,
        warnings: [...(response.warnings ?? []), ...(dynamics.warnings ?? [])],
      });
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }, [selected, lam, examples]);

  useEffect(() => {
    void measure();
  }, [measure]);

  const meter = (label: string, value: string, hint: string) => (
    <div style={{ border: "1px solid #ddd", borderRadius: 8, padding: "0.8rem 1rem", minWidth: "11rem" }}>
      <div style={{ fontSize: "0.8rem", color: "#666" }}>{label}</div>
      <div style={{ fontSize: "1.4rem", fontWeight: 700 }}>{value}</div>
      <div style={{ fontSize: "0.72rem", color: "#999" }}>{hint}</div>
    </div>
  );

  return (
    <div>
      <h1>Lab</h1>
      <div style={{ display: "flex", gap: "1rem", alignItems: "center", flexWrap: "wrap" }}>
        <select value={selected} onChange={(e) => setSelected(e.target.value)}>
          {Object.keys(examples).map((k) => (
            <option key={k}>{k}</option>
          ))}
        </select>
        <label>
          λ = {lam.toFixed(2)}{" "}
          <input
            type="range"
            min={0.05}
            max={5}
            step={0.05}
            value={lam}
            onChange={(e) => setLam(Number(e.target.value))}
          />
        </label>
        {busy && <span style={{ color: "#999" }}>measuring…</span>}
      </div>

      {error && <p style={{ color: "#b91c1c" }}>{error}</p>}

      {readings && (
        <>
          <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap", marginTop: "1.2rem" }}>
            {meter(
              "reciprocity defect ℛ",
              readings.reciprocity_defect! < 1e-10 ? "0 (potential)" : readings.reciprocity_defect!.toFixed(3),
              "0 ⟺ potential game, at every λ",
            )}
            {meter(
              "entropy production",
              readings.detailed_balance ? "0 (equilibrium)" : readings.epr!.toExponential(2),
              "0 ⟺ detailed balance",
            )}
            {meter(
              "distance to criticality",
              readings.distance_to_criticality!.toFixed(3),
              `ρ(SB) = ${readings.rho_sb!.toFixed(3)}; crossing 1 = bifurcation`,
            )}
          </div>
          {readings.warnings.length > 0 && (
            <p style={{ color: "#92400e", fontSize: "0.85rem" }}>⚠ {readings.warnings.join(" · ")}</p>
          )}
          <h3 style={{ marginTop: "1.4rem" }}>Equilibrium play σ*</h3>
          {readings.sigma!.map((s, i) => (
            <div key={i} style={{ display: "flex", gap: 4, alignItems: "center", marginBottom: 6 }}>
              <span style={{ width: "4.5rem", fontSize: "0.8rem" }}>player {i + 1}</span>
              {s.map((p, j) => (
                <div
                  key={j}
                  title={`action ${j + 1}: ${(p * 100).toFixed(1)}%`}
                  style={{
                    height: "1.1rem",
                    width: `${Math.max(p * 260, 2)}px`,
                    background: ["#0f3d3e", "#2d6a4f", "#74a892", "#c7d9b7"][j % 4],
                    borderRadius: 3,
                  }}
                />
              ))}
            </div>
          ))}
          <p style={{ fontSize: "0.8rem", color: "#777" }}>
            Reproduce in Python:{" "}
            <code>
              strataq.reciprocity_defect(game, strataq.logit_qre(game, {lam.toFixed(2)}))
            </code>
          </p>
        </>
      )}
    </div>
  );
}
