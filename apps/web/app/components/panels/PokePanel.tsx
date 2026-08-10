"use client";

import { useMemo, useState } from "react";
import { solveQRE, type TwoPlayerGame } from "../../../lib/qre";
import { Bars, NumberDial, PanelShell } from "./ui";

/* Doc 07's promised panel: poke-player selector, poke size, the two
   cross-readings side by side. The reciprocity meter's actual measurement
   procedure, live: nudge one player's incentives, re-equilibrate, read how
   much the OTHER player moved — both directions at once. */

const GAMES: Record<string, TwoPlayerGame & { note: string }> = {
  "coordination (potential)": {
    u1: [
      [2, 0, 0],
      [0, 2, 0],
      [0, 0, 2],
    ],
    u2: [
      [2, 0, 0],
      [0, 2, 0],
      [0, 0, 2],
    ],
    note: "potential game — the two cross-readings must AGREE (Onsager reciprocity)",
  },
  "rock–paper–scissors (harmonic)": {
    u1: [
      [0, -1, 1],
      [1, 0, -1],
      [-1, 1, 0],
    ],
    u2: [
      [0, 1, -1],
      [-1, 0, 1],
      [1, -1, 0],
    ],
    note: "harmonic game — the cross-readings DISAGREE; that asymmetry is ℛ",
  },
};

const ACTIONS = ["a1", "a2", "a3"];

export function PokePanel() {
  const [gameName, setGameName] = useState<keyof typeof GAMES>("coordination (potential)");
  const [action, setAction] = useState(0); // a: P1's poked action / P1's read action
  const [readAction, setReadAction] = useState(1); // b: P2's read action / P2's poked action
  const [size, setSize] = useState(0.3);
  const lam = 1.2;
  const game = GAMES[gameName];

  // Reciprocity compares chi_{(2,b),(1,a)} with chi_{(1,a),(2,b)}:
  // poke P1's a and read P2's b, versus poke P2's b and read P1's a.
  // (Comparing same-action entries can read zero even on harmonic games with
  // a player-exchange symmetry, e.g. RPS - the asymmetry lives off-diagonal.)
  const readings = useMemo(() => {
    const base = solveQRE(game, lam);
    const pokeRow: TwoPlayerGame = {
      u1: game.u1.map((row, i) => (i === action ? row.map((v) => v + size) : row)),
      u2: game.u2,
    };
    const pokeCol: TwoPlayerGame = {
      u1: game.u1,
      u2: game.u2.map((row) => row.map((v, j) => (j === readAction ? v + size : v))),
    };
    const p1 = solveQRE(pokeRow, lam);
    const p2 = solveQRE(pokeCol, lam);
    return {
      // poke player 1's a -> read player 2's whole mix shift
      read2: p1.sigma2.map((v, j) => v - base.sigma2[j]),
      // poke player 2's b -> read player 1's whole mix shift
      read1: p2.sigma1.map((v, i) => v - base.sigma1[i]),
    };
  }, [game, action, readAction, size]);

  const cross12 = readings.read2[readAction];
  const cross21 = readings.read1[action];
  const asym = Math.abs(cross12 - cross21);

  return (
    <PanelShell title="poke one player, read the other" provenance="client">
      <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap", marginBottom: "1rem" }}>
        <select value={gameName} onChange={(e) => setGameName(e.target.value as keyof typeof GAMES)} aria-label="game">
          {Object.keys(GAMES).map((g) => (
            <option key={g}>{g}</option>
          ))}
        </select>
        <select value={action} onChange={(e) => setAction(Number(e.target.value))} aria-label="P1 action a">
          {ACTIONS.map((a, i) => (
            <option key={a} value={i}>
              P1&apos;s {a}
            </option>
          ))}
        </select>
        <select value={readAction} onChange={(e) => setReadAction(Number(e.target.value))} aria-label="P2 action b">
          {ACTIONS.map((a, i) => (
            <option key={a} value={i}>
              P2&apos;s {a}
            </option>
          ))}
        </select>
        <div style={{ minWidth: 180 }}>
          <NumberDial value={size} setValue={setSize} min={0.05} max={1.5} step={0.05} label="poke size h" />
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.4rem" }}>
        <div>
          <div className="panel-label">
            poke P1&apos;s {ACTIONS[action]} → read P2&apos;s {ACTIONS[readAction]}:{" "}
            <span style={{ color: "var(--accent)" }}>{cross12.toFixed(4)}</span>
          </div>
          <Bars values={readings.read2} labels={ACTIONS} max={0.3} format={(v) => v.toFixed(4)} color="var(--accent)" />
        </div>
        <div>
          <div className="panel-label">
            poke P2&apos;s {ACTIONS[readAction]} → read P1&apos;s {ACTIONS[action]}:{" "}
            <span style={{ color: "var(--blue)" }}>{cross21.toFixed(4)}</span>
          </div>
          <Bars values={readings.read1} labels={ACTIONS} max={0.3} format={(v) => v.toFixed(4)} color="var(--blue)" />
        </div>
      </div>

      <div style={{ marginTop: "1rem", display: "flex", alignItems: "center", gap: "1rem", flexWrap: "wrap" }}>
        <span className="reading" data-tone={asym > 1e-3 ? "warn" : undefined} style={{ fontSize: "1.15rem" }}>
          |cross₁₂ − cross₂₁| = {asym.toExponential(2)}
        </span>
        <span className="badge" data-tone={asym > 1e-3 ? "warn" : "ok"}>
          {asym > 1e-3 ? "reciprocity broken — harmonic content present" : "reciprocal — consistent with a potential game"}
        </span>
      </div>
      <p style={{ fontSize: "0.78rem", color: "var(--text-faint)", marginTop: "0.6rem" }}>
        {game.note}. This works without knowing the payoffs: only pokes and observed shifts —
        that is why ℛ is estimable from real pass-through data.
      </p>
    </PanelShell>
  );
}
