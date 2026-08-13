"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { glauber, mixGame, responseAsymmetry } from "../../../lib/demos/gametheory";
import { LANDMARKS, QUADRANTS, QUADRANT_III, R_BANDS } from "../../../lib/demos/landmarks";
import { ProvenanceCard, Readout, Widget } from "../components/chrome";
import { useDragNumber } from "../components/drag";
import { useReducedMotion } from "../components/motion";

const W = 620;
const H = 430;
const GUT_X = 108;
const LOG_X0 = 140;
const LOG_X1 = 566;
const GUT_Y = 372;
const LOG_Y0 = 344;
const LOG_Y1 = 34;
const LO = -6;
const HI = Math.log10(3);
const SPAN = HI - LO;

const X = (r: number) => {
  if (!Number.isFinite(r) || r < 1e-12) return GUT_X;
  const l = Math.min(HI, Math.max(LO, Math.log10(r)));
  return LOG_X0 + (LOG_X1 - LOG_X0) * ((l - LO) / SPAN);
};
const Y = (e: number) => {
  if (!Number.isFinite(e) || e < 1e-12) return GUT_Y;
  const l = Math.min(HI, Math.max(LO, Math.log10(e)));
  return LOG_Y0 - (LOG_Y0 - LOG_Y1) * ((l - LO) / SPAN);
};

const LABELS = ["R", "P", "S"];
const ZERO_DELTA = [
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
];

export function Plane() {
  const [alpha, setAlpha] = useState(0.6);
  const [lam, setLam] = useState(1.2);
  const [delta, setDelta] = useState<number[][]>(ZERO_DELTA.map((r) => [...r]));
  const [open, setOpen] = useState<string | null>(null);
  const reduced = useReducedMotion();

  const game = useMemo(() => {
    const [base1, base2] = mixGame(alpha);
    const u1 = base1.map((row, i) => row.map((v, j) => v + delta[i][j]));
    return { u1, u2: base2 };
  }, [alpha, delta]);

  const reading = useMemo(() => {
    const chain = glauber(game.u1, game.u2, lam);
    const resp = responseAsymmetry(game.u1, game.u2, lam);
    return { ep: chain.ep, R: resp.R, converged: resp.converged, rho: resp.rho };
  }, [game, lam]);

  /** The locus a one-dimensional theory would predict: the pure alpha family. */
  const locus = useMemo(() => {
    const pts: { R: number; ep: number; a: number }[] = [];
    for (let k = 0; k <= 12; k++) {
      const a = k / 12;
      const [u1, u2] = mixGame(a);
      pts.push({ a, R: responseAsymmetry(u1, u2, lam).R, ep: glauber(u1, u2, lam).ep });
    }
    return pts;
  }, [lam]);

  /** How far off that locus the user's game now sits, in dissipation. */
  const offLocus = useMemo(() => {
    if (!Number.isFinite(reading.R) || reading.R < 1e-9 || reading.ep < 1e-12) return null;
    let lo = 0;
    let hi = locus.length - 1;
    if (reading.R <= locus[0].R || reading.R >= locus[hi].R) return null;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (locus[mid].R < reading.R) lo = mid;
      else hi = mid;
    }
    const t = (reading.R - locus[lo].R) / (locus[hi].R - locus[lo].R || 1);
    const predicted = locus[lo].ep + t * (locus[hi].ep - locus[lo].ep);
    if (predicted <= 0) return null;
    return { predicted, ratio: reading.ep / predicted };
  }, [reading, locus]);

  const [trail, setTrail] = useState<{ x: number; y: number }[]>([]);
  const lastPos = useRef({ x: 0, y: 0 });
  useEffect(() => {
    if (reduced) return;
    const p = { x: X(reading.R), y: Y(reading.ep) };
    if (Math.abs(p.x - lastPos.current.x) < 0.6 && Math.abs(p.y - lastPos.current.y) < 0.6) return;
    lastPos.current = p;
    setTrail((prev) => [...prev.slice(-5), p]);
    const timer = setTimeout(() => setTrail([]), 600);
    return () => clearTimeout(timer);
  }, [reading.R, reading.ep, reduced]);

  const dotDrag = useDragNumber({
    value: alpha,
    min: 0,
    max: 1,
    step: 0.01,
    onChange: setAlpha,
    axis: "x",
    travelPx: 380,
    label: "Your game: drag right toward rock–paper–scissors, left toward an exact potential game",
    valueText: (v) =>
      `${v === 0 ? "exact potential game" : `${(v * 100).toFixed(0)} percent toward rock\u2013paper\u2013scissors`}; response asymmetry ${reading.R < 1e-12 ? "zero" : reading.R.toFixed(3)}, dissipation ${reading.ep < 1e-12 ? "zero" : reading.ep.toFixed(3)} nats per step`,
  });

  const bothZero = reading.R < 1e-12 && reading.ep < 1e-12;
  const quadrant = bothZero ? "I" : "IV";
  const spec = QUADRANTS.find((q) => q.id === quadrant)!;
  const edited = delta.some((row) => row.some((v) => v !== 0));
  const openLandmark = LANDMARKS.find((l) => l.id === open) ?? null;

  return (
    <>
      <Widget
        hook="Put your game on the plane"
        lede={
          <p>
            Two coordinates, read from two entirely different objects: response asymmetry ℛ is a derivative at one
            equilibrium, dissipation σ_EP is a flux functional over the whole profile space. Drag your game left and
            right and watch both move.
          </p>
        }
        consequence={
          bothZero ? (
            <>
              At the potential end both coordinates are zero at machine precision, and the dot sits in the corner where
              Sioux Falls sits — an exact potential game read from real road-network data.
            </>
          ) : offLocus && Math.abs(offLocus.ratio - 1) < 0.02 ? (
            <>
              You are sitting on the faint locus, because every game in this one-parameter family does. Now drag one of
              the payoff cells on the right: the dot leaves the locus, and a theory with a single axis has nothing left
              to say about where it went.
            </>
          ) : offLocus ? (
            <>
              Your game dissipates {offLocus.ratio.toFixed(2)}× what the faint locus predicts at your ℛ. On a
              one-dimensional theory that number would be 1.00 for every game there is.
            </>
          ) : (
            <>
              Your dot is off the faint locus. That locus is what a one-dimensional theory predicts — that knowing one
              coordinate fixes the other.
            </>
          )
        }
        maths={
          <>
            <p>
              ℛ = ‖χ − χᵀ‖<sub>F</sub> / ‖χ + χᵀ‖<sub>F</sub> with χ = (I − SB)⁻¹S, S the block-diagonal softmax
              Jacobian at the logit fixed point and B the block-off-diagonal payoff cross-derivative. σ_EP is
              Schnakenberg entropy production of the Glauber chain on the nine joint profiles, in nats per step. Both are
              computed in this page from your dragged payoffs.
            </p>
            <p>
              The implementation here reproduces the committed calibration exactly. At λ = 1.2 it returns ℛ = 0.6928203
              for rock–paper–scissors and ℛ = 0.4346571 for the five-action version, against{" "}
              <code>benchmarks/results/reciprocity_harmonic.json</code>{" "}
              <code>R_rps_3 = 0.6928203230275507</code>, <code>R_rps_5 = 0.434657051228945</code>. On potential games it
              returns ~5e-17 against <code>reciprocity_potential.json</code> <code>max_R = 8.93e-17</code>.
            </p>
            <p>
              ℛ&apos;s magnitude scales with λ (finding F-0002) — only the zero versus non-zero verdict is λ-free. Every
              landmark on this plane was read at λ = 1.2, so comparing levels at any other λ is not meaningful, and the
              figure says so when you move the λ slider.
            </p>
            <p>
              The current reading is ℛ = <code>{Number.isFinite(reading.R) ? reading.R.toExponential(6) : "not available"}</code>,
              σ_EP = <code>{reading.ep.toExponential(6)}</code>, solver residual bound met:{" "}
              <code>{String(reading.converged)}</code>, ‖SB‖<sub>∞</sub> = <code>{reading.rho.toFixed(3)}</code> (values
              at or above 1 mean the resolvent is near-singular and the level should be read as direction only — the
              same caveat the phase map carries in <code>benchmarks/results/phase_map.json</code>).
            </p>
          </>
        }
      >
        <div className="plane-layout">
          <div className="plane-figure">
            <svg viewBox={`0 0 ${W} ${H}`} className="demo-svg" role="img" aria-label={`The irreversibility plane. Your game reads response asymmetry ${Number.isFinite(reading.R) ? reading.R.toPrecision(3) : "unavailable"} and entropy production ${reading.ep.toPrecision(3)} nats per step, in quadrant ${quadrant}, ${spec.name}.`}>
              <defs>
                <pattern id="empty-hatch" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                  <line x1="0" y1="0" x2="0" y2="8" stroke="var(--q-stalled-text)" strokeWidth="1.4" opacity="0.7" />
                </pattern>
              </defs>

              <rect x={72} y={LOG_Y1 - 12} width={LOG_X1 - 72 + 40} height={GUT_Y + 22 - LOG_Y1 + 12} fill="var(--surface-2)" stroke="var(--border)" />
              <rect x={72} y={LOG_Y1 - 12} width={LOG_X0 - 78} height={LOG_Y0 + 12 - LOG_Y1} fill="var(--q-driven)" opacity={0.07} />
              <rect x={LOG_X0 - 6} y={LOG_Y0 + 6} width={LOG_X1 - LOG_X0 + 46} height={GUT_Y + 22 - LOG_Y0 - 6} fill="url(#empty-hatch)" opacity={0.5} />
              <rect x={LOG_X0 - 6} y={LOG_Y1 - 12} width={LOG_X1 - LOG_X0 + 46} height={LOG_Y0 + 12 - LOG_Y1} fill="var(--q-whirlpool)" opacity={0.06} />
              <rect x={72} y={LOG_Y0 + 6} width={LOG_X0 - 78} height={GUT_Y + 22 - LOG_Y0 - 6} fill="var(--q-landscape)" opacity={0.1} />

              <line x1={LOG_X0 - 6} y1={LOG_Y1 - 12} x2={LOG_X0 - 6} y2={GUT_Y + 22} stroke="var(--text-3)" strokeWidth={1.5} />
              <line x1={72} y1={LOG_Y0 + 6} x2={LOG_X1 + 40} y2={LOG_Y0 + 6} stroke="var(--text-3)" strokeWidth={1.5} />
              <line x1={X(R_BANDS.zero)} y1={LOG_Y1 - 12} x2={X(R_BANDS.zero)} y2={LOG_Y0 + 6} stroke="var(--text-3)" strokeWidth={1} strokeDasharray="5 4" />
              <text x={X(R_BANDS.zero) + 5} y={LOG_Y1 + 8} fontSize={11} fill="var(--text-3)">
                ℛ = 0.02
              </text>

              <text x={90} y={GUT_Y + 36} fontSize={10.5} fill="var(--text-3)" fontFamily="var(--mono)" textAnchor="middle">
                exactly 0
              </text>
              {[-5, -4, -3, -2, -1, 0].map((e) => (
                <text key={e} x={X(Math.pow(10, e))} y={GUT_Y + 36} fontSize={10.5} fill="var(--text-3)" fontFamily="var(--mono)" textAnchor="middle">
                  1e{e}
                </text>
              ))}
              <text x={(LOG_X0 + LOG_X1) / 2} y={H - 6} fontSize={12} fill="var(--text-2)" textAnchor="middle">
                response asymmetry ℛ →
              </text>
              <text x={16} y={(LOG_Y1 + LOG_Y0) / 2} fontSize={12} fill="var(--text-2)" textAnchor="middle" transform={`rotate(-90 16 ${(LOG_Y1 + LOG_Y0) / 2})`}>
                dissipation σ_EP →
              </text>
              <text x={66} y={GUT_Y + 4} fontSize={10.5} fill="var(--text-3)" fontFamily="var(--mono)" textAnchor="end">
                0
              </text>
              {[-4, -2, 0].map((e) => (
                <text key={e} x={66} y={Y(Math.pow(10, e)) + 3} fontSize={10.5} fill="var(--text-3)" fontFamily="var(--mono)" textAnchor="end">
                  1e{e}
                </text>
              ))}

              {QUADRANTS.map((q) => {
                const pos: Record<string, [number, number, "start" | "end"]> = {
                  I: [80, GUT_Y + 10, "start"],
                  II: [80, LOG_Y1 + 24, "start"],
                  III: [LOG_X1 + 34, GUT_Y + 14, "end"],
                  IV: [LOG_X1 + 34, LOG_Y1 + 24, "end"],
                };
                return (
                  <text key={q.id} x={pos[q.id][0]} y={pos[q.id][1]} fontSize={21} fontWeight={700} fill={q.textVar} opacity={0.45} textAnchor={pos[q.id][2]}>
                    {q.id}
                  </text>
                );
              })}
              <text x={LOG_X0 + 4} y={GUT_Y + 16} fontSize={11} fill="var(--q-stalled-text)" fontStyle="italic">
                no real system measured here yet
              </text>

              <path
                d={locus.map((p, i) => `${i ? "L" : "M"}${X(p.R).toFixed(1)},${Y(p.ep).toFixed(1)}`).join("")}
                fill="none"
                stroke="var(--text-3)"
                strokeWidth={1.4}
                strokeDasharray="3 5"
                opacity={0.55}
              />
              <text x={X(locus[2].R) + 10} y={Y(locus[2].ep) + 16} fontSize={10.5} fill="var(--text-3)">
                what one axis would predict
              </text>

              {/* landmarks with one coordinate read are lines, not points */}
              <g>
                <line x1={LOG_X0 - 6} y1={92} x2={LOG_X1 + 40} y2={92} stroke="var(--q-driven)" strokeWidth={2} strokeDasharray="9 5" />
                <LandmarkHit id="caiso" cx={LOG_X0 + 60} cy={92} onOpen={setOpen} active={open === "caiso"} label="CAISO day-ahead" />
                <text x={LOG_X0 + 74} y={88} fontSize={11} fill="var(--q-driven-text)">
                  CAISO day-ahead · σ_EP positive, ℛ not read
                </text>
              </g>
              <g>
                <line x1={X(0.11809781461174602)} y1={LOG_Y1 - 6} x2={X(0.11809781461174602)} y2={LOG_Y0 + 6} stroke="var(--q-whirlpool)" strokeWidth={2} strokeDasharray="9 5" />
                <LandmarkHit id="blotto" cx={X(0.11809781461174602)} cy={LOG_Y0 - 26} onOpen={setOpen} active={open === "blotto"} label="Colonel Blotto" />
                <text x={X(0.11809781461174602) + 12} y={LOG_Y0 - 22} fontSize={11} fill="var(--q-whirlpool-text)">
                  Blotto · ℛ = 0.118
                </text>
              </g>

              {/* full readings */}
              <g>
                <line x1={X(4.778562071941612e-5)} y1={GUT_Y} x2={X(0.004993334763269472)} y2={GUT_Y} stroke="var(--q-landscape)" strokeWidth={2} />
                <LandmarkHit id="dominicks" cx={X(0.0011224257982100365)} cy={GUT_Y} onOpen={setOpen} active={open === "dominicks"} label="Dominick's retail panel" />
                <text x={X(0.0011224257982100365)} y={GUT_Y - 14} fontSize={11} fill="var(--q-landscape-text)" textAnchor="middle">
                  Dominick&apos;s · ℛ = 0.00112
                </text>
              </g>
              <g>
                <LandmarkHit id="sioux-falls" cx={GUT_X} cy={GUT_Y} onOpen={setOpen} active={open === "sioux-falls"} label="Sioux Falls road network" />
                <text x={GUT_X - 2} y={GUT_Y - 14} fontSize={11} fill="var(--q-landscape-text)" textAnchor="middle">
                  Sioux Falls
                </text>
              </g>
              <g>
                <LandmarkHit id="rps" cx={X(0.6928203230275507)} cy={Y(0.7839132980741325)} onOpen={setOpen} active={open === "rps"} label="Rock–paper–scissors" />
                <text x={X(0.6928203230275507) - 13} y={Y(0.7839132980741325) + 4} fontSize={11} fill="var(--q-whirlpool-text)" textAnchor="end">
                  rock–paper–scissors
                </text>
              </g>

              {trail.map((p, i) => (
                <circle key={i} cx={p.x} cy={p.y} r={7} fill="var(--accent)" opacity={0.25 * ((i + 1) / (trail.length + 1))} />
              ))}
              <g {...dotDrag.handleProps} data-dragging={dotDrag.dragging ? "true" : undefined} className="plane-dot">
                <circle className="morph" cx={X(reading.R)} cy={Y(reading.ep)} r={16} fill="var(--accent)" opacity={0.18} />
                <circle className="morph" cx={X(reading.R)} cy={Y(reading.ep)} r={8.5} fill="var(--accent-strong)" stroke="var(--surface)" strokeWidth={2} />
                <text className="morph" x={X(reading.R)} y={Y(reading.ep) + 30} fontSize={11.5} fontWeight={650} fill="var(--accent-strong)" textAnchor="middle">
                  your game
                </text>
              </g>
            </svg>
            <div className="plane-key" aria-hidden>
              {(["II", "IV", "I", "III"] as const).map((id) => {
                const q = QUADRANTS.find((x) => x.id === id)!;
                return (
                  <div key={id} style={{ borderLeftColor: q.colorVar }}>
                    <strong style={{ color: q.textVar }}>
                      {q.id}. {q.name}
                    </strong>{" "}
                    <span>{q.headline}</span>
                  </div>
                );
              })}
            </div>
            <p className="plane-legend">
              The dashed vertical at ℛ = 0.02 is the band an estimate has to clear to count as non-zero; the solid
              divisions are the exact zeros, where the theorem puts them. A <strong>dot</strong> is a system with both coordinates read; a <strong>dashed line</strong> is one where
              only one coordinate has been read, drawn across everything the other could be. Real-data landmarks are
              placed by their verdict against their own null and carry their own units — only the in-browser games share
              the chain&apos;s nats-per-step scale. Every landmark is clickable for its artifact and its caveats.
            </p>
          </div>

          <div className="plane-controls">
            <div className="demo-readouts demo-readouts-col">
              <Readout label="response asymmetry ℛ" value={Number.isFinite(reading.R) ? (reading.R < 1e-12 ? "0" : reading.R.toFixed(4)) : "withheld"} hand />
              <Readout label="dissipation σ_EP" value={reading.ep < 1e-12 ? "0" : reading.ep.toFixed(4)} unit="nats/step" hand />
              <div className="demo-readout">
                <div className="panel-label">quadrant</div>
                <p className="verdict-line" data-tone={quadrant === "I" ? "landscape" : "whirlpool"} aria-live="polite">
                  <span aria-hidden>{quadrant === "I" ? "◇" : "◉"}</span> {quadrant}. {spec.name}
                </p>
                <p className="verdict-note">{spec.consequence}</p>
              </div>
            </div>

            {!reading.converged ? (
              <p className="callout" data-tone="warn">
                The fixed-point solver did not settle on this game, so the reading is withheld rather than quoted from
                the last iterate.
              </p>
            ) : null}
            {Math.abs(lam - 1.2) > 1e-9 ? (
              <p className="callout" data-tone="warn">
                λ = {lam.toFixed(2)}. Every landmark here was read at λ = 1.2, and ℛ&apos;s magnitude scales with λ
                (F-0002), so your dot&apos;s <em>level</em> is no longer comparable with theirs. Its zero-versus-non-zero
                verdict still is.
              </p>
            ) : null}

            <div className="demo-control">
              <label htmlFor="plane-lam" className="panel-label">
                Precision λ = <span style={{ color: "var(--accent-strong)" }}>{lam.toFixed(2)}</span>
              </label>
              <input id="plane-lam" type="range" min={0.3} max={4} step={0.05} value={lam} onChange={(e) => setLam(Number(e.target.value))} />
            </div>

            <PayoffPad u1={game.u1} delta={delta} setDelta={setDelta} />
            {edited ? (
              <button type="button" className="btn" onClick={() => setDelta(ZERO_DELTA.map((r) => [...r]))}>
                Undo my payoff edits
              </button>
            ) : null}
          </div>
        </div>

        {openLandmark ? (
          <div className="plane-provenance">
            <div className="plane-provenance-head">
              <h3>{openLandmark.name}</h3>
              <button type="button" className="btn" onClick={() => setOpen(null)}>
                Close
              </button>
            </div>
            <p className="plane-reading">{openLandmark.reading}</p>
            <ProvenanceCard landmark={openLandmark} />
          </div>
        ) : null}
      </Widget>

      <Widget
        hook="Quadrant III is empty, and that is the experiment"
        lede={
          <p>
            You cannot reach the hatched quadrant from this page, and the reason is a theorem rather than a limitation of
            the widget. For an exact two-player game under this chain both coordinates vanish together: ℛ = 0 if and only
            if the game is potential, if and only if σ_EP = 0.
          </p>
        }
        consequence={
          <>
            Quadrant III is reachable only for a real system, where the two coordinates are read by two different
            instruments from two different kinds of data — pass-through asymmetry on one axis, trajectory irreversibility
            on the other — so one can sit at its own null while the other does not. Nobody has measured a system there
            yet.
          </>
        }
        maths={
          <>
            <p>{QUADRANT_III.why}</p>
            <p>{QUADRANT_III.candidate}</p>
            <p>
              The two coordinates are established as independent, not merely different. Stratified by the harmonic
              fraction, the correlation between them collapses as α → 1 and its residual reverses sign: ρ_S(σ_EP, ℛ)
              runs 0.882, 0.812, 0.856, 0.849, 0.800, 0.870, 0.801, 0.610, 0.323, −0.355 across α = 0.05 … 0.95
              (<code>benchmarks/results/chain_comovement.json</code>). The obvious repair — that ℛ misbehaves only
              because it is a ratio — was pre-registered and refuted by its own test: the numerator alone reaches ρ =
              −0.368 at high α (<code>benchmarks/results/decoupling_mechanism.json</code>, finding F-0007).
            </p>
            <p>
              Two scope limits belong on the same page as the claim. The α-stratified collapse is a{" "}
              <strong>two-player</strong> instrument: at N = 3 and N = 4 the meters do not couple at low α, so the
              collapse cannot be demonstrated there and the programme declines to certify the claim at N &gt; 2
              (F-0023, <code>benchmarks/results/plane_nplayers.json</code>). And the <em>sign</em> of the residual is
              λ-dependent (F-0010), so the headline is that the plane is two-dimensional — not that the second axis
              always points the way it points at λ = 1.2.
            </p>
          </>
        }
      >
        <ul className="quadrant-list">
          {QUADRANTS.map((q) => {
            const here = LANDMARKS.filter((l) => l.quadrant === q.id);
            return (
              <li key={q.id} className="quadrant-item" style={{ borderLeftColor: q.colorVar }}>
                <p className="quadrant-name" style={{ color: q.textVar }}>
                  {q.id}. {q.name} — <span>{q.headline}</span>
                </p>
                <p className="quadrant-consequence">{q.consequence}</p>
                <p className="quadrant-occupants">
                  {here.length ? here.map((l) => `${l.name} (${l.RLabel})`).join(" · ") : QUADRANT_III.status}
                </p>
              </li>
            );
          })}
        </ul>
      </Widget>
    </>
  );
}

function LandmarkHit({
  id,
  cx,
  cy,
  onOpen,
  active,
  label,
}: {
  id: string;
  cx: number;
  cy: number;
  onOpen: (v: string | null) => void;
  active: boolean;
  label: string;
}) {
  return (
    <g
      className="landmark"
      role="button"
      tabIndex={0}
      aria-label={`${label} — show where this number came from`}
      aria-pressed={active}
      onClick={() => onOpen(active ? null : id)}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onOpen(active ? null : id);
        }
      }}
    >
      <circle cx={cx} cy={cy} r={16} fill="transparent" />
      <circle cx={cx} cy={cy} r={active ? 8 : 6} fill="var(--surface)" stroke="var(--text)" strokeWidth={2} />
      <circle cx={cx} cy={cy} r={2.4} fill="var(--text)" />
    </g>
  );
}

function PayoffPad({
  u1,
  delta,
  setDelta,
  }: {
  u1: number[][];
  delta: number[][];
  setDelta: (v: number[][]) => void;
}) {
  return (
    <div className="payoff-pad">
      <p className="panel-label">Row player&apos;s payoffs — drag a cell up or down</p>
      <div className="payoff-grid">
        {u1.map((row, i) =>
          row.map((v, j) => (
            <PayoffCell
              key={`${i}-${j}`}
              i={i}
              j={j}
              value={v}
              onChange={(next) => {
                const d = delta.map((r) => [...r]);
                d[i][j] = Number((d[i][j] + (next - v)).toFixed(3));
                setDelta(d);
              }}
            />
          )),
        )}
      </div>
      <p className="payoff-help">
        Each cell is what the row player gets when the pair play {LABELS.join(", ")} against {LABELS.join(", ")}. Change
        one and the whole equilibrium, the whole chain and both coordinates move.
      </p>
    </div>
  );
}

function PayoffCell({ i, j, value, onChange }: { i: number; j: number; value: number; onChange: (v: number) => void }) {
  const drag = useDragNumber({
    value,
    min: -4,
    max: 4,
    step: 0.1,
    onChange,
    axis: "y",
    travelPx: 200,
    label: `Row payoff when row plays ${LABELS[i]} and column plays ${LABELS[j]}`,
    valueText: (v) => v.toFixed(1),
  });
  return (
    <div className="payoff-cell" {...drag.handleProps} data-dragging={drag.dragging ? "true" : undefined}>
      <span className="payoff-cell-key">
        {LABELS[i]}
        {LABELS[j]}
      </span>
      <span className="payoff-cell-val">{value.toFixed(1)}</span>
    </div>
  );
}
