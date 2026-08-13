import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Demos — four things you can only do here",
  description:
    "Play rock–paper–scissors and have your own rationality fitted by maximum likelihood; watch probability circulate on the nine joint states of a game; predict what real experimental subjects did; and put a game on the irreversibility plane next to Sioux Falls, Dominick's and CAISO.",
};

interface Card {
  href: string;
  eyebrow: string;
  title: string;
  blurb: string;
  claim: string;
  art: React.ReactNode;
}

const CARDS: Card[] = [
  {
    href: "/demos/you-vs-the-model",
    eyebrow: "30 seconds",
    title: "You vs the Model",
    blurb:
      "Play rock–paper–scissors. Your rationality parameter is fitted by maximum likelihood from the choices you make, placed on a scale against measured systems, and then played back at you as a prediction of your next move.",
    claim: "The estimator is the one the project runs on market data.",
    art: <ThumbRps />,
  },
  {
    href: "/demos/whirlpool",
    eyebrow: "watch it turn",
    title: "The Whirlpool",
    blurb:
      "The nine joint states of a two-player game, with probability flowing along the edges between them. Drag the game from a landscape to rock–paper–scissors and the flow stops cancelling.",
    claim: "At the potential end entropy production is exactly zero — a theorem, not a rounding convention.",
    art: <ThumbWhirl />,
  },
  {
    href: "/demos/ten-little-treasures",
    eyebrow: "you against 50 subjects",
    title: "Ten Little Treasures",
    blurb:
      "A payoff change that leaves the Nash prediction untouched and moves real behaviour from 48% to 96%. Guess where the subjects went, then watch one dial do what Nash cannot.",
    claim: "Every experimental number is from Goeree & Holt, AER 2001.",
    art: <ThumbBars />,
  },
  {
    href: "/demos/the-plane",
    eyebrow: "nowhere else on the web",
    title: "The Plane",
    blurb:
      "Response asymmetry against dissipation, with Sioux Falls, Dominick's, CAISO, rock–paper–scissors and Blotto as permanent landmarks. Drag your game and watch it leave the line a one-dimensional theory would predict.",
    claim: "Quadrant III is marked empty, because that is the programme's open decisive test.",
    art: <ThumbPlane />,
  },
];

export default function DemosIndex() {
  return (
    <>
      <header className="demo-index-head">
        <p className="eyebrow">explorables</p>
        <h1>Four demos, four ideas, one hand on each</h1>
        <p className="lede">
          Each of these runs its own mathematics live in your browser — a logit fixed point, a nine-state Glauber chain
          with its stationary distribution, Schnakenberg entropy production, a maximum-likelihood fit. Nothing is
          pre-rendered, and every published number on the pages names the file it was read from.
        </p>
      </header>
      <div className="demo-gallery">
        {CARDS.map((c) => (
          <Link key={c.href} href={c.href} className="card demo-card">
            <div className="demo-card-art" aria-hidden>
              {c.art}
            </div>
            <p className="eyebrow">{c.eyebrow}</p>
            <h2>{c.title}</h2>
            <p className="demo-card-blurb">{c.blurb}</p>
            <p className="demo-card-claim">{c.claim}</p>
            <span className="entry-go">Open it →</span>
          </Link>
        ))}
      </div>
    </>
  );
}

function ThumbRps() {
  return (
    <svg viewBox="0 0 160 90" role="presentation">
      <line x1="16" y1="60" x2="144" y2="60" stroke="var(--border-strong)" strokeWidth="2" />
      {[30, 62, 96].map((x) => (
        <line key={x} x1={x} y1="55" x2={x} y2="65" stroke="var(--text-3)" />
      ))}
      <circle cx="78" cy="60" r="8" fill="var(--accent-strong)" />
      <circle cx="60" cy="60" r="6" fill="var(--accent)" opacity="0.25" />
      <circle cx="48" cy="60" r="6" fill="var(--accent)" opacity="0.15" />
      <text x="78" y="40" textAnchor="middle" fontSize="13" fontWeight="700" fill="var(--accent-strong)">
        λ̂
      </text>
    </svg>
  );
}

function ThumbWhirl() {
  return (
    <svg viewBox="0 0 160 90" role="presentation">
      {[0, 1, 2].map((i) =>
        [0, 1, 2].map((j) => (
          <circle key={`${i}${j}`} cx={44 + j * 36} cy={18 + i * 27} r={7} fill="var(--surface)" stroke="var(--q-landscape)" strokeWidth="1.6" />
        )),
      )}
      <path d="M44 18 L80 18 L116 45 L80 72 L44 45 Z" fill="none" stroke="var(--q-whirlpool)" strokeWidth="2" opacity="0.85" />
      <circle cx="98" cy="31" r="2.6" fill="var(--q-whirlpool)" />
      <circle cx="62" cy="59" r="2.6" fill="var(--q-whirlpool)" />
    </svg>
  );
}

function ThumbBars() {
  return (
    <svg viewBox="0 0 160 90" role="presentation">
      <line x1="20" y1="76" x2="146" y2="76" stroke="var(--border-strong)" />
      <rect x="30" y="46" width="26" height="30" fill="var(--accent-strong)" />
      <rect x="68" y="44" width="26" height="32" fill="var(--text-3)" />
      <rect x="106" y="16" width="26" height="60" fill="var(--q-whirlpool)" />
      <line x1="20" y1="44" x2="146" y2="44" stroke="var(--text-3)" strokeDasharray="4 4" />
    </svg>
  );
}

function ThumbPlane() {
  return (
    <svg viewBox="0 0 160 90" role="presentation">
      <rect x="20" y="10" width="120" height="66" fill="var(--surface-2)" stroke="var(--border)" />
      <line x1="80" y1="10" x2="80" y2="76" stroke="var(--border-strong)" />
      <line x1="20" y1="45" x2="140" y2="45" stroke="var(--border-strong)" />
      <path d="M20 76 L140 12" stroke="var(--text-3)" strokeDasharray="3 4" fill="none" />
      <circle cx="34" cy="66" r="4" fill="var(--q-landscape)" />
      <circle cx="120" cy="22" r="4" fill="var(--q-whirlpool)" />
      <circle cx="104" cy="58" r="5" fill="var(--accent-strong)" />
      <text x="86" y="72" fontSize="9" fill="var(--q-stalled-text)" fontStyle="italic">
        empty
      </text>
    </svg>
  );
}
