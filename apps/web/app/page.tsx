import Link from "next/link";
import { SolveStudio } from "./solve/SolveStudio";
import { ENGINE_SITUATIONS, SITUATIONS } from "../lib/situations";

const GALLERY = [
  ...SITUATIONS.map((s) => ({ id: s.id, name: s.name, decision: s.decision, href: s.href, illustrative: s.illustrative })),
  ...ENGINE_SITUATIONS.map((s) => ({ id: s.id, name: s.name, decision: s.decision, href: s.href, illustrative: s.illustrative })),
];

export default function Home() {
  return (
    <div className="wrap">
      <section className="hero">
        <h1>Decide, when the other side is deciding too.</h1>
        <p className="lede">
          Give it the numbers you already have — your cost, their likely cost, what you can charge. Get the move to
          make, what they are likely to do, what it is worth, and what would change the answer. It all moves as you
          type.
        </p>
      </section>

      <SolveStudio />

      <section className="home-gallery" aria-labelledby="gallery-heading">
        <h2 id="gallery-heading">Other situations, same machinery</h2>
        <div className="gallery">
          {GALLERY.map((s) => (
            <Link key={s.id} href={s.href} className="card gallery-item">
              <h3>{s.name}</h3>
              <p className="gallery-decision">{s.decision}</p>
              <span className="badge" data-tone={s.illustrative ? undefined : "ok"}>
                {s.illustrative ? "illustrative numbers" : "measured numbers"}
              </span>
            </Link>
          ))}
        </div>
      </section>

      <section className="card home-play">
        <h2>Does any of this beat what you already do?</h2>
        <p>
          Run a hundred rounds against cost-plus, matching the competitor, reacting to their last move, and assuming
          they are perfect. Same rival, same luck, one line each. Some of the time a simpler rule wins — you will see
          which, and when.
        </p>
        <Link href="/play">
          <button data-primary="true">Run it against the usual rules</button>
        </Link>
      </section>
    </div>
  );
}
