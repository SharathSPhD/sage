import type { Metadata } from "next";
import Link from "next/link";
import { AllocationLab } from "./AllocationLab";

export const metadata: Metadata = {
  title: "Splitting a fixed budget — SAGE",
  description: "Three accounts, one budget, a rival splitting theirs. Where do you put it?",
};

export default function AllocationSituation() {
  return (
    <div className="wrap situation-page">
      <p className="crumb">
        <Link href="/situations">All situations</Link>
      </p>
      <h1 className="surface-title">Splitting a fixed budget</h1>
      <p className="surface-lede">
        You and a competitor each divide a fixed resource — heads, spend, engineering weeks — across the same three
        accounts. Whoever commits more takes the account. Neither of you sees the other&apos;s plan until it is set.
      </p>
      <AllocationLab />
      <section className="try-this" aria-labelledby="try-heading">
        <h2 id="try-heading">Three things worth trying</h2>
        <div className="try-grid">
          <article className="card">
            <h3>Equal budgets have no safe plan</h3>
            <p>
              Set both budgets to 4. No split gets more than a fraction of the weight — because any plan you would
              always run is a plan the rival can beat for free. Being unpredictable is not a hedge here, it is the
              answer.
            </p>
          </article>
          <article className="card">
            <h3>One extra unit changes the shape, not just the size</h3>
            <p>
              Give yourself 5 against their 4. Your weight moves toward covering — spreading to deny — and theirs moves
              toward concentrating on fewer accounts. The underdog should gamble; the favourite should not.
            </p>
          </article>
          <article className="card">
            <h3>A disciplined rival does not make you safer</h3>
            <p>
              Push their chase all the way up at equal budgets. Your plan stays spread. Sharpening an opponent in a
              game with no safe move does not give you one — it just makes their spread tighter too.
            </p>
          </article>
        </div>
      </section>
      <p className="source-note" data-illustrative={false}>
        <strong>Measured numbers.</strong> Engine-exact: every allocation is enumerated and solved by the deployed
        float64 solver. There is no sampling and no fitted parameter. What is illustrative is the story about accounts
        — the arithmetic is the same whether the units are sales heads or artillery.
      </p>
      <section className="card next-step">
        <h2>Then what</h2>
        <p>
          The same machinery, on a decision with a price attached: <Link href="/situations/pricing">the weekly shelf
          price</Link>.
        </p>
      </section>
    </div>
  );
}
