import type { Metadata } from "next";
import { DemoChrome, DemoHeader } from "../components/chrome";
import { Whirlpool } from "./Whirlpool";

export const metadata: Metadata = {
  title: "The Whirlpool",
  description:
    "Probability circulating on the nine joint states of a two-player game, with net currents and Schnakenberg entropy production computed live. At the potential end every edge cancels and entropy production is exactly zero.",
};

export default function Page() {
  return (
    <DemoChrome>
      <DemoHeader
        eyebrow="demo · watch it turn"
        title="The Whirlpool"
        standfirst={
          <p>
            Rock–paper–scissors never settles. A coordination game does. This figure shows why: the same nine states,
            the same rule for moving between them, and one dial that decides whether the flow along every edge cancels
            or goes round in a loop for ever.
          </p>
        }
      />
      <Whirlpool />
    </DemoChrome>
  );
}
