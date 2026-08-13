import type { Metadata } from "next";
import { DemoChrome, DemoHeader } from "../components/chrome";
import { Treasures } from "./Treasures";

export const metadata: Metadata = {
  title: "Ten Little Treasures",
  description:
    "A payoff change that leaves the Nash prediction exactly where it was and moves real experimental behaviour from 48% to 96%. Guess what the subjects did, then watch one precision dial do what Nash cannot.",
};

export default function Page() {
  return (
    <DemoChrome>
      <DemoHeader
        eyebrow="demo · you against 50 subjects"
        title="Ten Little Treasures"
        standfirst={
          <p>
            In 2001 Jacob Goeree and Charles Holt ran a set of games chosen so that game theory&apos;s predictions were
            either exactly right or spectacularly wrong, often between two versions of the same game. This page puts you
            in front of one of them before you see the answer. Every experimental number here is theirs.
          </p>
        }
      />
      <Treasures />
    </DemoChrome>
  );
}
