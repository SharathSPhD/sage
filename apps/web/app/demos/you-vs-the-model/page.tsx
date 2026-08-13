import type { Metadata } from "next";
import { DemoChrome, DemoHeader } from "../components/chrome";
import { RockPaperScissors } from "./RockPaperScissors";

export const metadata: Metadata = {
  title: "You vs the Model",
  description:
    "Play rock–paper–scissors and have your rationality parameter fitted by maximum likelihood from your own choices, placed against measured systems, and played back at you as a prediction.",
};

export default function Page() {
  return (
    <DemoChrome>
      <DemoHeader
        eyebrow="demo · 30 seconds"
        title="You vs the Model"
        standfirst={
          <p>
            Every number on this page is computed from choices you are about to make. The model fits your rationality
            with maximum likelihood, places you on a scale against systems that have actually been measured, and then
            uses your own fitted number to predict what you will do next.
          </p>
        }
      />
      <RockPaperScissors />
    </DemoChrome>
  );
}
