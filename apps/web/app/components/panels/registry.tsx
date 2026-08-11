"use client";

import { DynamicsTheater } from "./DynamicsTheater";
import { FreeEnergyDial } from "./FreeEnergyDial";
import { OnePriceObjection } from "./OnePriceObjection";
import { PokePanel } from "./PokePanel";
import { SimplexPortrait } from "./SimplexPortrait";
import { SoftmaxCollapse } from "./SoftmaxCollapse";
import { TwoDials } from "./TwoDials";

/* Each Learn explainer that promises an interactive panel gets it mounted
   inline, right after the prose — the explorable-explanation pattern. The
   keys are the docs/theory slugs. */
const REGISTRY: Record<string, React.ComponentType> = {
  "01-softmax-and-lambda": SoftmaxCollapse,
  "02-the-fixed-point": SimplexPortrait,
  "04-maxent": FreeEnergyDial,
  "05-gibbs-and-potential-games": DynamicsTheater,
  "06-detailed-balance-and-currents": DynamicsTheater,
  "07-reciprocity": PokePanel,
  "08-elasticity-vs-lambda": TwoDials,
  "09-the-one-price-objection": OnePriceObjection,
};

export function ExplorablePanel({ slug }: { slug: string }) {
  const Panel = REGISTRY[slug];
  if (!Panel) return null;
  return (
    <div style={{ margin: "2.5rem 0" }}>
      <Panel />
    </div>
  );
}

export const EXPLORABLE_SLUGS = Object.keys(REGISTRY);
