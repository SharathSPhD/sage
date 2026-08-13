"use client";

/* The problem picker. Each tab is one endpoint under /v1/solve. */

import { useState } from "react";
import { AllocationSolver } from "./AllocationSolver";
import { AuctionSolver } from "./AuctionSolver";
import { ElectricitySolver } from "./ElectricitySolver";
import { MatrixSolver } from "./MatrixSolver";
import { PricingSolver } from "./PricingSolver";
import { RoutingSolver } from "./RoutingSolver";

const TABS = [
  { id: "pricing", label: "Pricing", question: "What price do I set against a rival setting one too?" },
  { id: "auction", label: "Auction", question: "What do I bid in a sealed tender against one credible rival?" },
  {
    id: "electricity",
    label: "Electricity",
    question: "What do I offer into a uniform-price market, and where does it clear?",
  },
  { id: "routing", label: "Routing", question: "Which road do I charge for, and does the city get time back?" },
  { id: "allocation", label: "Allocation", question: "How do I split a fixed budget across contested accounts?" },
  { id: "standards", label: "Coordination", question: "Three moves a side and a table of my own — which move?" },
] as const;

export function Workbench({ initial = "pricing" }: { initial?: string }) {
  const [tab, setTab] = useState<string>(initial);
  const current = TABS.find((t) => t.id === tab) ?? TABS[0];

  return (
    <div>
      <div className="situation-picker" role="group" aria-label="Problem type">
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            data-on={t.id === tab}
            aria-pressed={t.id === tab}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>
      <p className="studio-decision">{current.question}</p>
      {tab === "pricing" && <PricingSolver />}
      {tab === "auction" && <AuctionSolver />}
      {tab === "electricity" && <ElectricitySolver />}
      {tab === "routing" && <RoutingSolver />}
      {tab === "allocation" && <AllocationSolver />}
      {tab === "standards" && <MatrixSolver />}
    </div>
  );
}
