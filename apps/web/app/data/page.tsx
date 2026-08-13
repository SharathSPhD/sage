import type { Metadata } from "next";
import { DataStudio } from "./DataStudio";

export const metadata: Metadata = {
  title: "Bring your own data",
  description:
    "Upload or paste a CSV of observed choices, map the columns, fit the precision those choices imply, and solve on the levels your file contains.",
};

export default function DataPage() {
  return <DataStudio />;
}
