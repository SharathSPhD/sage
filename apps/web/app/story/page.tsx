import type { Metadata } from "next";
import { StoryFlow } from "./StoryFlow";

export const metadata: Metadata = { title: "The 5-minute tour — SAGE" };

export default function StoryPage() {
  return <StoryFlow />;
}
