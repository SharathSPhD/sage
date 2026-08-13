import type { Metadata } from "next";

/* page.tsx is a client component (the whole route is one interactive flow), so
   the route's metadata lives here — same pattern the other routes get from
   their server-side page.tsx. */
export const metadata: Metadata = {
  title: "Diagnose your data — SAGE Labs",
  description:
    "Drop a series and get a position in the irreversibility plane: what kind of strategic system this is, what that changes, and what this data cannot tell you.",
};

export default function DiagnoseLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
