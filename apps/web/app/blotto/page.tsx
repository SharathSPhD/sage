import type { Metadata } from "next";
import { BlottoLab } from "./BlottoLab";

export const metadata: Metadata = { title: "Blotto lab — SAGE Labs" };

export default function BlottoPage() {
  return (
    <div className="wrap" style={{ paddingTop: "2.2rem" }}>
      <h1 style={{ marginBottom: "0.3rem" }}>Colonel Blotto — the allocation lab</h1>
      <p style={{ color: "var(--text-dim)", maxWidth: "46rem", marginTop: 0 }}>
        Two colonels spread troops across three battlefields; each field goes to whoever commits
        more. Zero-sum — yet <em>not</em> a pure whirlpool: the meter reads α ≈ 0.69 at equal
        budgets (finding F-0005), a genuine gradient component inside the circulation. Budgets
        are the conjugate field: slide them apart and watch the structure move. Every reading is
        the live float64 solver.
      </p>
      <BlottoLab />
    </div>
  );
}
