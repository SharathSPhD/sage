import { SiouxFallsLab } from "./SiouxFallsLab";

export const metadata = { title: "Sioux Falls network lab — SAGE Labs" };

export default function NetworkPage() {
  return (
    <div className="wrap" style={{ paddingTop: "2.2rem" }}>
      <h1 style={{ marginBottom: "0.3rem" }}>The Sioux Falls network lab</h1>
      <p style={{ color: "var(--text-dim)", maxWidth: "46rem", marginTop: 0 }}>
        The classic 24-node, 76-link benchmark network with its real demand table — the system
        where the reciprocity meter reads exactly zero (ℛ = 5.7×10⁻¹⁷, gate
        domains.congestion), because logit route choice over BPR travel times is an exact
        potential game. Toll a link and watch the whole equilibrium re-arrange; every solve is
        the deployed Fisk–Newton solver on the real data.
      </p>
      <SiouxFallsLab />
    </div>
  );
}
