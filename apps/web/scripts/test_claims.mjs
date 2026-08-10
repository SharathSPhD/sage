// Test scientific claims using client-side qre.ts

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import ts from "typescript";

const here = dirname(fileURLToPath(import.meta.url));
const src = readFileSync(join(here, "..", "lib", "qre.ts"), "utf-8");
const js = ts.transpileModule(src, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
}).outputText;
const mod = await import(`data:text/javascript;base64,${Buffer.from(js).toString("base64")}`);

const { softmax, solveQRE, argmaxMix, expectedPayoffsRow, entropy } = mod;

// CLAIM 1: TwoDials — λ never changes which price is best
console.log("\n=== CLAIM 1: λ never changes which price is best ===");

const COST = 1.6;
const PRICES = [1.7, 1.72, 1.74, 1.76, 1.78];

function profits(elasticity) {
  return PRICES.map((p) => (p - COST) * 100 * Math.exp(-elasticity * (p - PRICES[0])));
}

const elasticity = 18;
const u = profits(elasticity);
const best_price_idx = u.indexOf(Math.max(...u));
console.log(`At elasticity ${elasticity}, payoffs: [${u.map(v => v.toFixed(1)).join(", ")}]`);
console.log(`Best price: £${PRICES[best_price_idx]} (index ${best_price_idx})`);

// Test different λ values with frozen elasticity
const test_lambdas = [0.1, 0.5, 2, 5, 10];
let claim_holds = true;
for (const lam of test_lambdas) {
  const p = softmax(u, lam);
  const best_at_lam = p.indexOf(Math.max(...p));
  const matches = best_at_lam === best_price_idx;
  console.log(`  λ=${lam.toFixed(1)}: best action still ${matches ? "YES ✓" : "NO ✗"} (index ${best_at_lam})`);
  if (!matches) claim_holds = false;
}
console.log(`Claim: λ never changes best price: ${claim_holds ? "PASS" : "FAIL"}`);

// CLAIM 2: PokePanel reciprocity — stays ~0 for potential games at large pokes
console.log("\n=== CLAIM 2: Reciprocity stays ~0 for potential games at large pokes ===");

const coord_game = {
  u1: [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
  u2: [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
};
const lam = 1.2;

const base = solveQRE(coord_game, lam);
console.log(`Baseline: σ1=[${base.sigma1.map(v => v.toFixed(3)).join(", ")}], σ2=[${base.sigma2.map(v => v.toFixed(3)).join(", ")}]`);

const test_pokes = [0.3, 1.0, 1.5];
for (const h of test_pokes) {
  // Poke player 1's action 0
  const game_poke1 = {
    u1: [
      [coord_game.u1[0][0] + h, coord_game.u1[0][1] + h, coord_game.u1[0][2] + h],
      coord_game.u1[1],
      coord_game.u1[2]
    ],
    u2: coord_game.u2
  };
  const sol_poke1 = solveQRE(game_poke1, lam);
  
  // Poke player 2's action 0
  const game_poke2 = {
    u1: coord_game.u1,
    u2: [
      [coord_game.u2[0][0] + h, coord_game.u2[0][1], coord_game.u2[0][2]],
      [coord_game.u2[1][0] + h, coord_game.u2[1][1], coord_game.u2[1][2]],
      [coord_game.u2[2][0] + h, coord_game.u2[2][1], coord_game.u2[2][2]]
    ]
  };
  const sol_poke2 = solveQRE(game_poke2, lam);
  
  // Cross-readings
  const cross12 = sol_poke1.sigma2[0] - base.sigma2[0];
  const cross21 = sol_poke2.sigma1[0] - base.sigma1[0];
  const asym = Math.abs(cross12 - cross21);
  
  const passes = asym < 1e-3;
  console.log(`  h=${h}: |cross12 - cross21| = ${asym.toExponential(2)} ${passes ? "✓" : "✗"}`);
}

// CLAIM 3: OnePriceObjection — point rival vs QRE distribution
console.log("\n=== CLAIM 3: OnePriceObjection — argmax point rival vs QRE distribution ===");

// Duopoly
const PRICES2 = ["£1.70", "£1.72", "£1.74", "£1.76", "£1.78"];
const margin = [10, 12, 14, 16, 18];
const u_fn = (i, j) => {
  const share = i < j ? 0.75 : i === j ? 0.5 : 0.25;
  return margin[i] * share * 10;
};
const u1_duopoly = PRICES2.map((_, i) => PRICES2.map((_, j) => u_fn(i, j)));
const u2_duopoly = PRICES2.map((_, i) => PRICES2.map((_, j) => u_fn(j, i)));
const duopoly = { u1: u1_duopoly, u2: u2_duopoly };

const test_rival_lams = [0.1, 1.0, 10.0];
for (const rival_lam of test_rival_lams) {
  const qre = solveQRE(duopoly, rival_lam);
  const rivalMix = qre.sigma2;
  
  // Point rival: argmax of payoffs against MY QRE mix
  const rivalPayoffs = duopoly.u2.reduce(
    (acc, row, i) => acc.map((v, j) => v + row[j] * qre.sigma1[i]),
    new Array(PRICES2.length).fill(0)
  );
  const pointRival = argmaxMix(rivalPayoffs);
  
  // My expected payoffs against each rival model
  const pointEU = expectedPayoffsRow(duopoly.u1, pointRival);
  const qreEU = expectedPayoffsRow(duopoly.u1, rivalMix);
  
  const pointBest = pointEU.indexOf(Math.max(...pointEU));
  const qreBest = qreEU.indexOf(Math.max(...qreEU));
  const regret = qreEU[qreBest] - qreEU[pointBest];
  
  const same = pointBest === qreBest;
  console.log(`  rival λ=${rival_lam}: point→idx${pointBest} QRE→idx${qreBest} regret=${regret.toFixed(2)} ${same ? "same" : "DIFF"}`);
}

console.log("\nAll numerical claims tested.");
