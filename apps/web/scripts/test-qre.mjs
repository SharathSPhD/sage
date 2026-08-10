/* Golden-file agreement test: lib/qre.ts vs the gated strataq library.
 * Run: node scripts/test-qre.mjs   (after `npx tsc` or via the checked-in build)
 * Kept dependency-free: compiles qre.ts on the fly with the TypeScript API.
 */

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

const goldens = JSON.parse(readFileSync(join(here, "..", "lib", "goldens.json"), "utf-8"));

const TOL = 1e-6; // client float64 vs library float64: iteration paths differ, fixed points agree
let failures = 0;

for (const [name, kase] of Object.entries(goldens.cases)) {
  const [u1, u2] = kase.payoffs;
  for (const solve of kase.solves) {
    const got = mod.solveQRE({ u1, u2 }, solve.lam);
    const want = solve.sigma;
    const err = Math.max(
      ...got.sigma1.map((v, i) => Math.abs(v - want[0][i])),
      ...got.sigma2.map((v, j) => Math.abs(v - want[1][j])),
    );
    const ok = got.converged && err < TOL;
    if (!ok) failures++;
    console.log(`${ok ? "PASS" : "FAIL"} ${name} lam=${solve.lam} max|Δσ|=${err.toExponential(2)}`);
  }
}

if (failures > 0) {
  console.error(`${failures} golden disagreement(s) — client math has drifted from the library`);
  process.exit(1);
}
console.log("client math agrees with library goldens");
