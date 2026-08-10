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

// ---- primitives: every exported function the panels rely on ----
const prim = goldens.primitives;
const close = (a, b, tol = 1e-9) => Math.abs(a - b) < tol;
const vecClose = (a, b, tol = 1e-9) => a.length === b.length && a.every((v, i) => close(v, b[i], tol));

for (const c of prim.softmax) {
  const got = mod.softmax(c.values, c.lam);
  const okP = vecClose(got, c.probs);
  const okH = close(mod.entropy(got), c.entropy);
  if (!okP || !okH) failures++;
  console.log(`${okP && okH ? "PASS" : "FAIL"} softmax+entropy lam=${c.lam}`);
}
for (const c of prim.logit_flow) {
  const got = mod.logitFlow(c.u, c.x, c.lam);
  const ok = vecClose(got, c.flow);
  if (!ok) failures++;
  console.log(`${ok ? "PASS" : "FAIL"} logitFlow lam=${c.lam} x=${c.x}`);
}
{
  const c = prim.coordination_rest_point;
  const path = mod.logitTrajectory(c.u, c.seed, c.lam, { dt: 0.05, steps: 3000 });
  const end = path[path.length - 1];
  const ok = vecClose(end, c.rest, 1e-5) && path.every((p) => close(p.reduce((a, b) => a + b, 0), 1, 1e-8));
  if (!ok) failures++;
  console.log(`${ok ? "PASS" : "FAIL"} logitTrajectory rest point + simplex invariant`);
}
for (const c of prim.argmax) {
  const ok = vecClose(mod.argmaxMix(c.values), c.mix);
  if (!ok) failures++;
  console.log(`${ok ? "PASS" : "FAIL"} argmaxMix ${JSON.stringify(c.values)}`);
}
{
  const c = prim.expected_payoffs_col;
  const ok = vecClose(mod.expectedPayoffsCol(c.u2, c.row_mix), c.out);
  if (!ok) failures++;
  console.log(`${ok ? "PASS" : "FAIL"} expectedPayoffsCol`);
}

if (failures > 0) {
  console.error(`${failures} golden disagreement(s) — client math has drifted from the library`);
  process.exit(1);
}
console.log("client math agrees with library goldens");
