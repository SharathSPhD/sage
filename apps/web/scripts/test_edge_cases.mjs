// Test edge cases for numerical stability

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

const { softmax, argmaxMix, entropy } = mod;

console.log("\n=== EDGE CASE 1: λ = Infinity (Nash) ===");
const payoffs = [102, 108, 115, 113, 104];
try {
  const probs = softmax(payoffs, Infinity);
  console.log(`softmax(payoffs, Infinity): [${probs.map(v => v.toFixed(3)).join(", ")}]`);
  console.log(`  Result: ${probs.some(isNaN) ? "CONTAINS NaN ✗" : "valid ✓"}`);
} catch (e) {
  console.log(`  ERROR: ${e.message}`);
}

console.log("\n=== EDGE CASE 2: argmaxMix with tie ===");
const tied = [10, 10, 10];
try {
  const mix = argmaxMix(tied);
  console.log(`argmaxMix([10, 10, 10]): [${mix.map(v => v.toFixed(3)).join(", ")}]`);
  const sum = mix.reduce((a, b) => a + b, 0);
  console.log(`  Sum: ${sum.toFixed(6)} ${Math.abs(sum - 1) < 1e-10 ? "✓" : "✗ NOT NORMALIZED"}`);
} catch (e) {
  console.log(`  ERROR: ${e.message}`);
}

console.log("\n=== EDGE CASE 3: entropy of pure strategy ===");
const pure = [1, 0, 0];
try {
  const h = entropy(pure);
  console.log(`entropy([1, 0, 0]): ${h.toFixed(6)}`);
  console.log(`  Result: ${isNaN(h) ? "NaN ✗" : h === 0 ? "zero ✓" : "nonzero ✗"}`);
} catch (e) {
  console.log(`  ERROR: ${e.message}`);
}

console.log("\n=== EDGE CASE 4: entropy of uniform ===");
const uniform = [0.25, 0.25, 0.25, 0.25];
try {
  const h = entropy(uniform);
  console.log(`entropy([0.25, 0.25, 0.25, 0.25]): ${h.toFixed(6)}`);
  console.log(`  Expected: ~${Math.log(4).toFixed(6)} (ln(4))`);
  console.log(`  Valid: ${isNaN(h) ? "NaN ✗" : "✓"}`);
} catch (e) {
  console.log(`  ERROR: ${e.message}`);
}

console.log("\nEdge case testing complete.");
