/* Client-side mathematics for the /demos explorables.
 *
 * HYBRID-COMPUTE RULE (same as lib/qre.ts): this module exists so a drag feels
 * instant. It is not the authority for any published number. Where a demo shows
 * a committed reading it comes from lib/demos/landmarks.ts, which quotes the
 * artifact file in benchmarks/results/ it was read from.
 *
 * Two of these functions do reproduce committed artifact values exactly, and the
 * demos say so on the page:
 *   responseAsymmetry(RPS, -RPS, 1.2)      -> 0.6928203230275508
 *   responseAsymmetry(RPS5, -RPS5, 1.2)    -> 0.4346570512289449
 *   responseAsymmetry(MP, -MP, 1.2)        -> 1.2
 * against benchmarks/results/reciprocity_harmonic.json
 *   { R_rps_3: 0.6928203230275507, R_rps_5: 0.434657051228945,
 *     R_matching_pennies: 1.1999999999999995 }
 * and glauber(potential game).ep -> ~1e-33 against
 * benchmarks/results/equilibrium_reads_zero.json { max_epr: 4.19e-30 }.
 */

export type Matrix = number[][];

/** Numerically safe softmax of lam * values. */
export function softmax(values: number[], lam: number): number[] {
  const scaled = values.map((v) => lam * v);
  const m = Math.max(...scaled);
  const w = scaled.map((v) => Math.exp(v - m));
  const z = w.reduce((a, b) => a + b, 0);
  return w.map((v) => v / z);
}

export interface QREPoint {
  sigma1: number[];
  sigma2: number[];
  residual: number;
  iterations: number;
  converged: boolean;
}

function iterate(u1: Matrix, u2: Matrix, lam: number, damping: number, maxIter: number, tol: number): QREPoint {
  const n1 = u1.length;
  const n2 = u1[0].length;
  let s1 = new Array<number>(n1).fill(1 / n1);
  let s2 = new Array<number>(n2).fill(1 / n2);
  let residual = Infinity;
  let it = 0;
  for (; it < maxIter; it++) {
    const eu1 = u1.map((row) => row.reduce((acc, v, j) => acc + v * s2[j], 0));
    const eu2 = new Array<number>(n2).fill(0);
    for (let j = 0; j < n2; j++) {
      let acc = 0;
      for (let i = 0; i < n1; i++) acc += u2[i][j] * s1[i];
      eu2[j] = acc;
    }
    const b1 = softmax(eu1, lam);
    const b2 = softmax(eu2, lam);
    residual = 0;
    for (let i = 0; i < n1; i++) residual = Math.max(residual, Math.abs(b1[i] - s1[i]));
    for (let j = 0; j < n2; j++) residual = Math.max(residual, Math.abs(b2[j] - s2[j]));
    s1 = s1.map((v, i) => v + damping * (b1[i] - v));
    s2 = s2.map((v, j) => v + damping * (b2[j] - v));
    if (residual < tol) break;
  }
  return { sigma1: s1, sigma2: s2, residual, iterations: it, converged: residual < tol };
}

/**
 * Damped logit fixed point with one automatic retry at heavier damping.
 * Cycling is real for near-harmonic games at high lam; when it happens we say so
 * rather than quoting the last iterate.
 */
export function solveQRE(u1: Matrix, u2: Matrix, lam: number): QREPoint {
  const first = iterate(u1, u2, lam, 0.35, 20000, 1e-13);
  if (first.converged) return first;
  return iterate(u1, u2, lam, 0.1, 60000, 1e-12);
}

/** In-place Gauss-Jordan inverse. Returns null on a singular matrix. */
export function invert(M: Matrix): Matrix | null {
  const n = M.length;
  const A = M.map((row, i) => [...row, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    if (Math.abs(A[p][c]) < 1e-14) return null;
    [A[c], A[p]] = [A[p], A[c]];
    const d = A[c][c];
    for (let k = 0; k < 2 * n; k++) A[c][k] /= d;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = A[r][c];
      if (f === 0) continue;
      for (let k = 0; k < 2 * n; k++) A[r][k] -= f * A[c][k];
    }
  }
  return A.map((row) => row.slice(n));
}

export interface ResponseReading {
  /** R = ||chi - chi^T||_F / ||chi + chi^T||_F. Zero iff the game is potential. */
  R: number;
  converged: boolean;
  /** Spectral radius proxy of S*B; >= 1 means the resolvent is near-singular. */
  rho: number;
}

/**
 * Response asymmetry R of the equilibrium response matrix chi = (I - S B)^-1 S,
 * with S the block-diagonal softmax Jacobian at the QRE point and B the
 * block-off-diagonal payoff cross-derivative. R = 0 exactly for potential games
 * at every lam (Result 2); its magnitude scales with lam (F-0002), so a level is
 * only comparable at a fixed lam.
 */
export function responseAsymmetry(u1: Matrix, u2: Matrix, lam: number): ResponseReading {
  const pt = solveQRE(u1, u2, lam);
  const n1 = u1.length;
  const n2 = u1[0].length;
  const N = n1 + n2;
  const S: Matrix = Array.from({ length: N }, () => new Array<number>(N).fill(0));
  for (let i = 0; i < n1; i++)
    for (let k = 0; k < n1; k++) S[i][k] = lam * ((i === k ? pt.sigma1[i] : 0) - pt.sigma1[i] * pt.sigma1[k]);
  for (let i = 0; i < n2; i++)
    for (let k = 0; k < n2; k++)
      S[n1 + i][n1 + k] = lam * ((i === k ? pt.sigma2[i] : 0) - pt.sigma2[i] * pt.sigma2[k]);
  const B: Matrix = Array.from({ length: N }, () => new Array<number>(N).fill(0));
  for (let i = 0; i < n1; i++) for (let j = 0; j < n2; j++) B[i][n1 + j] = u1[i][j];
  for (let j = 0; j < n2; j++) for (let i = 0; i < n1; i++) B[n1 + j][i] = u2[i][j];

  const SB: Matrix = Array.from({ length: N }, (_, i) =>
    Array.from({ length: N }, (_, j) => {
      let acc = 0;
      for (let k = 0; k < N; k++) acc += S[i][k] * B[k][j];
      return acc;
    }),
  );
  let rho = 0;
  for (let i = 0; i < N; i++) {
    let row = 0;
    for (let j = 0; j < N; j++) row += Math.abs(SB[i][j]);
    rho = Math.max(rho, row);
  }
  const M: Matrix = Array.from({ length: N }, (_, i) =>
    Array.from({ length: N }, (_, j) => (i === j ? 1 : 0) - SB[i][j]),
  );
  const Mi = invert(M);
  if (!Mi) return { R: NaN, converged: false, rho };
  const chi: Matrix = Array.from({ length: N }, (_, i) =>
    Array.from({ length: N }, (_, j) => {
      let acc = 0;
      for (let k = 0; k < N; k++) acc += Mi[i][k] * S[k][j];
      return acc;
    }),
  );
  let num = 0;
  let den = 0;
  for (let i = 0; i < N; i++)
    for (let j = 0; j < N; j++) {
      num += (chi[i][j] - chi[j][i]) ** 2;
      den += (chi[i][j] + chi[j][i]) ** 2;
    }
  if (den <= 0) return { R: NaN, converged: false, rho };
  return { R: Math.sqrt(num) / Math.sqrt(den), converged: pt.converged, rho };
}

export interface ChainReading {
  /** Stationary distribution over the n1*n2 joint profiles. */
  pi: number[];
  /** One-step transition kernel. */
  W: Matrix;
  /** Net probability current J[x][y] = pi_x W_xy - pi_y W_yx. */
  J: Matrix;
  /** Schnakenberg entropy production of the stationary chain, nats per step. */
  ep: number;
  /** Largest |J| over edges. */
  maxCurrent: number;
  n1: number;
  n2: number;
}

/**
 * Glauber (logit revision) chain on the joint profile lattice: at each step one
 * of the two players is drawn at random and redraws its action from the logit
 * response to the other's current action. Detailed balance holds exactly when the
 * game is a potential game, so ep is zero to machine precision there — that is a
 * theorem about the chain, not a display convention.
 */
export function glauber(u1: Matrix, u2: Matrix, lam: number): ChainReading {
  const n1 = u1.length;
  const n2 = u1[0].length;
  const N = n1 * n2;
  const idx = (i: number, j: number) => i * n2 + j;
  const W: Matrix = Array.from({ length: N }, () => new Array<number>(N).fill(0));
  for (let i = 0; i < n1; i++)
    for (let j = 0; j < n2; j++) {
      const p1 = softmax(
        u1.map((row) => row[j]),
        lam,
      );
      const p2 = softmax(u2[i], lam);
      for (let k = 0; k < n1; k++) if (k !== i) W[idx(i, j)][idx(k, j)] += 0.5 * p1[k];
      for (let k = 0; k < n2; k++) if (k !== j) W[idx(i, j)][idx(i, k)] += 0.5 * p2[k];
    }
  for (let x = 0; x < N; x++) {
    let s = 0;
    for (let y = 0; y < N; y++) if (y !== x) s += W[x][y];
    W[x][x] = 1 - s;
  }
  let pi = new Array<number>(N).fill(1 / N);
  for (let t = 0; t < 20000; t++) {
    const next = new Array<number>(N).fill(0);
    for (let x = 0; x < N; x++) {
      const px = pi[x];
      if (px === 0) continue;
      for (let y = 0; y < N; y++) next[y] += px * W[x][y];
    }
    let d = 0;
    for (let k = 0; k < N; k++) d = Math.max(d, Math.abs(next[k] - pi[k]));
    pi = next;
    if (d < 1e-15) break;
  }
  const J: Matrix = Array.from({ length: N }, () => new Array<number>(N).fill(0));
  let ep = 0;
  let maxCurrent = 0;
  for (let x = 0; x < N; x++)
    for (let y = x + 1; y < N; y++) {
      const f = pi[x] * W[x][y];
      const b = pi[y] * W[y][x];
      if (f <= 0 && b <= 0) continue;
      J[x][y] = f - b;
      J[y][x] = b - f;
      maxCurrent = Math.max(maxCurrent, Math.abs(f - b));
      if (f > 0 && b > 0) ep += (f - b) * Math.log(f / b);
    }
  return { pi, W, J, ep: Math.max(ep, 0), maxCurrent, n1, n2 };
}

/* ---------------- canonical games ---------------- */

/** Rock-paper-scissors, row player. */
export const RPS: Matrix = [
  [0, -1, 1],
  [1, 0, -1],
  [-1, 1, 0],
];

/** Exact potential (common interest) 3x3 coordination game. */
export const COORD: Matrix = [
  [2, 0, 0],
  [0, 2, 0],
  [0, 0, 2],
];

export const negate = (m: Matrix): Matrix => m.map((row) => row.map((v) => -v));

/**
 * alpha = 0 is an exact potential game, alpha = 1 is rock-paper-scissors.
 * This is the same convex morph the phase map sweeps; alpha here is the mixing
 * weight, not the Hodge harmonic fraction of the normalised game.
 */
export function mixGame(alpha: number): [Matrix, Matrix] {
  const u1 = COORD.map((row, i) => row.map((v, j) => (1 - alpha) * v + alpha * RPS[i][j]));
  const u2 = COORD.map((row, i) => row.map((v, j) => (1 - alpha) * v - alpha * RPS[i][j]));
  return [u1, u2];
}

/* ---------------- estimation ---------------- */

export interface LambdaFit {
  lambda: number;
  logLik: number;
  /** Profile-likelihood interval: the lam range within 1.92 log-lik of the peak. */
  ciLow: number;
  ciHigh: number;
  /** True when the likelihood is flat enough that the fit should not be quoted. */
  flat: boolean;
}

/**
 * Grid-search maximum likelihood for lam from observed choices against the
 * expected payoffs the chooser faced. Mirrors the estimator in strataq.fit();
 * the 1.92 cut is the chi-square(1) 95% profile interval. `flat` is true when no
 * lam on the grid is distinguishable from the best one, which is the case the
 * instrument must warn on rather than quote; a boundary maximum at lam = 0 is a
 * real fit and is quoted.
 */
export function fitLambda(choices: number[], payoffRows: number[][], grid?: number[]): LambdaFit | null {
  if (choices.length < 5) return null;
  const gridPoints = grid ?? Array.from({ length: 401 }, (_, i) => i * 0.05);
  let best = -Infinity;
  let bestLam = 0;
  const lls: number[] = [];
  for (const lam of gridPoints) {
    let ll = 0;
    for (let t = 0; t < choices.length; t++) {
      const p = softmax(payoffRows[t], lam);
      ll += Math.log(Math.max(p[choices[t]], 1e-12));
    }
    lls.push(ll);
    if (ll > best) {
      best = ll;
      bestLam = lam;
    }
  }
  let ciLow = gridPoints[0];
  let ciHigh = gridPoints[gridPoints.length - 1];
  for (let i = 0; i < gridPoints.length; i++) {
    if (lls[i] >= best - 1.92) {
      ciLow = gridPoints[i];
      break;
    }
  }
  for (let i = gridPoints.length - 1; i >= 0; i--) {
    if (lls[i] >= best - 1.92) {
      ciHigh = gridPoints[i];
      break;
    }
  }
  const flat = best - Math.min(...lls) < 1.92;
  return { lambda: bestLam, logLik: best, ciLow, ciHigh, flat };
}

/**
 * Logit equilibrium of a 2x2 game by bisection on the column player's
 * probability. Exact where the damped iteration cycles, which it does on
 * matching pennies at every lam worth showing.
 */
export function solve2x2(u1: Matrix, u2: Matrix, lam: number): [number, number] {
  const sig = (x: number) => 1 / (1 + Math.exp(-x));
  const f = (q: number) => q * (u1[0][0] - u1[1][0]) + (1 - q) * (u1[0][1] - u1[1][1]);
  const g = (p: number) => p * (u2[0][0] - u2[0][1]) + (1 - p) * (u2[1][0] - u2[1][1]);
  const h = (q: number) => sig(lam * g(sig(lam * f(q)))) - q;
  let lo = 0;
  let hi = 1;
  if (h(lo) < 0) return [0.5, 0.5];
  for (let i = 0; i < 120; i++) {
    const m = (lo + hi) / 2;
    if (h(m) > 0) lo = m;
    else hi = m;
  }
  const q = (lo + hi) / 2;
  return [sig(lam * f(q)), q];
}
