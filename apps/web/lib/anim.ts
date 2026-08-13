"use client";

/* Motion primitives.
 *
 * Everything that changes on this site changes by interpolation, not by
 * jumping: a number tweens to its new value, a curve morphs point by
 * point, a bar grows. The interpolation runs on requestAnimationFrame
 * over plain numbers, so the SVG geometry is recomputed each frame and
 * morphs identically in every browser (CSS `transition: d` does not).
 *
 * Every hook reads `prefers-reduced-motion` and, when it is set, snaps
 * to the target on the first frame instead of animating.
 */

import { useEffect, useRef, useState } from "react";

/** True when the visitor has asked for reduced motion. Live-updating. */
export function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const sync = () => setReduced(mq.matches);
    sync();
    mq.addEventListener("change", sync);
    return () => mq.removeEventListener("change", sync);
  }, []);
  return reduced;
}

/* easeOutCubic: fast start, soft landing — the same feel as the CSS --ease. */
const ease = (t: number) => 1 - Math.pow(1 - t, 3);

/** A single number that glides to its target instead of jumping. */
export function useTween(target: number, ms = 420): number {
  const reduced = usePrefersReducedMotion();
  const [value, setValue] = useState(target);
  const from = useRef(target);
  const raf = useRef<number | null>(null);

  useEffect(() => {
    if (reduced || ms <= 0 || !Number.isFinite(target)) {
      from.current = target;
      setValue(target);
      return;
    }
    const start = performance.now();
    const a = from.current;
    const b = target;
    if (a === b) return;
    const step = (now: number) => {
      const t = Math.min(1, (now - start) / ms);
      const v = a + (b - a) * ease(t);
      from.current = v;
      setValue(v);
      if (t < 1) raf.current = requestAnimationFrame(step);
    };
    raf.current = requestAnimationFrame(step);
    return () => {
      if (raf.current !== null) cancelAnimationFrame(raf.current);
    };
  }, [target, ms, reduced]);

  return value;
}

/**
 * A whole vector that morphs to its target. Length changes are handled by
 * resampling the previous vector onto the new length first, so a grid that
 * gains or loses levels still morphs rather than snapping.
 */
export function useTweenArray(target: number[], ms = 420): number[] {
  const reduced = usePrefersReducedMotion();
  const [value, setValue] = useState<number[]>(target);
  const from = useRef<number[]>(target);
  const raf = useRef<number | null>(null);
  const key = target.length > 0 ? `${target.length}:${target.join(",")}` : "0:";

  useEffect(() => {
    const b = target;
    if (reduced || ms <= 0 || b.length === 0) {
      from.current = b;
      setValue(b);
      return;
    }
    const a = resample(from.current, b.length);
    if (a.every((v, i) => Math.abs(v - b[i]) < 1e-12)) {
      from.current = b;
      setValue(b);
      return;
    }
    const start = performance.now();
    const step = (now: number) => {
      const t = Math.min(1, (now - start) / ms);
      const e = ease(t);
      const next = b.map((bv, i) => a[i] + (bv - a[i]) * e);
      from.current = next;
      setValue(next);
      if (t < 1) raf.current = requestAnimationFrame(step);
    };
    raf.current = requestAnimationFrame(step);
    return () => {
      if (raf.current !== null) cancelAnimationFrame(raf.current);
    };
    // `key` stands in for the array's contents; `target` itself is a new
    // reference on every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, ms, reduced]);

  return value.length === target.length ? value : target;
}

/** Linear resample of `xs` onto `n` points, so vectors of different length can morph. */
export function resample(xs: number[], n: number): number[] {
  if (n <= 0) return [];
  if (xs.length === 0) return new Array(n).fill(0);
  if (xs.length === n) return xs;
  if (xs.length === 1) return new Array(n).fill(xs[0]);
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    const u = (i * (xs.length - 1)) / (n - 1);
    const lo = Math.floor(u);
    const hi = Math.min(xs.length - 1, lo + 1);
    out.push(xs[lo] + (xs[hi] - xs[lo]) * (u - lo));
  }
  return out;
}

/**
 * Hold the last non-null value. A solve in flight leaves the previous
 * answer on screen rather than blanking the panel — no skeleton flash,
 * no layout jump.
 */
export function useLast<T>(value: T | null | undefined): T | null {
  const held = useRef<T | null>(null);
  if (value !== null && value !== undefined) held.current = value;
  return held.current;
}
