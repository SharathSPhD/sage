"use client";

import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";

/** True when the visitor has asked the platform for reduced motion. */
export function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const apply = () => setReduced(mq.matches);
    apply();
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  }, []);
  return reduced;
}

/** Gate every animation on the element actually being on screen. */
export function useInView<T extends Element>(): [React.RefObject<T | null>, boolean] {
  const ref = useRef<T | null>(null);
  const [inView, setInView] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (typeof IntersectionObserver === "undefined") {
      setInView(true);
      return;
    }
    const io = new IntersectionObserver((entries) => setInView(entries[0]?.isIntersecting ?? false), {
      rootMargin: "80px",
    });
    io.observe(el);
    return () => io.disconnect();
  }, []);
  return [ref, inView];
}

const PauseCtx = createContext<{ paused: boolean; toggle: () => void }>({ paused: false, toggle: () => {} });

export function PauseProvider({ children }: { children: React.ReactNode }) {
  const [paused, setPaused] = useState(false);
  const toggle = useCallback(() => setPaused((p) => !p), []);
  return <PauseCtx.Provider value={{ paused, toggle }}>{children}</PauseCtx.Provider>;
}

export function usePause() {
  return useContext(PauseCtx);
}

/** The one control that stops every moving thing on the page. */
export function PauseButton() {
  const { paused, toggle } = usePause();
  return (
    <button type="button" className="btn demo-pause" onClick={toggle} aria-pressed={paused}>
      <span aria-hidden>{paused ? "▶" : "❚❚"}</span>
      {paused ? "Resume motion" : "Pause motion"}
    </button>
  );
}

/**
 * requestAnimationFrame loop that respects the global pause, the reduced-motion
 * preference and whether the figure is on screen.
 */
export function useAnimationFrame(callback: (dtMs: number) => void, active: boolean) {
  const { paused } = usePause();
  const reduced = useReducedMotion();
  const cb = useRef(callback);
  cb.current = callback;
  useEffect(() => {
    if (!active || paused || reduced) return;
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const dt = Math.min(64, now - last);
      last = now;
      cb.current(dt);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [active, paused, reduced]);
}
