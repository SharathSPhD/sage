/* Client for the problem endpoints of the strataq API.
 *
 * One function per shape: build the same body the Python constructor takes,
 * POST it, read back the Solution with its domain-named fields. Nothing is
 * recomputed in the browser, so a number on screen is the number the library
 * returns for the same inputs.
 *
 * Endpoints: /v1/solve/{pricing,auction,routing,allocation,electricity} and
 * /v1/solve/qre for a payoff table you supply yourself.
 */

"use client";

import { useEffect, useState } from "react";

const BASE = "/api/v1/solve";

export type ProblemKind =
  | "pricing"
  | "auction"
  | "routing"
  | "allocation"
  | "electricity"
  | "qre";

export interface Solved {
  success: boolean;
  message: string;
  warnings: string[];
}

export interface PricingSolution extends Solved {
  price: number;
  profit: number;
  margin: number;
  price_grid: number[];
  profit_curve: number[];
  rival_prices: number[][];
  expected_rival_prices: number[];
  own_price_distribution: number[];
  elasticities: number[][];
  costs: number[];
  firm: number;
  n_firms: number;
  precision: number;
  demand_model: string;
}

export interface AuctionSolution extends Solved {
  bid: number;
  surplus: number;
  win_probability: number;
  bid_grid: number[];
  surplus_curve: number[];
  win_curve: number[];
  rival_bids: number[][];
  own_bid_distribution: number[];
  expected_clearing_bid: number;
  valuation: number;
  reserve: number | null;
  kind: string;
  bidder: number;
  n_bidders: number;
  precision: number;
}

export interface ElectricitySolution extends Solved {
  offer: number;
  offer_curve: [number, number][];
  clearing_price: number;
  clearing_price_distribution: [number, number][];
  revenue: number;
  profit: number;
  dispatch_probability: number;
  profit_curve: number[];
  offers: number[];
  costs: number[];
  capacities: number[];
  demand: number;
  generator: number;
  precision: number;
}

export interface AllocationSolution extends Solved {
  allocation: number[];
  win_probability: number;
  expected_value: number;
  allocation_distribution: number[];
  rival_distribution: number[];
  allocations: number[][];
  rival_allocations: number[][];
  value_curve: number[];
  field_values: number[];
  budget: number;
  rival_budget: number;
  n_fields: number;
  precision: number;
}

export interface TollEffect {
  revenue: number;
  delta_total_cost: number;
  delta_flows: number[];
}

export interface RoutingSolution extends Solved {
  flows: number[];
  travel_times: number[];
  total_cost: number;
  mean_travel_time: number;
  route_flows: number[];
  route_costs: number[];
  toll_effect: TollEffect | null;
  tolls: number[];
  total_demand: number;
  n_links: number;
  n_routes: number;
  n_od: number;
  precision: number;
  residual: number;
  n_iter: number;
}

export interface QRESolution {
  sigma: number[][];
  residual: number;
  n_iter: number;
  warnings: string[];
}

/** POST one problem body and return the Solution. Throws with the API's message. */
export async function solve<T>(kind: ProblemKind, body: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${BASE}/${kind}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const payload = await response.json();
      if (payload && payload.detail) {
        detail = typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail);
      }
    } catch {
      /* the body was not JSON; the status line is all there is */
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export interface SolveState<T> {
  data: T | null;
  error: string | null;
  busy: boolean;
}

/**
 * Solve on every change to the body, debounced, with the previous answer left
 * on screen until the next one lands. Solves take single-digit milliseconds
 * server-side, so the debounce is what keeps a drag smooth, not the solver.
 */
export function useSolve<T>(kind: ProblemKind, body: unknown, delay = 160): SolveState<T> {
  const key = JSON.stringify(body);
  const [state, setState] = useState<SolveState<T>>({ data: null, error: null, busy: true });

  useEffect(() => {
    const controller = new AbortController();
    setState((s) => ({ ...s, busy: true }));
    const timer = setTimeout(() => {
      solve<T>(kind, JSON.parse(key), controller.signal)
        .then((data) => {
          if (controller.signal.aborted) return;
          setState({ data, error: null, busy: false });
        })
        .catch((e: Error) => {
          if (controller.signal.aborted || e.name === "AbortError") return;
          setState((s) => ({ data: s.data, error: e.message, busy: false }));
        });
    }, delay);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [kind, key, delay]);

  return state;
}

/**
 * The same problem at a series of input values — comparative statics. Each
 * point is its own solve; a point that fails comes back null rather than
 * taking the row out from under the others.
 */
export function useSweep<T>(kind: ProblemKind, bodies: unknown[], delay = 420): (T | null)[] {
  const key = JSON.stringify(bodies);
  const [rows, setRows] = useState<(T | null)[]>([]);

  useEffect(() => {
    const controller = new AbortController();
    const timer = setTimeout(() => {
      const list = JSON.parse(key) as unknown[];
      Promise.all(list.map((b) => solve<T>(kind, b, controller.signal).catch(() => null))).then(
        (result) => {
          if (!controller.signal.aborted) setRows(result);
        },
      );
    }, delay);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [kind, key, delay]);

  return rows;
}

/** Evenly spaced sweep points around the current value, inside the field's range. */
export function sweepPoints(current: number, min: number, max: number, step: number, n = 7): number[] {
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    const raw = min + ((max - min) * i) / (n - 1);
    const snapped = Math.round(raw / step) * step;
    const value = Number(snapped.toFixed(8));
    if (!out.includes(value)) out.push(value);
  }
  if (!out.includes(current)) {
    out.push(current);
    out.sort((a, b) => a - b);
  }
  return out;
}

/** Number of levels a [start, stop, step] grid produces, the way the library counts. */
export function gridLevels(start: number, stop: number, step: number): number {
  if (!(step > 0) || stop < start) return 0;
  return Math.floor((stop - start) / step + 1e-9) + 1;
}
