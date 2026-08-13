/* Ready-to-send bodies for the endpoints a visitor is most likely to try.
 *
 * These are illustrative inputs — the numbers are the ones the studio ships
 * as its defaults, not measurements of anything. Every other endpoint falls
 * back to a body generated from its own schema.
 */

export const EXAMPLE_BODIES: Record<string, unknown> = {
  "POST /v1/solve/pricing": {
    costs: [1.0, 1.05],
    grid_range: [1.09, 1.89, 0.1],
    demand: { kind: "logit", price_sensitivity: 3.6, quality: [5.94, 5.94], market_size: 400 },
    precision: 1.5,
  },
  "POST /v1/solve/auction": {
    costs: [85000, 88000],
    grid_range: [88000, 116000, 4000],
    reserve: 112000,
    precision: 5e-4,
  },
  "POST /v1/solve/electricity": {
    costs: [20.0, 22.0],
    offers_range: [20.0, 60.0, 5.0],
    capacities: [100.0, 100.0],
    demand: 80.0,
    precision: 0.05,
  },
  "POST /v1/solve/routing": {
    network: "sioux_falls",
    precision: 0.5,
    k_routes: 3,
    max_od: 12,
  },
  "POST /v1/solve/allocation": {
    budget: 5,
    field_values: [1.0, 1.0, 2.0],
    rival_budget: 5,
    precision: 2.0,
  },
  "POST /v1/solve/qre": {
    payoffs: [
      [
        [0, -1, 1],
        [1, 0, -1],
        [-1, 1, 0],
      ],
      [
        [0, 1, -1],
        [-1, 0, 1],
        [1, -1, 0],
      ],
    ],
    lam: 1.5,
  },
  "POST /v1/fit": {
    payoffs: [
      [
        [0, -1, 1],
        [1, 0, -1],
        [-1, 1, 0],
      ],
      [
        [0, 1, -1],
        [-1, 0, 1],
        [1, -1, 0],
      ],
    ],
    counts: [
      [40, 35, 25],
      [30, 40, 30],
    ],
    method: "mle",
    ci: "bootstrap",
    n_boot: 200,
  },
  "POST /v1/response": {
    payoffs: [
      [
        [0, -1, 1],
        [1, 0, -1],
        [-1, 1, 0],
      ],
      [
        [0, 1, -1],
        [-1, 0, 1],
        [1, -1, 0],
      ],
    ],
    lam: 1.5,
  },
  "POST /v1/decompose": {
    payoffs: [
      [
        [0, -1, 1],
        [1, 0, -1],
        [-1, 1, 0],
      ],
      [
        [0, 1, -1],
        [-1, 0, 1],
        [1, -1, 0],
      ],
    ],
    lam: 1.5,
  },
  "POST /v1/dynamics/stationary": {
    payoffs: [
      [
        [0, -1, 1],
        [1, 0, -1],
        [-1, 1, 0],
      ],
      [
        [0, 1, -1],
        [-1, 0, 1],
        [1, -1, 0],
      ],
    ],
    lam: 1.5,
  },
  "POST /v1/domains/blotto/read": { budget_a: 3, budget_b: 3, lam: 1.5 },
  "POST /v1/domains/sioux_falls/sue": { theta: 0.5 },
};

/** The Python call that corresponds to an endpoint, where the library has one. */
export const PYTHON_EQUIVALENT: Record<string, string> = {
  "POST /v1/solve/pricing":
    "import strataq as sq\n\n" +
    "solution = sq.PricingProblem(\n" +
    "    costs=[1.00, 1.05],\n" +
    "    grid=(1.09, 1.89, 0.10),\n" +
    "    demand=sq.LogitDemand(3.6, [5.94, 5.94], market_size=400),\n" +
    "    precision=1.5,\n" +
    ").solve()\n" +
    "print(solution.price, solution.profit)",
  "POST /v1/solve/auction":
    "import strataq as sq\n\n" +
    "solution = sq.AuctionProblem(\n" +
    "    costs=[85_000, 88_000],\n" +
    "    grid=(88_000, 116_000, 4_000),\n" +
    "    reserve=112_000,\n" +
    "    precision=5e-4,\n" +
    ").solve()\n" +
    "print(solution.bid, solution.win_probability)",
  "POST /v1/solve/electricity":
    "import strataq as sq\n\n" +
    "solution = sq.ElectricityProblem(\n" +
    "    costs=[20.0, 22.0],\n" +
    "    offers=(20.0, 60.0, 5.0),\n" +
    "    capacities=[100.0, 100.0],\n" +
    "    demand=80.0,\n" +
    "    precision=0.05,\n" +
    ").solve()\n" +
    "print(solution.offer, solution.clearing_price)",
  "POST /v1/solve/routing":
    "import strataq as sq\n\n" +
    'solution = sq.RoutingProblem(network="sioux_falls", tolls={28: 5.0}, precision=0.5).solve()\n' +
    "print(solution.total_cost, solution.toll_effect)",
  "POST /v1/solve/allocation":
    "import strataq as sq\n\n" +
    "solution = sq.AllocationProblem(budget=5, field_values=[1.0, 1.0, 2.0], precision=2.0).solve()\n" +
    "print(solution.allocation, solution.win_probability)",
};
