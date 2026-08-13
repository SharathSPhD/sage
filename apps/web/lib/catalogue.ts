/* The problem types the app solves, and where each one lives. */

export interface ProblemEntry {
  id: string;
  name: string;
  /** The decision, in one line. */
  question: string;
  /** The quantities the endpoint returns, named as the solution names them. */
  returns: string;
  endpoint: string;
  href: string;
}

export const PROBLEMS: ProblemEntry[] = [
  {
    id: "pricing",
    name: "Pricing",
    question: "What price do I set against a rival setting one too?",
    returns: "price, profit, margin, rival price distribution, elasticities",
    endpoint: "/v1/solve/pricing",
    href: "/situations/pricing",
  },
  {
    id: "auction",
    name: "Auction and tender",
    question: "What do I bid in a sealed tender against one credible rival?",
    returns: "bid, expected surplus, win probability, rival bid distribution",
    endpoint: "/v1/solve/auction",
    href: "/situations/procurement",
  },
  {
    id: "electricity",
    name: "Electricity offers",
    question: "What do I offer into a uniform-price market, and where does it clear?",
    returns: "offer, clearing price, revenue, dispatch probability",
    endpoint: "/v1/solve/electricity",
    href: "/situations/electricity",
  },
  {
    id: "routing",
    name: "Traffic assignment",
    question: "Which link do I toll, and does the network get time back?",
    returns: "link flows, travel times, total cost, toll revenue",
    endpoint: "/v1/solve/routing",
    href: "/situations/routing",
  },
  {
    id: "allocation",
    name: "Budget allocation",
    question: "How do I split a fixed budget across contested accounts?",
    returns: "allocation, win probability, expected value",
    endpoint: "/v1/solve/allocation",
    href: "/situations/allocation",
  },
  {
    id: "standards",
    name: "Payoff table",
    question: "Three moves a side and a table of my own — which move?",
    returns: "each side's move distribution and the value of every move",
    endpoint: "/v1/solve/qre",
    href: "/situations/standards",
  },
];
