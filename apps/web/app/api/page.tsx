import type { Metadata } from "next";
import { ApiConsole } from "./ApiConsole";

export const metadata: Metadata = {
  title: "API",
  description:
    "Every strataq endpoint with its request and response schema, a live try-it console, and copyable curl and Python — read from the service's own openapi.json.",
};

export default function ApiPage() {
  return <ApiConsole />;
}
