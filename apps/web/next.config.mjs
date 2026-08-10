/** @type {import('next').NextConfig} */
const API_ORIGIN = process.env.SAGE_API_ORIGIN ?? "http://150.136.84.2";

// The browser only ever talks to /api/* on this origin; Next proxies to the
// backend server-side. This sidesteps both CORS and the HTTPS-page →
// HTTP-backend mixed-content block on the deployed frontend.
const nextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_ORIGIN}/:path*` }];
  },
};
export default nextConfig;
