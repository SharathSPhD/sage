/** @type {import('next').NextConfig} */
const nextConfig = {
  env: { SAGE_API_BASE: process.env.SAGE_API_BASE ?? "http://localhost:8000" },
};
export default nextConfig;
