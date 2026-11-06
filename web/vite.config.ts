import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies /api to the local FastAPI process so the browser
// sees one origin in development, exactly as Vercel does in production.
// Without it every local request would need CORS, and the two environments
// would differ in the one way most likely to break on deploy.
//
// The target is a literal rather than read from the environment on purpose:
// `process` is a Node global, and reaching for it here once made this config
// typecheck locally - against an @types/node that happened to sit in a
// parent directory - while failing in CI, which has no such parent. Point a
// local UI at a different API with VITE_API_BASE instead.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
  build: { outDir: "dist", sourcemap: true },
});
