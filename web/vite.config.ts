import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies /api to the local FastAPI process so the browser
// sees one origin in development, exactly as Vercel will rewrite /api to the
// deployed service in production. Without this, every local request would
// need CORS and the two environments would differ in the one way most likely
// to break on deploy.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_API_TARGET ?? "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: { outDir: "dist", sourcemap: true },
});
