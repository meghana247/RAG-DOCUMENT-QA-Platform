import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { fileURLToPath } from 'node:url'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    // Evaluation results live with the eval harness; the UI reads the same file CI writes.
    alias: { '@eval': fileURLToPath(new URL('../eval', import.meta.url)) },
  },
  server: {
    port: 3000,
    fs: { allow: ['..'] },
    proxy: { '/api': { target: 'http://localhost:8000', rewrite: (p) => p.replace(/^\/api/, '') } },
  },
})
