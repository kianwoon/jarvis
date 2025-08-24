import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  esbuild: {
    logOverride: { 'this-is-undefined-in-esm': 'silent' }
  },
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        'multi-agent': resolve(__dirname, 'multi-agent.html'),
        'settings': resolve(__dirname, 'settings.html'),
        'workflow': resolve(__dirname, 'workflow.html'),
        'meta-task': resolve(__dirname, 'meta-task.html'),
        'knowledge-graph': resolve(__dirname, 'knowledge-graph.html'),
        'idc': resolve(__dirname, 'idc.html'),
        'notebook': resolve(__dirname, 'notebook.html'),
        'admin': resolve(__dirname, 'admin.html')
      }
    }
  },
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})