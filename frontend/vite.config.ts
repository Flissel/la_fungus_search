
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/ws': {
        target: 'http://localhost:8011',
        ws: true,
        changeOrigin: true,
      },
      '/start': 'http://localhost:8011',
      '/config': 'http://localhost:8011',
      '/search': 'http://localhost:8011',
      '/answer': 'http://localhost:8011',
      '/status': 'http://localhost:8011',
      '/pause': 'http://localhost:8011',
      '/resume': 'http://localhost:8011',
      '/stop': 'http://localhost:8011',
      '/doc': 'http://localhost:8011',
      '/settings': 'http://localhost:8011',
      '/corpus': 'http://localhost:8011',
      '/jobs': 'http://localhost:8011',
    }
  }
})


