import { defineConfig } from 'vite';

export default defineConfig({
  base: '/zipa-onnx-inference/',
  define: {
    __BUILD_TIME__: JSON.stringify(new Date().toISOString()),
  },
  build: {
    outDir: 'dist',
    target: 'es2022',
  },
  worker: {
    format: 'es',
  },
});
