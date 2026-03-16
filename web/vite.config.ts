import { defineConfig } from 'vite';

export default defineConfig({
  base: '/zipa-onnx-inference/',
  build: {
    outDir: 'dist',
    target: 'es2022',
  },
  worker: {
    format: 'es',
  },
});
