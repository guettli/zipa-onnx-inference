import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  timeout: 120_000,
  use: {
    // Use Chromium only — Chrome DevTools Protocol for memory metrics
    // (Safari does not expose heap stats via CDP)
    channel: 'chromium',
    headless: true,
    baseURL: 'http://localhost:4173',
  },
  projects: [{ name: 'chromium', use: { channel: 'chromium' } }],
  // Start Vite preview before tests
  webServer: {
    command: 'pnpm preview',
    url: 'http://localhost:4173/zipa-onnx-inference/',
    reuseExistingServer: true,
    timeout: 30_000,
  },
});
