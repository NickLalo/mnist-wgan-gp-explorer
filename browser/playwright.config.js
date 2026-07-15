import {defineConfig} from '@playwright/test';

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 120_000,
  expect: {timeout: 90_000},
  use: {
    baseURL: 'http://127.0.0.1:4173',
    viewport: {width: 1600, height: 950},
    screenshot: 'only-on-failure',
  },
  webServer: {
    command: 'npm run build && npm run preview -- --host 127.0.0.1 --port 4173',
    url: 'http://127.0.0.1:4173',
    timeout: 120_000,
    reuseExistingServer: true,
  },
});
