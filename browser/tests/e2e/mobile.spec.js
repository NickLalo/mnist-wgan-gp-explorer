import {devices, expect, test} from '@playwright/test';

const iphone = devices['iPhone 13'];

test.use({
  viewport: iphone.viewport,
  deviceScaleFactor: iphone.deviceScaleFactor,
  isMobile: iphone.isMobile,
  hasTouch: iphone.hasTouch,
  userAgent: iphone.userAgent,
});

test('uses the low-memory inference path on phones', async ({page}) => {
  const modelRequests = [];
  const wasmRequests = [];
  let crashed = false;
  page.on('request', request => {
    const pathname = new URL(request.url()).pathname;
    if (pathname.includes('/models/')) modelRequests.push(request.url());
    if (pathname.endsWith('.wasm')) wasmRequests.push(request.url());
  });
  page.on('crash', () => {
    crashed = true;
  });

  await page.goto('/');
  await expect(page.locator('#allImage')).toHaveAttribute('src', /^blob:/);
  await expect.poll(() => page.locator('#allImage').evaluate(image => image.naturalWidth)).toBeGreaterThan(0);

  expect(await page.evaluate(() => navigator.userAgent)).toContain('iPhone');
  expect(modelRequests.some(url => url.endsWith('/generator.onnx'))).toBe(true);
  expect(modelRequests.some(url => url.endsWith('/quality-scorer.onnx'))).toBe(false);
  expect(wasmRequests).toHaveLength(1);
  expect(wasmRequests[0]).not.toContain('asyncify');

  // Give a delayed WebProcess/browser crash time to surface after the first render.
  await page.waitForTimeout(5000);
  expect(crashed).toBe(false);
  await expect(page.locator('main')).toBeVisible();
});
