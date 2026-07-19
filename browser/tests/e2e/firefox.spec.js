import {expect, test} from '@playwright/test';

test('generates promptly without Firefox WebGPU or the slow quality pass', async ({page}) => {
  const modelRequests = [];
  const wasmRequests = [];
  const consoleErrors = [];
  page.on('request', request => {
    if (new URL(request.url()).pathname.includes('/models/')) modelRequests.push(request.url());
    if (new URL(request.url()).pathname.endsWith('.wasm')) wasmRequests.push(request.url());
  });
  page.on('console', message => {
    if (message.type() === 'error') consoleErrors.push(message.text());
  });

  await page.goto('/');
  await expect.poll(() => page.evaluate(() => Boolean(navigator.gpu))).toBe(true);
  await expect(page.locator('#allImage')).toHaveAttribute('src', /^blob:/);
  await expect.poll(() => page.locator('#allImage').evaluate(image => image.naturalWidth)).toBeGreaterThan(0);

  expect(modelRequests.some(url => url.endsWith('/generator.onnx'))).toBe(true);
  expect(modelRequests.some(url => url.endsWith('/quality-scorer.onnx'))).toBe(false);
  expect(wasmRequests).toHaveLength(1);
  expect(wasmRequests[0]).not.toContain('asyncify');
  expect(consoleErrors).toEqual([]);
});
