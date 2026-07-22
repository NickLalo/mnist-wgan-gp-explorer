import {devices, expect, test} from '@playwright/test';

const iphone = devices['iPhone 13'];

test.use({
  viewport: iphone.viewport,
  deviceScaleFactor: iphone.deviceScaleFactor,
  isMobile: iphone.isMobile,
  hasTouch: iphone.hasTouch,
  userAgent: iphone.userAgent,
});

test('uses compact generator and critic inference on phones', async ({page}) => {
  const modelRequests = [];
  const wasmRequests = [];
  let crashed = false;
  await page.route('**/models/generator-uint8.onnx', async route => {
    await new Promise(resolve => setTimeout(resolve, 1600));
    await route.continue();
  });
  page.on('request', request => {
    const pathname = new URL(request.url()).pathname;
    if (pathname.includes('/models/')) modelRequests.push(request.url());
    if (pathname.endsWith('.wasm')) wasmRequests.push(request.url());
  });
  page.on('crash', () => {
    crashed = true;
  });

  await page.goto('/');
  const dots = page.locator('#allStatus .loading-dot');
  await expect(dots).toHaveCount(5);
  const dotLayout = await page.locator('#allStatus .loading-dots').evaluate(element => {
    const children = [...element.children];
    return {
      widths: children.map(dot => dot.getBoundingClientRect().width),
      opacities: children.map(dot => Number(getComputedStyle(dot).opacity)),
      overflow: element.scrollWidth - element.clientWidth,
    };
  });
  expect(new Set(dotLayout.widths.map(width => width.toFixed(2))).size).toBe(1);
  expect(dotLayout.opacities.every(opacity => opacity === 0 || opacity === 1)).toBe(true);
  expect(dotLayout.overflow).toBeLessThanOrEqual(0);
  await expect(page.locator('#allImage')).toHaveAttribute('src', /^blob:/);
  await expect.poll(() => page.locator('#allImage').evaluate(image => image.naturalWidth)).toBeGreaterThan(0);

  expect(await page.evaluate(() => navigator.userAgent)).toContain('iPhone');
  expect(modelRequests.some(url => url.endsWith('/generator-uint8.onnx'))).toBe(true);
  expect(modelRequests.some(url => url.endsWith('/quality-scorer-uint8.onnx'))).toBe(true);
  expect(wasmRequests).toHaveLength(1);
  expect(wasmRequests[0]).not.toContain('asyncify');

  // Give a delayed WebProcess/browser crash time to surface after the first render.
  await page.waitForTimeout(5000);
  expect(crashed).toBe(false);
  await expect(page.locator('main')).toBeVisible();

  await page.locator('[data-panel="onePanel"]').click();
  await expect(page.locator('#oneImage')).toHaveAttribute('src', /^blob:/);
  await expect(page.locator('#oneSamples')).toHaveAttribute('max', '10000');
  await page.evaluate(() => {
    const localFetch = window.fetch.bind(window);
    window.fetch = (input, init) => {
      const value = input instanceof Request ? input.url : input;
      if (new URL(String(value), window.location.href).pathname.endsWith('/api/digit')) {
        return Promise.reject(new DOMException('Test request stopped', 'AbortError'));
      }
      return localFetch(input, init);
    };
  });
  await page.locator('#oneSamples').evaluate(element => {
    element.value = '10000';
    element.dispatchEvent(new Event('input', {bubbles: true}));
    element.dispatchEvent(new Event('change', {bubbles: true}));
  });
  await expect(page.locator('#oneZoomValue')).toHaveText('25%', {timeout: 1200});
});
