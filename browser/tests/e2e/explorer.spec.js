import {expect, test} from '@playwright/test';

async function waitForGeneratedImage(page, selector) {
  const image = page.locator(selector);
  await expect(image).toHaveAttribute('src', /^blob:/);
  await expect.poll(() => image.evaluate(element => element.naturalWidth)).toBeGreaterThan(0);
  return image;
}

async function elementGeometry(locator) {
  return locator.evaluate(element => {
    const {x, y, width, height} = element.getBoundingClientRect();
    return {x, y, width, height};
  });
}

test('all three modes generate locally without API network traffic', async ({page}, testInfo) => {
  const apiRequests = [];
  const consoleErrors = [];
  await page.route('**/models/*.onnx', async route => {
    await new Promise(resolve => setTimeout(resolve, 600));
    await route.continue();
  });
  page.on('request', request => {
    if (new URL(request.url()).pathname.includes('/api/')) apiRequests.push(request.url());
  });
  page.on('console', message => {
    if (message.type() === 'error') consoleErrors.push(message.text());
  });

  await page.goto('/');
  await expect(page.locator('script[src*="cloudflareinsights"]')).toHaveCount(0);
  await expect(page.locator('#allStatus')).toBeVisible();
  await expect(page.locator('#allStatus')).toHaveClass(/generating/);
  await expect(page.locator('#allStatus')).toContainText('Generating Numbers');
  const allImage = await waitForGeneratedImage(page, '#allImage');
  await expect(page.locator('#allStatus')).not.toBeVisible();
  await expect.poll(() => allImage.evaluate(element => element.naturalHeight)).toBeGreaterThan(800);
  await page.screenshot({path: testInfo.outputPath('all-digits.png'), fullPage: true});

  const allTab = page.locator('[data-panel="allPanel"]');
  const oneTab = page.locator('[data-panel="onePanel"]');
  const allTabBefore = await elementGeometry(allTab);
  const oneTabBefore = await elementGeometry(oneTab);
  await oneTab.click();
  expect(await elementGeometry(allTab)).toEqual(allTabBefore);
  expect(await elementGeometry(oneTab)).toEqual(oneTabBefore);
  await page.locator('#oneSamples').evaluate(element => {
    element.value = '60';
    element.dispatchEvent(new Event('input', {bubbles: true}));
    element.dispatchEvent(new Event('change', {bubbles: true}));
  });
  const oneImage = await waitForGeneratedImage(page, '#oneImage');
  await expect.poll(() => oneImage.evaluate(element => element.naturalWidth)).toBeGreaterThan(1000);
  await page.screenshot({path: testInfo.outputPath('one-digit.png'), fullPage: true});

  await page.locator('[data-panel="explorePanel"]').click();
  const exploreImage = await waitForGeneratedImage(page, '#exploreImage');
  await expect.poll(() => exploreImage.evaluate(element => element.naturalWidth)).toBe(280);
  await page.locator('#pad').press('ArrowRight');
  await expect(page.locator('#coords')).toContainText('x 0.17');
  await page.screenshot({path: testInfo.outputPath('latent-explorer.png'), fullPage: true});

  expect(apiRequests).toEqual([]);
  expect(consoleErrors).toEqual([]);
});

test.describe('phone layout', () => {
  test.use({viewport: {width: 390, height: 844}, isMobile: true, hasTouch: true});

  test('keeps controls and both latent explorer panels within the viewport', async ({page}) => {
    const fitsViewport = selector => page.locator(selector).evaluate(element => {
      const rect = element.getBoundingClientRect();
      return rect.left >= 0 && rect.right <= window.innerWidth + 1 && element.scrollWidth <= element.clientWidth + 1;
    });

    await page.goto('/');
    await waitForGeneratedImage(page, '#allImage');
    await expect(page.locator('#allImage')).toHaveAttribute('draggable', 'false');
    await expect(page.locator('#allImage')).toHaveCSS('user-select', 'none');
    expect(await fitsViewport('#allPanel .controls')).toBe(true);
    await expect(page.locator('#allPanel .output')).toHaveClass(/all-grid-fits/);
    const allOutput = await elementGeometry(page.locator('#allPanel .output'));
    const allImage = await elementGeometry(page.locator('#allImage'));
    expect(allImage.x).toBeGreaterThanOrEqual(allOutput.x);
    expect(allImage.x + allImage.width).toBeLessThanOrEqual(allOutput.x + allOutput.width);
    await expect.poll(() => page.locator('#allPanel .output').evaluate(element => element.scrollWidth - element.clientWidth)).toBeLessThanOrEqual(0);
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);

    await page.locator('[data-panel="onePanel"]').click();
    await waitForGeneratedImage(page, '#oneImage');
    await expect(page.locator('#oneImage')).toHaveAttribute('draggable', 'false');
    await expect(page.locator('#oneImage')).toHaveCSS('user-select', 'none');
    expect(await fitsViewport('#onePanel .controls')).toBe(true);

    await page.locator('[data-panel="explorePanel"]').click();
    await waitForGeneratedImage(page, '#exploreImage');
    await expect(page.locator('#exploreImage')).toHaveAttribute('draggable', 'false');
    await expect(page.locator('#exploreImage')).toHaveCSS('user-select', 'none');
    expect(await fitsViewport('#explorePanel .controls')).toBe(true);
    expect(await fitsViewport('#exploreLayout')).toBe(true);

    const latentModule = await elementGeometry(page.locator('.latent-module'));
    const outputModule = await elementGeometry(page.locator('.output-module'));
    const pad = await elementGeometry(page.locator('#pad'));
    const sample = await elementGeometry(page.locator('#exploreImage'));
    expect(outputModule.y).toBeGreaterThanOrEqual(latentModule.y + latentModule.height);
    expect(pad.width).toBeGreaterThan(240);
    expect(sample.width).toBeGreaterThan(300);
    await expect(page.locator('#newPlane')).toHaveCSS('min-height', '42px');

    const initialSampleSource = await page.locator('#exploreImage').getAttribute('src');
    const client = await page.context().newCDPSession(page);
    const centerX = pad.x + pad.width / 2;
    const centerY = pad.y + pad.height / 2;
    await client.send('Input.dispatchTouchEvent', {type: 'touchStart', touchPoints: [
      {x: centerX + 40, y: centerY, id: 1, radiusX: 4, radiusY: 4, force: 1},
    ]});
    await client.send('Input.dispatchTouchEvent', {type: 'touchEnd', touchPoints: []});
    await expect(page.locator('#coords')).not.toContainText('x 0.00');
    await expect.poll(() => page.locator('#exploreImage').getAttribute('src')).not.toBe(initialSampleSource);
    const sampleSource = await page.locator('#exploreImage').getAttribute('src');
    await client.send('Input.dispatchTouchEvent', {type: 'touchStart', touchPoints: [
      {x: centerX - 40, y: centerY, id: 1, radiusX: 4, radiusY: 4, force: 1},
      {x: centerX + 40, y: centerY, id: 2, radiusX: 4, radiusY: 4, force: 1},
    ]});
    await client.send('Input.dispatchTouchEvent', {type: 'touchMove', touchPoints: [
      {x: centerX - 70, y: centerY, id: 1, radiusX: 4, radiusY: 4, force: 1},
      {x: centerX + 70, y: centerY, id: 2, radiusX: 4, radiusY: 4, force: 1},
    ]});
    await client.send('Input.dispatchTouchEvent', {type: 'touchEnd', touchPoints: []});
    await expect.poll(async () => Number(await page.locator('#areaLimit').inputValue())).toBeLessThan(1.5);
    await expect(page.locator('#coords')).not.toContainText('bounds ±1.50');
    await expect(page.locator('#exploreImage')).toHaveAttribute('src', sampleSource);
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  });
});
