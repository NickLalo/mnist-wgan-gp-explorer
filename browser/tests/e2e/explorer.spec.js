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
