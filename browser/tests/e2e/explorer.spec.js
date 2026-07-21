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
    const delay = route.request().url().endsWith('/generator.onnx') ? 3200 : 600;
    await new Promise(resolve => setTimeout(resolve, delay));
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
  await expect(page.locator('#allStatus .slow-generation-note')).toHaveCount(0);
  await expect(page.locator('#allStatus .loading-dot')).toHaveCount(5);
  await expect.poll(() => page.locator('#allStatus .loading-dot').first().evaluate(
    element => getComputedStyle(element).animationDuration,
  )).toBe('2.8s');
  const dotAnimation = await page.locator('#allStatus .loading-dots').evaluate(element => {
    const dots = [...element.querySelectorAll('.loading-dot')];
    const animations = dots.map(dot => dot.getAnimations()[0]);
    animations.forEach(animation => animation.pause());
    const opacityAt = milliseconds => {
      animations.forEach(animation => {
        animation.currentTime = milliseconds;
      });
      return dots.map(dot => Number(getComputedStyle(dot).opacity));
    };
    return {
      revealing: opacityAt(1400),
      allVisible: opacityAt(1750),
      removingLast: opacityAt(2050),
      removingPrevious: opacityAt(2180),
    };
  });
  expect(dotAnimation.revealing).toEqual([1, 1, 1, 1, 0]);
  expect(dotAnimation.allVisible).toEqual([1, 1, 1, 1, 1]);
  expect(dotAnimation.removingLast).toEqual([1, 1, 1, 1, 0]);
  expect(dotAnimation.removingPrevious).toEqual([1, 1, 1, 0, 0]);
  await expect(page.locator('#allStatus .slow-generation-note')).toHaveText(
    "Hmm, that's weird. It loaded faster on my machine",
    {timeout: 2700},
  );
  const allImage = await waitForGeneratedImage(page, '#allImage');
  await expect(page.locator('#allStatus')).not.toBeVisible();
  await expect(page.locator('#allStatus .slow-generation-note')).toHaveCount(0);
  await expect.poll(() => allImage.evaluate(element => element.naturalHeight)).toBeGreaterThan(800);
  await page.screenshot({path: testInfo.outputPath('all-digits.png'), fullPage: true});

  const allTab = page.locator('[data-panel="allPanel"]');
  const oneTab = page.locator('[data-panel="onePanel"]');
  const allTabBefore = await elementGeometry(allTab);
  const oneTabBefore = await elementGeometry(oneTab);
  await page.evaluate(() => {
    const randomValues = [0.7, 0.4, 0.25, 0.6, 0.8];
    Math.random = () => randomValues.shift() ?? 0;
  });
  await oneTab.click();
  await expect(page.locator('#oneDigit')).toHaveValue('7');
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

  const previousOneImage = await oneImage.getAttribute('src');
  await page.locator('#oneZoom').evaluate(element => {
    element.value = '0';
    element.dispatchEvent(new Event('input', {bubbles: true}));
    element.dispatchEvent(new Event('change', {bubbles: true}));
  });
  await expect(page.locator('#oneZoomValue')).toHaveText('10%');
  await expect.poll(() => oneImage.getAttribute('src')).not.toBe(previousOneImage);
  await expect.poll(() => oneImage.evaluate(element => element.naturalWidth)).toBe(719);

  const exploreTab = page.locator('[data-panel="explorePanel"]');
  await exploreTab.click();
  await expect(page.locator('#exploreDigit')).toHaveValue('4');
  await expect(page.locator('#exploreSeed')).toHaveValue('536870912');
  const exploreImage = await waitForGeneratedImage(page, '#exploreImage');
  await expect.poll(() => exploreImage.evaluate(element => element.naturalWidth)).toBe(280);
  const firstExploreImage = await exploreImage.getAttribute('src');
  await exploreTab.click();
  await expect(page.locator('#exploreDigit')).toHaveValue('6');
  await expect(page.locator('#exploreSeed')).toHaveValue('1717986918');
  await expect.poll(() => exploreImage.getAttribute('src')).not.toBe(firstExploreImage);
  await page.locator('#pad').press('ArrowRight');
  await expect(page.locator('#coords')).toContainText('x 0.17');
  await page.screenshot({path: testInfo.outputPath('latent-explorer.png'), fullPage: true});

  expect(apiRequests).toEqual([]);
  expect(consoleErrors).toEqual([]);
});

test('waits 400ms after All Digits and One Digit slider changes', async ({page}) => {
  const apiCalls = pathname => page.evaluate(path => (
    window.__localApiCalls.filter(call => call.pathname === path)
  ), pathname);
  const moveSlider = (selector, value, commit = false) => page.locator(selector).evaluate((element, update) => {
    element.value = update.value;
    element.dispatchEvent(new Event('input', {bubbles: true}));
    if (update.commit) element.dispatchEvent(new Event('change', {bubbles: true}));
    window.__lastSliderChange = performance.now();
  }, {value, commit});

  await page.goto('/');
  await waitForGeneratedImage(page, '#allImage');
  await page.evaluate(() => {
    window.__localApiCalls = [];
    const localFetch = window.fetch.bind(window);
    window.fetch = (input, init) => {
      const value = input instanceof Request ? input.url : input;
      const pathname = new URL(String(value), window.location.href).pathname;
      if (pathname.startsWith('/api/')) window.__localApiCalls.push({pathname, at: performance.now()});
      return localFetch(input, init);
    };
  });

  await moveSlider('#allSamples', '10');
  await page.waitForTimeout(200);
  await moveSlider('#allSamples', '11', true);
  await page.waitForTimeout(300);
  expect(await apiCalls('/api/all')).toEqual([]);
  await expect.poll(async () => (await apiCalls('/api/all')).length).toBe(1);
  expect((await apiCalls('/api/all'))[0].at - await page.evaluate(() => window.__lastSliderChange)).toBeGreaterThanOrEqual(375);

  await page.locator('[data-panel="onePanel"]').click();
  await waitForGeneratedImage(page, '#oneImage');
  await expect(page.locator('#oneSamples')).toHaveAttribute('max', '10000');
  await page.evaluate(() => {
    window.__localApiCalls = [];
  });
  await moveSlider('#oneSamples', '60');
  await page.waitForTimeout(200);
  await moveSlider('#oneSamples', '61', true);
  await page.waitForTimeout(300);
  expect(await apiCalls('/api/digit')).toEqual([]);
  await expect.poll(async () => (await apiCalls('/api/digit')).length).toBe(1);
  expect((await apiCalls('/api/digit'))[0].at - await page.evaluate(() => window.__lastSliderChange)).toBeGreaterThanOrEqual(375);

  const invalid = await page.evaluate(async () => {
    const response = await fetch('/api/digit?digit=3&samples=10001&seed=112&scale=1');
    return {status: response.status, body: await response.json()};
  });
  expect(invalid.status).toBe(400);
  expect(invalid.body.detail).toContain('samples must be an integer from 1 to 10000');
});

test('reports a local inference module startup failure', async ({page}) => {
  await page.route('**/assets/index-*.js', route => route.abort());

  await page.goto('/');

  await expect(page.locator('#allStatus')).toHaveClass(/error/);
  await expect(page.locator('#allStatus')).toContainText('Could not start local inference');
  await expect(page.locator('#allStatus')).not.toHaveClass(/generating/);
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
