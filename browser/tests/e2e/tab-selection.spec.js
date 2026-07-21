import {devices, expect, test} from '@playwright/test';

const iphone = devices['iPhone 13'];

test.use({
  viewport: iphone.viewport,
  deviceScaleFactor: iphone.deviceScaleFactor,
  isMobile: iphone.isMobile,
  hasTouch: iphone.hasTouch,
  userAgent: iphone.userAgent,
});

test('tab labels cannot be selected on phones', async ({page}) => {
  await page.goto('/', {waitUntil: 'domcontentloaded'});

  const selectionStyles = await page.locator('.tab').evaluateAll(tabs => tabs.map(tab => {
    const style = getComputedStyle(tab);
    return {
      userSelect: style.userSelect,
      webkitUserSelect: style.webkitUserSelect,
    };
  }));

  expect(selectionStyles).toHaveLength(3);
  expect(selectionStyles).toEqual(selectionStyles.map(() => ({
    userSelect: 'none',
    webkitUserSelect: 'none',
  })));
});
