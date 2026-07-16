import assert from 'node:assert/strict';
import test from 'node:test';

import {createSerialExecutor} from '../src/serial.js';

test('serial executor never overlaps tasks', async () => {
  const runSerially = createSerialExecutor();
  let active = 0;
  let maximumActive = 0;
  const order = [];
  const task = value => runSerially(async () => {
    active += 1;
    maximumActive = Math.max(maximumActive, active);
    order.push(`start-${value}`);
    await new Promise(resolve => setTimeout(resolve, 5));
    order.push(`end-${value}`);
    active -= 1;
    return value;
  });

  assert.deepEqual(await Promise.all([task(1), task(2), task(3)]), [1, 2, 3]);
  assert.equal(maximumActive, 1);
  assert.deepEqual(order, ['start-1', 'end-1', 'start-2', 'end-2', 'start-3', 'end-3']);
});

test('serial executor skips an aborted task while it waits', async () => {
  const runSerially = createSerialExecutor();
  let releaseFirst;
  const first = runSerially(() => new Promise(resolve => { releaseFirst = resolve; }));
  const controller = new AbortController();
  let secondStarted = false;
  const second = runSerially(() => { secondStarted = true; }, controller.signal);

  controller.abort();
  await Promise.resolve();
  releaseFirst();
  await first;
  await assert.rejects(second, error => error.name === 'AbortError');
  assert.equal(secondStarted, false);
});
