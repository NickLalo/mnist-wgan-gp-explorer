import test from 'node:test';
import assert from 'node:assert/strict';

import {candidateCount, selectQualityIndices} from '../src/quality.js';

test('candidate pools match the Python sampling policy', () => {
  assert.equal(candidateCount(10), 14);
  assert.equal(candidateCount(100), 120);
});

test('an explicit class failure is replaced by the best backup candidate', () => {
  const labels = [3, 3, 3];
  const logits = new Float32Array(30);
  logits[0] = 10;
  logits[13] = 10;
  logits[23] = 12;
  const selected = selectQualityIndices({
    labels,
    keepPerClass: 2,
    criticScores: new Float32Array([0, 1, 2]),
    logits,
    unsupported: new Float32Array(3),
    disconnected: new Float32Array(3),
    profiles: new Float32Array(12),
  });
  assert.deepEqual(selected, [1, 2]);
});
