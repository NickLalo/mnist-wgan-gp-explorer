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
    shade: new Float32Array(6),
    halo: new Float32Array(3),
    strokeShadeCvThresholds: Array(10).fill(1),
    strokeShadeDipThresholds: Array(10).fill(1),
    strokeHaloThresholds: Array(10).fill(1),
  });
  assert.deepEqual(selected, [1, 2]);
});

test('an excessive centerline shade dip is replaced by a backup candidate', () => {
  const labels = [2, 2, 2];
  const logits = new Float32Array(30);
  logits[2] = 10;
  logits[12] = 10;
  logits[22] = 10;
  const shade = new Float32Array(6);
  shade[1] = 0.2;
  const selected = selectQualityIndices({
    labels,
    keepPerClass: 2,
    criticScores: new Float32Array([2, 1, 0]),
    logits,
    unsupported: new Float32Array(3),
    disconnected: new Float32Array(3),
    profiles: new Float32Array(12),
    shade,
    halo: new Float32Array(3),
    strokeShadeCvThresholds: Array(10).fill(1),
    strokeShadeDipThresholds: Array(10).fill(0.1),
    strokeShadeRejectionMultiplier: 1,
    strokeHaloThresholds: Array(10).fill(1),
  });
  assert.deepEqual(selected, [1, 2]);
});

test('an excessive pale outer halo is replaced by a backup candidate', () => {
  const labels = [6, 6, 6];
  const logits = new Float32Array(30);
  logits[6] = 10;
  logits[16] = 10;
  logits[26] = 10;
  const selected = selectQualityIndices({
    labels,
    keepPerClass: 2,
    criticScores: new Float32Array([2, 1, 0]),
    logits,
    unsupported: new Float32Array(3),
    disconnected: new Float32Array(3),
    profiles: new Float32Array(12),
    shade: new Float32Array(6),
    halo: new Float32Array([0.2, 0, 0]),
    strokeShadeCvThresholds: Array(10).fill(1),
    strokeShadeDipThresholds: Array(10).fill(1),
    strokeHaloThresholds: Array(10).fill(0.1),
  });
  assert.deepEqual(selected, [1, 2]);
});
