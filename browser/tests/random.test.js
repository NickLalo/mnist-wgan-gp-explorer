import test from 'node:test';
import assert from 'node:assert/strict';

import {gaussianNoise, latentPlane} from '../src/random.js';

test('seeded Gaussian noise is deterministic and changes with the seed', () => {
  const first = gaussianNoise(2, 8, 112);
  assert.deepEqual(first, gaussianNoise(2, 8, 112));
  assert.notDeepEqual(first, gaussianNoise(2, 8, 113));
});

test('latent directions are orthogonal and have the expected radius', () => {
  const {base, horizontal, vertical} = latentPlane(112, 96);
  const dot = (left, right) => left.reduce((total, value, index) => total + value * right[index], 0);
  assert.ok(Math.abs(dot(base, horizontal)) < 1e-4);
  assert.ok(Math.abs(dot(base, vertical)) < 1e-4);
  assert.ok(Math.abs(dot(horizontal, vertical)) < 1e-4);
  assert.ok(Math.abs(Math.sqrt(dot(horizontal, horizontal)) - Math.sqrt(96)) < 1e-5);
  assert.ok(Math.abs(Math.sqrt(dot(vertical, vertical)) - Math.sqrt(96)) < 1e-5);
});
