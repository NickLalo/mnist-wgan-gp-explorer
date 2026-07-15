function splitMix32(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x9e3779b9) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 16), 0x21f0aaad);
    value = Math.imul(value ^ (value >>> 15), 0x735a2d97);
    return (value ^ (value >>> 15)) >>> 0;
  };
}

export function seededUniform(seed) {
  const split = splitMix32(Number(seed) >>> 0);
  let first = split(), second = split(), third = split(), fourth = split();
  return () => {
    first >>>= 0; second >>>= 0; third >>>= 0; fourth >>>= 0;
    const result = (first + second + fourth) >>> 0;
    fourth = (fourth + 1) >>> 0;
    first = (second ^ (second >>> 9)) >>> 0;
    second = (third + (third << 3)) >>> 0;
    third = ((third << 21) | (third >>> 11)) >>> 0;
    third = (third + result) >>> 0;
    return result / 0x100000000;
  };
}

export function gaussianNoise(count, latentDimension, seed) {
  const uniform = seededUniform(seed);
  const values = new Float32Array(count * latentDimension);
  let index = 0;
  while (index < values.length) {
    let first, second, radius;
    do {
      first = 2 * uniform() - 1;
      second = 2 * uniform() - 1;
      radius = first * first + second * second;
    } while (radius === 0 || radius >= 1);
    const scale = Math.sqrt(-2 * Math.log(radius) / radius);
    values[index++] = first * scale;
    if (index < values.length) values[index++] = second * scale;
  }
  return values;
}

function dot(first, second) {
  let total = 0;
  for (let index = 0; index < first.length; index += 1) total += first[index] * second[index];
  return total;
}

function subtractProjection(vector, basis) {
  const factor = dot(vector, basis) / Math.max(dot(basis, basis), 1e-8);
  for (let index = 0; index < vector.length; index += 1) vector[index] -= factor * basis[index];
}

function normalize(vector, radius = 1) {
  const norm = Math.max(Math.sqrt(dot(vector, vector)), 1e-8);
  for (let index = 0; index < vector.length; index += 1) vector[index] *= radius / norm;
}

export function latentPlane(seed, latentDimension) {
  const values = gaussianNoise(3, latentDimension, seed);
  const base = values.slice(0, latentDimension);
  const horizontal = values.slice(latentDimension, 2 * latentDimension);
  const vertical = values.slice(2 * latentDimension);
  subtractProjection(horizontal, base);
  normalize(horizontal);
  subtractProjection(vertical, base);
  subtractProjection(vertical, horizontal);
  const radius = Math.sqrt(latentDimension);
  normalize(horizontal, radius);
  normalize(vertical, radius);
  return {base, horizontal, vertical};
}
