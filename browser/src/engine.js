import {selectQualityIndices, candidateCount} from './quality.js';
import {gaussianNoise, latentPlane} from './random.js';
import {renderAllDigits, renderOneDigit, renderSingleImage} from './render.js';
import {createSerialExecutor} from './serial.js';

const assetPath = path => `${import.meta.env.BASE_URL}${path}`;
const isFirefox = navigator.userAgent.includes('Firefox/');
const isMobile = navigator.userAgentData?.mobile === true
  || /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent);
const useWebGpu = Boolean(navigator.gpu) && !isFirefox && !isMobile;
const runtimePromise = (
  useWebGpu ? import('onnxruntime-web/webgpu') : import('onnxruntime-web/wasm')
).then(runtime => {
  runtime.env.logLevel = 'error';
  runtime.env.wasm.numThreads = 1;
  return runtime;
});

let manifestPromise;
let generatorPromise;
let scorerPromise;
const runSerially = createSerialExecutor();

async function manifest() {
  manifestPromise ??= window.__networkFetch(assetPath('models/manifest.json')).then(response => {
    if (!response.ok) throw new Error(`Could not load model manifest (${response.status})`);
    return response.json();
  });
  return manifestPromise;
}

async function createSession(modelPath) {
  const ort = await runtimePromise;
  const providers = useWebGpu ? ['webgpu', 'wasm'] : ['wasm'];
  try {
    return await ort.InferenceSession.create(assetPath(`models/${modelPath}`), {
      executionProviders: providers,
      graphOptimizationLevel: 'all',
    });
  } catch (error) {
    if (providers.length === 1) throw error;
    console.warn('WebGPU initialization failed; using WebAssembly instead.', error);
    return ort.InferenceSession.create(assetPath(`models/${modelPath}`), {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
  }
}

function modelPath(model) {
  return useWebGpu ? model.path : (model.wasm_path ?? model.path);
}

async function generatorSession() {
  if (!generatorPromise) {
    generatorPromise = manifest().then(data => createSession(modelPath(data.models.generator)));
  }
  return generatorPromise;
}

async function scorerSession() {
  if (!scorerPromise) {
    scorerPromise = manifest().then(data => createSession(modelPath(data.models.quality_scorer)));
  }
  return scorerPromise;
}

function throwIfAborted(signal) {
  if (signal?.aborted) throw new DOMException('The operation was aborted.', 'AbortError');
}

function oneHot(labels, start, stop) {
  const values = new Float32Array((stop - start) * 10);
  for (let index = start; index < stop; index += 1) values[(index - start) * 10 + labels[index]] = 1;
  return values;
}

async function generate(noise, labels, latentDimension, signal) {
  const ort = await runtimePromise;
  const session = await generatorSession();
  const images = new Float32Array(labels.length * 28 * 28);
  const batchSize = 256;
  for (let start = 0; start < labels.length; start += batchSize) {
    throwIfAborted(signal);
    const stop = Math.min(start + batchSize, labels.length);
    const result = await session.run({
      noise: new ort.Tensor('float32', noise.slice(start * latentDimension, stop * latentDimension), [stop - start, latentDimension]),
      label_one_hot: new ort.Tensor('float32', oneHot(labels, start, stop), [stop - start, 10]),
    });
    images.set(result.images.data, start * 28 * 28);
    await new Promise(resolve => setTimeout(resolve, 0));
  }
  return images;
}

async function score(images, labels, signal) {
  const ort = await runtimePromise;
  const session = await scorerSession();
  const outputs = {
    criticScores: new Float32Array(labels.length),
    logits: new Float32Array(labels.length * 10),
    unsupported: new Float32Array(labels.length),
    disconnected: new Float32Array(labels.length),
    profiles: new Float32Array(labels.length * 4),
    shade: new Float32Array(labels.length * 2),
    halo: new Float32Array(labels.length),
  };
  const batchSize = 256;
  for (let start = 0; start < labels.length; start += batchSize) {
    throwIfAborted(signal);
    const stop = Math.min(start + batchSize, labels.length);
    const result = await session.run({
      images: new ort.Tensor('float32', images.slice(start * 28 * 28, stop * 28 * 28), [stop - start, 1, 28, 28]),
      label_one_hot: new ort.Tensor('float32', oneHot(labels, start, stop), [stop - start, 10]),
    });
    outputs.criticScores.set(result.critic_scores.data, start);
    outputs.logits.set(result.logits.data, start * 10);
    outputs.unsupported.set(result.unsupported.data, start);
    outputs.disconnected.set(result.disconnected.data, start);
    outputs.profiles.set(result.profiles.data, start * 4);
    outputs.shade.set(result.shade.data, start * 2);
    outputs.halo.set(result.halo.data, start);
    await new Promise(resolve => setTimeout(resolve, 0));
  }
  return outputs;
}

async function qualityGenerate(classes, requestedPerClass, seed, signal) {
  const settings = await manifest();
  // CPU-only browsers use compact uint8 graphs. Their combined generator and
  // critic download is smaller than the old generator-only float32 path, so
  // phones and Firefox can now run the same selective quality pass as WebGPU.
  const perClass = candidateCount(requestedPerClass, settings.sampling.quality_oversample);
  const labels = classes.flatMap(digit => Array(perClass).fill(digit));
  const noise = gaussianNoise(labels.length, settings.latent_dim, seed);
  const candidates = await generate(noise, labels, settings.latent_dim, signal);
  const scoring = await score(candidates, labels, signal);
  const selected = selectQualityIndices({
    labels,
    keepPerClass: requestedPerClass,
    ...scoring,
    rejectionThreshold: settings.sampling.quality_rejection_threshold,
    detachedInkThreshold: settings.sampling.detached_ink_threshold,
    unsupportedInkThreshold: settings.sampling.unsupported_ink_threshold,
    strokeShadeCvThresholds: settings.sampling.stroke_shade_cv_thresholds,
    strokeShadeDipThresholds: settings.sampling.stroke_shade_dip_thresholds,
    strokeShadeRejectionMultiplier: settings.sampling.stroke_shade_rejection_multiplier,
    strokeHaloThresholds: settings.sampling.stroke_halo_thresholds,
  });
  const images = new Float32Array(selected.length * 28 * 28);
  selected.forEach((candidate, outputIndex) => {
    images.set(candidates.subarray(candidate * 28 * 28, (candidate + 1) * 28 * 28), outputIndex * 28 * 28);
  });
  return {images, settings};
}

function integerParameter(parameters, name, minimum, maximum) {
  const value = Number(parameters.get(name));
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    throw new Error(`${name} must be an integer from ${minimum} to ${maximum}`);
  }
  return value;
}

function floatParameter(parameters, name, minimum, maximum) {
  const value = Number(parameters.get(name));
  if (!Number.isFinite(value) || value < minimum || value > maximum) {
    throw new Error(`${name} must be a number from ${minimum} to ${maximum}`);
  }
  return value;
}

async function allDigits(parameters, signal) {
  const samples = integerParameter(parameters, 'samples', 1, 100);
  const seed = integerParameter(parameters, 'seed', 0, 2 ** 31 - 1);
  const {images, settings} = await qualityGenerate([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], samples, seed, signal);
  return renderAllDigits(images, samples, settings.rendering.paper_color, settings.rendering.ink_color);
}

async function oneDigit(parameters, signal) {
  const digit = integerParameter(parameters, 'digit', 0, 9);
  const samples = integerParameter(parameters, 'samples', 1, 10000);
  const seed = integerParameter(parameters, 'seed', 0, 2 ** 31 - 1);
  const scale = floatParameter(parameters, 'scale', 0.4, 4);
  const {images, settings} = await qualityGenerate([digit], samples, seed, signal);
  return renderOneDigit(images, samples, scale, settings.rendering.paper_color, settings.rendering.ink_color);
}

async function explore(parameters, signal) {
  const digit = integerParameter(parameters, 'digit', 0, 9);
  const seed = integerParameter(parameters, 'seed', 0, 2 ** 31 - 1);
  const x = floatParameter(parameters, 'x', -10, 10);
  const y = floatParameter(parameters, 'y', -10, 10);
  const settings = await manifest();
  const plane = latentPlane(seed, settings.latent_dim);
  const noise = new Float32Array(settings.latent_dim);
  for (let index = 0; index < noise.length; index += 1) {
    noise[index] = plane.base[index] + x * plane.horizontal[index] + y * plane.vertical[index];
  }
  const images = await generate(noise, [digit], settings.latent_dim, signal);
  return renderSingleImage(images, settings.rendering.paper_color, settings.rendering.ink_color);
}

function errorResponse(error) {
  return new Response(JSON.stringify({detail: error.message}), {
    status: 400,
    headers: {'Content-Type': 'application/json'},
  });
}

export async function handleApiRequest(input, init = {}) {
  const signal = init.signal ?? (input instanceof Request ? input.signal : undefined);
  return runSerially(async () => {
    try {
      throwIfAborted(signal);
      const url = new URL(input instanceof Request ? input.url : input, window.location.href);
      if (url.pathname.endsWith('/api/all')) return await allDigits(url.searchParams, signal);
      if (url.pathname.endsWith('/api/digit')) return await oneDigit(url.searchParams, signal);
      if (url.pathname.endsWith('/api/explore')) return await explore(url.searchParams, signal);
      return new Response(JSON.stringify({detail: 'Unknown local endpoint'}), {status: 404});
    } catch (error) {
      if (error.name === 'AbortError') throw error;
      console.error(error);
      return errorResponse(error);
    }
  }, signal);
}
