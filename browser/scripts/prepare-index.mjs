import {copyFile, mkdir, readFile, rm, writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const browserDirectory = path.resolve(scriptDirectory, '..');
const repositoryRoot = path.resolve(browserDirectory, '..');
const sourceHtml = path.join(repositoryRoot, 'src/mnist_wgan/static/index.html');
const sourceFavicon = path.join(repositoryRoot, 'src/mnist_wgan/static/favicon.svg');
const publicDirectory = path.join(browserDirectory, 'public');

const interceptionBootstrap = `
  <script>
    (() => {
      const networkFetch = window.fetch.bind(window);
      let localHandler = null;
      let startupError = null;
      const pending = [];
      const isLocalApi = input => {
        const value = input instanceof Request ? input.url : input;
        const path = new URL(String(value), window.location.href).pathname;
        return /\\/api\\/(all|digit|explore)$/.test(path);
      };
      const settle = () => {
        while (pending.length) {
          const {input, init, resolve, reject} = pending.shift();
          if (startupError) reject(startupError);
          else localHandler(input, init).then(resolve, reject);
        }
      };
      window.__networkFetch = networkFetch;
      window.__installLocalInference = handler => { localHandler = handler; settle(); };
      window.__failLocalInference = error => { startupError = error; settle(); };
      window.fetch = (input, init = {}) => {
        if (!isLocalApi(input)) return networkFetch(input, init);
        if (startupError) return Promise.reject(startupError);
        if (localHandler) return localHandler(input, init);
        return new Promise((resolve, reject) => pending.push({input, init, resolve, reject}));
      };
    })();
  </script>
  <script type="module" src="/src/main.js"></script>`;

let html = await readFile(sourceHtml, 'utf8');
html = html.replace(
  '<link rel="icon" href="/favicon.ico?v=2" type="image/x-icon" sizes="any">',
  '<link rel="icon" href="./favicon.svg?v=3" type="image/svg+xml">',
);
html = html.replace('</head>', `${interceptionBootstrap}\n</head>`);

await mkdir(publicDirectory, {recursive: true});
await rm(path.join(publicDirectory, 'ort'), {recursive: true, force: true});
await copyFile(sourceFavicon, path.join(publicDirectory, 'favicon.svg'));
await writeFile(path.join(browserDirectory, 'index.html'), html);
