# Browser-only explorer

This directory builds a static version of the MNIST WGAN-GP Explorer for GitHub Pages. The EMA
generator and optional quality filter run with ONNX Runtime inside the visitor's browser. Firefox
uses the generator output directly because its supported WebAssembly path makes the quality filter
much slower. Generated images, labels, seeds, and latent coordinates are never sent to an inference
server.

The browser application deliberately remains separate from the FastAPI application. During a
build, `scripts/prepare-index.mjs` reuses the established HTML and CSS from
`src/mnist_wgan/static/index.html`, then replaces its three image API calls with local inference.
This keeps both interfaces visually aligned without maintaining a second copy of the design.

## Develop locally

Node.js 24 is used by the deployment workflow.

```bash
npm install --prefix browser
npm run dev --prefix browser
```

Open the address printed by Vite. Build and test the production site with:

```bash
npm run test --prefix browser
npm run build --prefix browser
npm run test:e2e --prefix browser
```

The Playwright command requires its Chromium and Firefox runtimes (`npx --prefix browser playwright
install chromium firefox`) and the usual browser system libraries.

## Refresh the exported checkpoint

The committed ONNX files are deterministic exports of the bundled inference checkpoint:

```bash
uv run --group browser-export python scripts/export_browser_models.py
```

The exporter validates both graphs with ONNX Runtime and records model and checkpoint hashes in
`public/models/manifest.json`. Commit the refreshed model files and manifest together.

Browser seeds are stable within this application, but its small JavaScript Gaussian generator is
not PyTorch's random-number implementation. A numeric seed therefore does not select the exact
same latent vectors in the browser and Python applications.

## Publish

The `pages.yml` workflow builds and deploys this directory after relevant changes reach `main`.
In the GitHub repository settings, select **GitHub Actions** as the Pages source once; subsequent
deployments are automatic.
