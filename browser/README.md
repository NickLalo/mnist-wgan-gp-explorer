# Browser-only explorer

This directory builds a static version of the MNIST WGAN-GP Explorer for GitHub Pages. The EMA
generator and quality scorer run with ONNX Runtime inside the visitor's browser. WebGPU browsers
use the float32 graphs; Firefox and mobile browsers use static uint8 exports. The quality scorer
keeps its small, rank-sensitive final critic block in float32 while quantizing the earlier
convolutions and classifier. The two compact CPU graphs total about 3.69 MB, 58% less than their
float32 equivalents, so every browser can use the critic, class margin, stroke checks, shade
continuity, and outer-halo evidence. Generated images, labels, seeds, and latent coordinates are
never sent to an inference server.

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

The exporter validates the float32 graphs, creates calibrated per-channel uint8 Conv/linear graphs
for WebAssembly, preserves the scorer tail needed to maintain critic ordering, checks numeric and
ranking error budgets, and records every model and checkpoint hash in
`public/models/manifest.json`. Commit the refreshed model files and manifest together.

Browser seeds are stable within this application, but its small JavaScript Gaussian generator is
not PyTorch's random-number implementation. A numeric seed therefore does not select the exact
same latent vectors in the browser and Python applications.

## Publish

The `pages.yml` workflow builds and deploys this directory after relevant changes reach `main`.
In the GitHub repository settings, select **GitHub Actions** as the Pages source once; subsequent
deployments are automatic.
