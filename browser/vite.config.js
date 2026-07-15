import {defineConfig} from 'vite';
import {fileURLToPath} from 'node:url';
import path from 'node:path';

const browserDirectory = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  root: browserDirectory,
  base: './',
  publicDir: path.join(browserDirectory, 'public'),
  build: {
    outDir: path.join(browserDirectory, 'dist'),
    emptyOutDir: true,
    rollupOptions: {input: path.join(browserDirectory, 'index.html')},
  },
  server: {open: '/'},
});
