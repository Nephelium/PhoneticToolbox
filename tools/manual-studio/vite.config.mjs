import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const studio = path.dirname(fileURLToPath(import.meta.url));
export default defineConfig({
  plugins: [vue()],
  resolve: { alias: { vue: path.join(studio, 'node_modules/vue/dist/vue.runtime.esm-bundler.js') } },
  server: { host: '127.0.0.1', fs: { allow: [path.resolve(studio, '../..')] } },
  build: { outDir: 'dist' },
});
