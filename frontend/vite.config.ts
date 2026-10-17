import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';

export default defineConfig({ plugins: [vue()], base: './', cacheDir: process.env.PTB_VITE_CACHE_DIR || 'node_modules/.vite', build: { target: 'es2022', rolldownOptions: { input: ['index.html','feedback-manager.html'] } } });
