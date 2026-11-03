import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';

export default defineConfig({ plugins: [vue()], base: './',
  // All messages are safe named-text functions; no runtime message compiler is needed.
  resolve:{alias:[{find:/^vue-i18n$/,replacement:'vue-i18n/dist/vue-i18n.runtime.esm-bundler.js'}]},
  define:{__VUE_I18N_LEGACY_API__:false,__VUE_I18N_FULL_INSTALL__:false,__INTLIFY_DROP_MESSAGE_COMPILER__:true,__INTLIFY_PROD_DEVTOOLS__:false},
  cacheDir: process.env.PTB_VITE_CACHE_DIR || 'node_modules/.vite', build: { target: 'es2022', rolldownOptions: { input: ['index.html','feedback-manager.html'] } } });
