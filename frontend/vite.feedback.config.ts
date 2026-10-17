import {defineConfig} from 'vite';
import vue from '@vitejs/plugin-vue';
export default defineConfig({plugins:[vue()],base:'./',publicDir:false,build:{target:'es2022',outDir:'../output/build-feedback-manager/web',emptyOutDir:true,rolldownOptions:{input:'feedback-manager.html'}}});
