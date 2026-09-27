import { createApp } from 'vue';
import App from './App.vue';
import { initializePlatform } from './platform/desktop.ts';
import {installPageZoom} from './state/pageZoom.ts';
installPageZoom();

initializePlatform().then(()=>createApp(App).mount('#app')).catch(()=>{document.getElementById('app')!.textContent='桌面连接失败，请关闭此窗口后重试。';});
