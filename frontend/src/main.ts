import { createApp } from 'vue';
import App from './App.vue';
import { initializePlatform } from './platform/desktop.ts';
import {installPageZoom} from './state/pageZoom.ts';
import {initializeIcons} from './state/app-icon.ts';
import {initializeBackground} from './state/background.ts';
import './design/background.css';
installPageZoom();

initializePlatform().then(()=>Promise.all([initializeIcons(),initializeBackground()])).then(()=>createApp(App).mount('#app')).catch(()=>{document.getElementById('app')!.textContent='桌面连接失败，请关闭此窗口后重试。';});
