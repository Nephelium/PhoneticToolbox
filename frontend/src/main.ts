import { createApp } from 'vue';
import App from './App.vue';
import { initializePlatform } from './platform/desktop.ts';
import {installPageZoom} from './state/pageZoom.ts';
import {initializeIcons} from './state/app-icon.ts';
import {initializeBackground} from './state/background.ts';
import {i18n,initializeLocale,tr} from './i18n/index.ts';
import './design/background.css';
import './design/theme-transitions.css';
import {installThemeTransitions} from './state/theme-transitions.ts';
installPageZoom();
installThemeTransitions();

initializePlatform().then(()=>Promise.all([initializeIcons(),initializeBackground(),initializeLocale()])).then(()=>createApp(App).use(i18n).mount('#app')).catch(async()=>{await initializeLocale();document.getElementById('app')!.textContent=tr('common.startup.connectionFailed');});
