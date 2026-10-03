import {ref} from 'vue';
import {projects} from '../platform/browser.ts';
import {normalizeWaveformAppearance,waveformCss,type WaveformAppearance} from '../design/waveform-color.ts';
export const waveformAppearance=ref<WaveformAppearance>(normalizeWaveformAppearance(null));
let installed=false;
function apply(){
 const root=document.documentElement,color=waveformCss(waveformAppearance.value);
 root.style.setProperty('--waveform-color',color);
 root.dataset.waveformColorMode=waveformAppearance.value.mode;
 root.dataset.waveformColor=color;
}
export function installWaveformAppearance(){
 if(installed)return;installed=true;
 waveformAppearance.value=normalizeWaveformAppearance(projects.read<unknown>('waveformAppearance',null));apply();
}
export function setWaveformAppearance(value:Partial<WaveformAppearance>){
 waveformAppearance.value=normalizeWaveformAppearance({...waveformAppearance.value,...value});apply();
 return projects.write('waveformAppearance',waveformAppearance.value);
}
