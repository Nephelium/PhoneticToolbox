export type WaveformColorMode='blue'|'theme'|'custom';
export interface WaveformAppearance {mode:WaveformColorMode;custom:string}
export function normalizeHex(value:unknown):string|null {
 if(typeof value!=='string')return null;
 const color=value.trim().toLowerCase();
 if(/^#[0-9a-f]{6}$/.test(color))return color;
 if(/^#[0-9a-f]{3}$/.test(color))return '#'+[...color.slice(1)].map(c=>c+c).join('');
 return null;
}
export function normalizeWaveformAppearance(value:unknown):WaveformAppearance {
 const saved=value&&typeof value==='object'&&!Array.isArray(value)?value as Record<string,unknown>:{};
 return {mode:saved.mode==='blue'||saved.mode==='custom'?saved.mode:'theme',custom:normalizeHex(saved.custom)??'#2463eb'};
}
export function waveformCss(value:WaveformAppearance){return value.mode==='custom'?value.custom:value.mode==='theme'?'var(--accent)':'var(--wave)';}
/** Preserve the established blue print palette unless a color is explicitly chosen. */
export function waveformExportColor(fallback:string){
 const root=document.documentElement;
 if(!root.dataset.waveformColorMode||root.dataset.waveformColorMode==='blue')return fallback;
 return normalizeHex(getComputedStyle(root).getPropertyValue('--waveform-color'))??fallback;
}
