import {palettes,paletteTokens,customPalette,normalizePalette,normalizeMode,type ThemeMode} from '../design/themes.ts';
import {ref} from 'vue';
import {projects} from '../platform/browser.ts';
import {normalizeHex} from '../design/waveform-color.ts';
export const customPaletteColor=ref('#657b35');
export const paletteRevision=ref(0);
export {normalizePalette,normalizeMode,type ThemeMode};
function renderPalettes(){
 palettes[palettes.findIndex(p=>p.id==='custom')]=customPalette(customPaletteColor.value);
 const style=document.getElementById('ptb-palettes')!;
 style.textContent=palettes.flatMap(p=>(['light','dark'] as const).map(mode=>`:root[data-palette="${p.id}"][data-theme="${mode}"]{${Object.entries(paletteTokens(p.id,mode)).map(([k,v])=>`${k}:${v}`).join(';')}}`)).join('\n');
 document.documentElement.dataset.customPalette=customPaletteColor.value;paletteRevision.value++;
}
export function setCustomPaletteColor(color:string){
 const valid=normalizeHex(color);if(!valid)return false;
 customPaletteColor.value=valid;renderPalettes();return projects.write('customPaletteColor',valid);
}
export function installPalettes(){
 if(document.getElementById('ptb-palettes'))return;
 const style=document.createElement('style');style.id='ptb-palettes';
 document.head.append(style);
 customPaletteColor.value=normalizeHex(projects.read('customPaletteColor',null))??'#657b35';renderPalettes();
}
