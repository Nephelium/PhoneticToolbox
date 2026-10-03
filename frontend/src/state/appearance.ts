import {palettes,paletteTokens,normalizePalette,normalizeMode,type ThemeMode} from '../design/themes.ts';
export {normalizePalette,normalizeMode,type ThemeMode};
export function installPalettes(){
 if(document.getElementById('ptb-palettes'))return;
 const style=document.createElement('style');style.id='ptb-palettes';
 style.textContent=palettes.flatMap(p=>(['light','dark'] as const).map(mode=>`:root[data-palette="${p.id}"][data-theme="${mode}"]{${Object.entries(paletteTokens(p.id,mode)).map(([k,v])=>`${k}:${v}`).join(';')}}`)).join('\n');
 document.head.append(style);
}
