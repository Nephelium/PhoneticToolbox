// Shared workbench font payload, received only from the owning parent frame.
let current=null,held=0,queued=null;
export function canvasFont(size,ipa=false){return `${size*(current?.size||12)/12}px ${ipa?(current?.figureIpa||'"PTB-Doulos",serif'):(current?.figure||'"Segoe UI","Microsoft YaHei",sans-serif')}`;}
export function svgFontFamily(){return current?.resolved?`${JSON.stringify(current.resolved.figureLatin)},${JSON.stringify(current.resolved.figureZh)},sans-serif`:'"Segoe UI","Microsoft YaHei",sans-serif';}
export async function applyFonts(value){
 if(!value||typeof value.css!=='string'||typeof value.ui!=='string')return;
 if(held){queued=value;return;}current=value;
 let style=document.getElementById('ptb-font-faces');if(!style){style=document.createElement('style');style.id='ptb-font-faces';document.head.append(style);}style.textContent=value.css;
 const root=document.documentElement.style;root.setProperty('--sans',value.ui);root.setProperty('--serif',value.ui);root.setProperty('--font-ipa',value.ipa);root.setProperty('--font-mono',value.mono);root.setProperty('--font-figure',value.figure);
 await Promise.all([document.fonts.load(canvasFont(14),'中文 Time'),document.fonts.load(canvasFont(14,true),'a ɑ tʰ')]);await document.fonts.ready;document.dispatchEvent(new Event('m10-theme'));
}
export async function freezeFonts(){held++;await document.fonts.ready;return ()=>{held--;if(!held&&queued){const value=queued;queued=null;void applyFonts(value);}};}
