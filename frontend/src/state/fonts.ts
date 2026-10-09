import {ref,shallowRef} from 'vue';
import {defaults,normalizeFonts,fontKey,figureFonts,quoteFamily,type FontPreferences,candidates} from '../design/fonts.ts';
import doulosUrl from '../assets/DoulosSIL-Regular.ttf?url';
import monoUrl from '../assets/JetBrainsMono-Regular.woff2?url';
import {projects} from '../platform/browser.ts';
export const preferences=ref(defaults()),fontError=ref(''),fontRevision=ref(0);
export const fontPayload=shallowRef<{css:string;ui:string;figure:string;ipa:string;figureIpa:string;mono:string;size:number;bodySize:number}>();
let owner:string|undefined,sequence=0,scopeRevision=0;
let fixedFont:Promise<FontFace>|undefined;
let bundledMono:Promise<FontFace>|undefined;
const bundledFamily='JetBrains Mono';
const source=(name:string)=>name===bundledFamily?`url("${new URL(monoUrl,location.href).href}")`:`local(${quoteFamily(name)})`;
const loaded=new Map<string,Promise<string>>();
const latinRange='U+0000-024F,U+1E00-1EFF,U+2000-206F,U+2070-209F,U+20A0-20CF,U+2100-214F,U+2190-22FF';
export const ipaRange='U+0250-02FF,U+0300-036F,U+1D00-1DBF,U+A700-A71F,U+AB30-AB6F,U+10780-107BF,U+1DF00-1DFFF';
const ipaCss=()=>`@font-face{font-family:PTB-Doulos;src:url("${new URL(doulosUrl,location.href).href}")}@font-face{font-family:PTB-IPA-Symbols;src:url("${new URL(doulosUrl,location.href).href}");unicode-range:${ipaRange}}`;
async function available(name:string){
 if(name===bundledFamily){
  if(!bundledMono)bundledMono=new FontFace(bundledFamily,source(name)).load().catch(e=>{bundledMono=undefined;throw e;});
  document.fonts.add(await bundledMono);return;
 }
 const face=new FontFace('PTB-check',`local(${quoteFamily(name)})`);await face.load();
}
async function resolve(request:string,fallback:string[]){
 if(request){try{await available(request);return request;}catch{throw Error(`字体 ${request} 在当前设备不可用，请安装后重试或选择其他字体。`);}}
 for(const name of fallback)try{await available(name);return name;}catch{/* Try next installed system face. */}
 throw Error('找不到可用的系统字体，请在设置中指定本机字体。');
}
async function latinFace(name:string){
 if(!loaded.has(name))loaded.set(name,(async()=>{const alias='PTB-Latin-'+Array.from(name).map(c=>c.codePointAt(0)!.toString(16)).join('-');const face=new FontFace(alias,source(name),{unicodeRange:latinRange});await face.load();document.fonts.add(face);return alias;})().catch(e=>{loaded.delete(name);throw e;}));
 return loaded.get(name)!;
}
export async function prepareFonts(p:FontPreferences){
 const f=figureFonts(p);
 const [zh,latin,mono,fzh,flat]=await Promise.all([resolve(p.zh,candidates.zh),resolve(p.latin,candidates.latin),resolve(p.mono,candidates.mono),resolve(f.zh,candidates.zh),resolve(f.latin,candidates.latin)]);
 const [uiLatin,figureLatin]=await Promise.all([latinFace(latin),latinFace(flat)]);
 const ui=`"PTB-IPA-Symbols","${uiLatin}",${quoteFamily(zh)},sans-serif`,figure=`"PTB-IPA-Symbols","${figureLatin}",${quoteFamily(fzh)},sans-serif`;
 const css=ipaCss()+`@font-face{font-family:"JetBrains Mono";src:${source(bundledFamily)}}`+[...new Map([[uiLatin,latin],[figureLatin,flat]])].map(([alias,name])=>`@font-face{font-family:"${alias}";src:${source(name)};unicode-range:${latinRange}}`).join('');
 return {css,ui,figure,ipa:`"PTB-Doulos",${quoteFamily(zh)},serif`,figureIpa:`"PTB-Doulos",${quoteFamily(fzh)},serif`,mono:`"PTB-IPA-Symbols",${quoteFamily(mono)},${quoteFamily(zh)},monospace`,size:f.size,bodySize:p.bodySize,resolved:{zh,latin,mono,figureZh:fzh,figureLatin:flat}};
}
function install(payload:Awaited<ReturnType<typeof prepareFonts>>){
 let style=document.getElementById('ptb-font-faces');if(!style){style=document.createElement('style');style.id='ptb-font-faces';document.head.append(style);}style.textContent=payload.css;
 const root=document.documentElement.style;for(const [name,value] of Object.entries({'--font':payload.ui,'--font-figure':payload.figure,'--font-ipa':payload.ipa,'--font-figure-ipa':payload.figureIpa,'--font-mono':payload.mono,'--body-size':payload.bodySize+'px','--figure-size':payload.size+'px'}))root.setProperty(name,value);
 fontPayload.value=payload;fontRevision.value++;window.dispatchEvent(new Event('ptb-fonts-changed'));
}
export async function setFonts(value:unknown,persist=true){
 const p=normalizeFonts(value),ticket=++sequence,payload=await prepareFonts(p);
 if(ticket!==sequence)return false;
 // Verify the immutable IPA resource too; never silently substitute another face.
 if(!fixedFont)fixedFont=new FontFace('PTB-Doulos',`url("${doulosUrl}")`).load().catch(e=>{fixedFont=undefined;throw e;});
 document.fonts.add(await fixedFont);
 if(ticket!==sequence)return false;
 if(persist&&!projects.write(fontKey(owner),p))throw Error('字体偏好保存失败，请检查本机存储权限。');
 install(payload);preferences.value=p;fontError.value='';return true;
}
export async function selectFontOwner(next?:string){
 const scope=++scopeRevision;owner=next;const saved=projects.read(fontKey(owner),defaults());
 // Clear prior owner's visible state before asynchronous font resolution.
 preferences.value=defaults();fontPayload.value=undefined;fontError.value='';++sequence;
 for(const name of ['--font','--font-figure','--font-ipa','--font-figure-ipa','--font-mono','--body-size','--figure-size'])document.documentElement.style.removeProperty(name);
 try{await setFonts(saved,false);}catch(e){
  if(scope!==scopeRevision)return;
  const error=e instanceof Error?e.message:'字体加载失败。';
  // SimSun and Times New Roman are defaults, not redistributed system fonts.
  // A device without them still gets a usable UI; the stored preference is retained.
  let recovered=await setFonts(defaults(),false).catch(()=>false);
  if(!recovered&&scope===scopeRevision)recovered=await setFonts({...defaults(),zh:'',latin:''},false).catch(()=>false);
  if(scope===scopeRevision)fontError.value=error+(recovered?' 当前界面使用可用的默认或兼容字体，已保存的选择保留。':'');
 }
}
export async function fontsReady(){await document.fonts.load('14px PTB-Doulos','a ɑ tʰ ã');await document.fonts.ready;}
export function exportFontSnapshot(){const f=figureFonts(preferences.value);const p=fontPayload.value as Awaited<ReturnType<typeof prepareFonts>>|undefined;return {schema_version:'font/1' as const,zh:p?.resolved.figureZh||f.zh||'Microsoft YaHei',latin:p?.resolved.figureLatin||f.latin||'Segoe UI',ipa:'Doulos SIL' as const,size_px:f.size,fallback_policy:'portable' as const};}
export {doulosUrl};
