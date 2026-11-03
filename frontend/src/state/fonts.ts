import {messageRef} from '../i18n/index.ts';
import {uiMessage,uiError,uiErrorMessage,localized,joinMessages} from '../modules/module-localization.ts';
import {tr,locale,registerLocaleEffect,type AppLocale} from '../i18n/index.ts';
import {ref,shallowRef} from 'vue';
import {defaults,normalizeFonts,fontKey,figureFonts,quoteFamily,type FontPreferences,candidates,localeCjkCandidates} from '../design/fonts.ts';
import doulosUrl from '../assets/DoulosSIL-Regular.ttf?url';
import monoUrl from '../assets/JetBrainsMono-Regular.woff2?url';
import {projects} from '../platform/browser.ts';
export const preferences=ref(defaults()),fontError=messageRef(),fontRevision=ref(0);
export const fontPayload=shallowRef<{css:string;ui:string;figure:string;ipa:string;figureIpa:string;mono:string;size:number;bodySize:number}>();
let owner:string|undefined,sequence=0,scopeRevision=0;
let fixedFont:Promise<FontFace>|undefined;
let bundledMono:Promise<FontFace>|undefined;
const bundledFamily='JetBrains Mono';
const source=(name:string)=>name===bundledFamily?`url("${new URL(monoUrl,location.href).href}")`:`local(${quoteFamily(name)})`;
const loaded=new Map<string,Promise<string>>();
const availableFaces=new Map<string,Promise<void>>();
const latinRange='U+0000-024F,U+1E00-1EFF,U+2000-206F,U+2070-209F,U+20A0-20CF,U+2100-214F,U+2190-22FF';
export const ipaRange='U+0250-02FF,U+0300-036F,U+1D00-1DBF,U+A700-A71F,U+AB30-AB6F,U+10780-107BF,U+1DF00-1DFFF';
const ipaCss=()=>`@font-face{font-family:PTB-Doulos;src:url("${new URL(doulosUrl,location.href).href}")}@font-face{font-family:PTB-IPA-Symbols;src:url("${new URL(doulosUrl,location.href).href}");unicode-range:${ipaRange}}`;
async function available(name:string){
 if(name===bundledFamily){
  if(!bundledMono)bundledMono=new FontFace(bundledFamily,source(name)).load().catch(e=>{bundledMono=undefined;throw e;});
  document.fonts.add(await bundledMono);return;
 }
 if(!availableFaces.has(name))availableFaces.set(name,(async()=>{const face=new FontFace('PTB-check',`local(${quoteFamily(name)})`);await face.load();})().catch(error=>{availableFaces.delete(name);throw error;}));
 await availableFaces.get(name);
}
async function resolve(request:string,fallback:string[]){
 if(request){try{await available(request);return request;}catch{throw uiError(uiMessage('state.the_font_value_is_unavailable_on_this_device_install',{v0:(request)}));}}
 for(const name of fallback)try{await available(name);return name;}catch{/* Try next installed system face. */}
 throw uiError(uiMessage('state.no_usable_system_font_was_found_specify_an_installed'));
}
async function latinFace(name:string){
 if(!loaded.has(name))loaded.set(name,(async()=>{const alias='PTB-Latin-'+Array.from(name).map(c=>c.codePointAt(0)!.toString(16)).join('-');const face=new FontFace(alias,source(name),{unicodeRange:latinRange});await face.load();document.fonts.add(face);return alias;})().catch(e=>{loaded.delete(name);throw e;}));
 return loaded.get(name)!;
}
async function resolveCjk(request:string,code:AppLocale){
 for(const family of localeCjkCandidates(code,request))try{await available(family);return family;}catch{/* Try the next installed Japanese face. */}
 return resolve(request,candidates.zh);
}
export async function prepareFonts(p:FontPreferences,code:AppLocale=locale.value){
 const f=figureFonts(p);
 const [zh,latin,mono,fzh,flat]=await Promise.all([resolveCjk(p.zh,code),resolve(p.latin,candidates.latin),resolve(p.mono,candidates.mono),resolveCjk(f.zh,code),resolve(f.latin,candidates.latin)]);
 const [uiLatin,figureLatin]=await Promise.all([latinFace(latin),latinFace(flat)]);
 const ui=`"PTB-IPA-Symbols","${uiLatin}",${quoteFamily(zh)},sans-serif`,figure=`"PTB-IPA-Symbols","${figureLatin}",${quoteFamily(fzh)},sans-serif`;
 const css=ipaCss()+`@font-face{font-family:"JetBrains Mono";src:${source(bundledFamily)}}`+[...new Map([[uiLatin,latin],[figureLatin,flat]])].map(([alias,name])=>`@font-face{font-family:"${alias}";src:${source(name)};unicode-range:${latinRange}}`).join('');
 return {css,ui,figure,ipa:`"PTB-Doulos",${quoteFamily(zh)},serif`,figureIpa:`"PTB-Doulos",${quoteFamily(fzh)},serif`,mono:`"PTB-IPA-Symbols",${quoteFamily(mono)},${quoteFamily(zh)},monospace`,size:f.size,bodySize:p.bodySize,resolved:{zh,latin,mono,figureZh:fzh,figureLatin:flat},locale:code};
}
function install(payload:Awaited<ReturnType<typeof prepareFonts>>){
 let style=document.getElementById('ptb-font-faces');if(!style){style=document.createElement('style');style.id='ptb-font-faces';document.head.append(style);}style.textContent=payload.css;
 const root=document.documentElement.style;for(const [name,value] of Object.entries({'--font':payload.ui,'--font-figure':payload.figure,'--font-ipa':payload.ipa,'--font-figure-ipa':payload.figureIpa,'--font-mono':payload.mono,'--body-size':payload.bodySize+'px','--figure-size':payload.size+'px'}))root.setProperty(name,value);
 document.documentElement.dataset.fontLocale=payload.locale;
 fontPayload.value=payload;fontRevision.value++;window.dispatchEvent(new Event('ptb-fonts-changed'));
}
export async function setFonts(value:unknown,persist=true){
 const p=normalizeFonts(value),ticket=++sequence;let payload=await prepareFonts(p);
 while(ticket===sequence&&payload.locale!==locale.value)payload=await prepareFonts(p);
 if(ticket!==sequence)return false;
 // Verify the immutable IPA resource too; never silently substitute another face.
 if(!fixedFont)fixedFont=new FontFace('PTB-Doulos',`url("${doulosUrl}")`).load().catch(e=>{fixedFont=undefined;throw e;});
 document.fonts.add(await fixedFont);
 while(ticket===sequence&&payload.locale!==locale.value)payload=await prepareFonts(p);
 if(ticket!==sequence)return false;
 if(persist&&!projects.write(fontKey(owner),p))throw uiError(uiMessage('state.font_preferences_could_not_be_saved_check_local_storage'));
 install(payload);if(JSON.stringify(preferences.value)!==JSON.stringify(p))preferences.value=p;fontError.value='';return true;
}
registerLocaleEffect(async code=>{
 if(!fontPayload.value)return;
 const snapshot=JSON.stringify(preferences.value),ticket=sequence;
 try{
  const payload=await prepareFonts(preferences.value,code);
  if(locale.value===code&&ticket===sequence&&JSON.stringify(preferences.value)===snapshot){install(payload);await document.fonts.ready;}
 }catch(error){if(locale.value===code&&ticket===sequence)fontError.value=uiErrorMessage(error);}
});
export async function selectFontOwner(next?:string){
 const scope=++scopeRevision;owner=next;const saved=projects.read(fontKey(owner),defaults());
 // Clear prior owner's visible state before asynchronous font resolution.
 preferences.value=defaults();fontPayload.value=undefined;fontError.value='';++sequence;
 for(const name of ['--font','--font-figure','--font-ipa','--font-figure-ipa','--font-mono','--body-size','--figure-size'])document.documentElement.style.removeProperty(name);
 try{await setFonts(saved,false);}catch(e){
  if(scope!==scopeRevision)return;
  const error=e instanceof Error?uiErrorMessage(e):uiMessage('state.the_font_could_not_be_loaded');
  // SimSun and Times New Roman are defaults, not redistributed system fonts.
  // A device without them still gets a usable UI; the stored preference is retained.
  let recovered=await setFonts(defaults(),false).catch(()=>false);
  if(!recovered&&scope===scopeRevision)recovered=await setFonts({...defaults(),zh:'',latin:''},false).catch(()=>false);
  if(scope===scopeRevision)fontError.value=joinMessages(error,recovered?uiMessage('state.the_interface_is_using_an_available_default_or_compatible'):'');
 }
}
export async function fontsReady(){await document.fonts.load('14px PTB-Doulos','a ɑ tʰ ã');await document.fonts.ready;}
export function exportFontSnapshot(){const f=figureFonts(preferences.value);const p=fontPayload.value as Awaited<ReturnType<typeof prepareFonts>>|undefined;return {schema_version:'font/1' as const,zh:p?.resolved.figureZh||f.zh||'Microsoft YaHei',latin:p?.resolved.figureLatin||f.latin||'Segoe UI',ipa:'Doulos SIL' as const,size_px:f.size,fallback_policy:'portable' as const};}
export {doulosUrl};
