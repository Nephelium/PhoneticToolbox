// The embedded workbench owns presentation bindings only. Locale changes never
// submit an engine request, replace controls, or change scientific state.
import chinese from './locales/zh-CN.mjs';
import {localeLoaders} from './locales/loaders.mjs';

let locale='zh-CN',messages=chinese,selection=0;
const loaded=new Map([['zh-CN',Promise.resolve(chinese)]]),bindings=new Map(),listeners=new Set();
const loads=new Map();
export const localeCacheInfo=()=>({locale,cached:[...loaded.keys()],loads:Object.fromEntries(loads)});
export const currentLocale=()=>locale;
export function tr(key,values={}){
  const source=messages[key]??chinese[key]??key;
  return source.replace(/\{([A-Za-z_][A-Za-z0-9_]*|[0-9]+)\}/g,(match,name)=>Object.hasOwn(values,name)?String(typeof values[name]==='function'?values[name]():values[name]):match);
}
export class LocalizedLabel{
  constructor(key,values={}){this.key=key;this.values=values;}
  toString(){return tr(this.key,this.values);}
  [Symbol.toPrimitive](){return this.toString();}
}
export const msg=(key,values)=>new LocalizedLabel(key,values);
const valueOf=value=>{while(typeof value==='function')value=value();return String(value??'');};
export const join=(...parts)=>({toString:()=>parts.map(valueOf).join('')});
export function bindText(node,value){
  let row=bindings.get(node);if(!row){row={attributes:new Map()};bindings.set(node,row);}
  row.text=value;node.textContent=valueOf(value);
  row.target=node.nodeType===3?node:node.firstChild;
  if(!row.target){row.target=node.ownerDocument.createTextNode('');node.append(row.target);}
  return value;
}
export function appendText(node,value){const previous=bindings.get(node)?.text??node.textContent;return bindText(node,join(previous,value));}
export function bindAttribute(node,name,value){
  let row=bindings.get(node);if(!row){row={attributes:new Map()};bindings.set(node,row);}
  row.attributes.set(name,value);node.setAttribute(name,valueOf(value));return value;
}
export function textNode(value){const node=document.createTextNode('');bindText(node,value);return node;}
export function setHTML(node,value){node.innerHTML=valueOf(value);registerStaticBindings(node);return value;}
export const parameterLabel=parameter=>msg('modules.vocal-tract-frame.parameters.'+parameter.name);
export function onLocaleChange(listener){listeners.add(listener);return()=>listeners.delete(listener);}
export function registerStaticBindings(root=document){
  const document=root.ownerDocument??root;
  const walker=document.createTreeWalker(root,128); // Explicit key anchors only.
  const anchors=[];while(walker.nextNode())if(walker.currentNode.data.startsWith('m10-i18n:'))anchors.push(walker.currentNode);
  for(const anchor of anchors){const target=anchor.nextSibling;if(target?.nodeType===3)bindText(target,msg(anchor.data.slice(9)));}
  const nodes=[...(root.nodeType===1?[root]:[]),...root.querySelectorAll('[data-m10-i18n],[data-m10-i18n-text]')];
  for(const node of nodes){const text=node.getAttribute('data-m10-i18n-text');if(text)bindText(node,msg(text));const mappings=node.getAttribute('data-m10-i18n');if(!mappings)continue;for(const [attribute,key]of Object.entries(JSON.parse(mappings)))bindAttribute(node,attribute,msg(key));}
}
export function refreshBindings(){
  for(const [node,row]of bindings){
    if(!node.isConnected){bindings.delete(node);continue;}
    if(Object.hasOwn(row,'text')&&(row.target===node||row.target?.parentNode===node))row.target.nodeValue=valueOf(row.text);
    for(const [name,value]of row.attributes)node.setAttribute(name,valueOf(value));
  }
}
function refresh(){
  document.documentElement.lang=locale;
  refreshBindings();
  for(const listener of listeners)listener(locale);
  document.dispatchEvent(new CustomEvent('m10-locale',{detail:{locale}}));
}
export async function setFrameLocale(code){
  if(code!=='zh-CN'&&!Object.hasOwn(localeLoaders,code))return false;
  const request=++selection;if(code===locale)return true;
  let pending=loaded.get(code);if(!pending){loads.set(code,(loads.get(code)??0)+1);pending=localeLoaders[code]();loaded.set(code,pending);pending.catch(()=>loaded.delete(code));}
  const catalog=await pending;if(request!==selection)return false;
  locale=code;messages=catalog;refresh();return true;
}
export class LocalizedError extends Error{
  constructor(value){super(String(value));Object.defineProperty(this,'message',{get:()=>String(value),configurable:true});this.name='Error';}
}
export const uiError=value=>new LocalizedError(value);
export function receiveError(display,raw){
  if(display&&typeof display==='object'&&['message','joined'].includes(display.type)){
    let remaining=512;
    const decode=(value,depth=0)=>{
      if(--remaining<0||depth>16)throw Error('Message structure exceeds limits');
      if(value===null||['string','boolean'].includes(typeof value)||typeof value==='number'&&Number.isFinite(value))return value;
      if(!value||typeof value!=='object'||Array.isArray(value))throw Error('Invalid message structure');
      if(value.type==='joined'&&Array.isArray(value.parts))return join(...value.parts.map(part=>decode(part,depth+1)));
      if(value.type!=='message'||typeof value.key!=='string'||!Object.hasOwn(chinese,value.key))throw Error('Unknown message key');
      const values=value.values;
      if(values!==undefined&&(!values||typeof values!=='object'||Array.isArray(values)))throw Error('Invalid message values');
      return msg(value.key,values?Object.fromEntries(Object.entries(values).map(([key,part])=>[key,decode(part,depth+1)])):{});
    };
    try{return uiError(decode(display));}catch{/* Unknown diagnostics remain literal. */}
  }
  if(display&&typeof display==='object'&&typeof display.key==='string'&&Object.hasOwn(chinese,display.key)){
    const values=display.values;
    if(values===undefined||values&&typeof values==='object'&&!Array.isArray(values)&&Object.values(values).every(value=>value===null||['string','number','boolean','undefined'].includes(typeof value)))return uiError(msg(display.key,values));
  }
  return Error(typeof raw==='string'?raw:String(raw??''));
}

if(typeof window!=='undefined'){
  // Card/SVG updates remove old nodes frequently. Release their display
  // closures after the current DOM update rather than retaining detached work.
  let cleanupQueued=false;
  const observer=new MutationObserver(()=>{if(cleanupQueued)return;cleanupQueued=true;queueMicrotask(()=>{cleanupQueued=false;for(const node of bindings.keys())if(!node.isConnected)bindings.delete(node);});});
  observer.observe(document.documentElement,{childList:true,subtree:true});
  window.addEventListener('pagehide',()=>observer.disconnect(),{once:true});
  window.addEventListener('message',event=>{
    if(event.source!==parent||event.origin!==location.origin||event.data?.type!=='m10-locale')return;
    void setFrameLocale(event.data.locale).catch(()=>{});
  });
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',()=>registerStaticBindings(),{once:true});else registerStaticBindings();
}
