import {computed,ref,watch,type Ref,type WritableComputedRef} from 'vue';
import {createI18n,type MessageFunction,type MessageContext} from 'vue-i18n';
import languages from './languages.json' with {type:'json'};
import {chineseMessages,catalogueLoaders} from './generated/loaders.ts';
import type {AppLocale} from './generated/keys.ts';
import {installErrorFormatter,RuntimeDisplayError} from './runtime-errors.ts';
export type {AppLocale,MessageKey} from './generated/keys.ts';
export type MessageValues=Record<string,string|number|boolean|null|undefined>;
export interface MessageDescriptor {key:string;values?:MessageValues}
export interface LocaleStore {read<T>(key:string,fallback:T):T;write(key:string,value:unknown):boolean}
export const supportedLocales=languages as Array<{code:AppLocale;nativeName:string;direction:'ltr'|'rtl';formatLocale:string}>;
export const isAppLocale=(value:unknown):value is AppLocale=>typeof value==='string'&&supportedLocales.some(row=>row.code===value);
// Message functions support named data only: no HTML, linked-message execution,
// or eval. Literal email addresses, pipes and scientific braces remain literal.
function messages(rows:Record<string,string>):Record<string,MessageFunction<string>>{
 return Object.fromEntries(Object.entries(rows).map(([key,text])=>{
  const parts=text.split(/(\{(?:[A-Za-z_][A-Za-z0-9_]*|[0-9]+)\})/g).map(part=>/^\{(?:[A-Za-z_][A-Za-z0-9_]*|[0-9]+)\}$/.test(part)?{name:part.slice(1,-1)}:{text:part});
  const format:MessageFunction<string>=(context:MessageContext<string>)=>parts.map(part=>{
   if(part.name===undefined)return part.text!;
   const value=context.named(part.name);
   // Scientific notation such as VoQS {V} is literal unless data is supplied.
   return value===undefined?'{'+part.name+'}':value===null?'':String(value);
  }).join('');
  return [key,format];
 }));
}
const missing=new Set<string>();
const errorKeysByText=new Map<string,string>();
function registerErrorMessages(rows:Record<string,string>){
 for(const [key,text]of Object.entries(rows))if((key.startsWith('platform.')||key.startsWith('native.')||key.startsWith('common.language.'))&&!/\.(?:wav|mp4|json|zip)$/i.test(text)&&!errorKeysByText.has(text))errorKeysByText.set(text,key);
}
registerErrorMessages(chineseMessages);
const initialMessages=Object.fromEntries(supportedLocales.map(({code})=>[code,code==='zh-CN'?messages(chineseMessages):{}])) as Record<AppLocale,Record<string,MessageFunction<string>>>;
export const i18n=createI18n({legacy:false,locale:'zh-CN',fallbackLocale:'zh-CN',messages:initialMessages,messageResolver:(object,key)=>(object as Record<string,unknown>)[key] as any,missing:(language,key)=>{missing.add(language+':'+key);return key;},missingWarn:false,fallbackWarn:false});
export const locale=i18n.global.locale as Ref<AppLocale>;
export const pendingLocale=ref<AppLocale|null>(null),localeError=ref('');
const loaded=new Set<AppLocale>(['zh-CN']),flights=new Map<AppLocale,Promise<void>>(),loads=new Map<AppLocale,number>();
let store:LocaleStore|undefined,revision=0,nativeUpdate:((code:AppLocale)=>Promise<void|boolean>)|undefined,nativeInitial:AppLocale|undefined;
const localeEffects=new Set<(code:AppLocale)=>Promise<void>>();
/** Register shared display resources once, without component remounts. */
export function registerLocaleEffect(effect:(code:AppLocale)=>Promise<void>){localeEffects.add(effect);return ()=>localeEffects.delete(effect);}
export function tr(key:string,values:MessageValues={}):string{return i18n.global.t(key,values);}
/** Compatibility for existing error records and platform APIs returning text.
 * This is used only at error boundaries, never for user data or rendered DOM. */
export function errorMessageDescriptor(text:string):MessageDescriptor|undefined {
 const key=i18n.global.te(text,'zh-CN')?text:errorKeysByText.get(text);
 return key?{key}:undefined;
}
export function formatErrorCode(text:string):string {const value=errorMessageDescriptor(text);return value?tr(value.key):text;}
installErrorFormatter(value=>tr(value.key,value.values));
/** Rehydrate only known message descriptors at the Worker/native boundary. */
export function receiveDisplayError(display:unknown,raw:unknown):Error {
 if(display&&typeof display==='object'&&!Array.isArray(display)){
  const value=display as Record<string,unknown>;
  if(typeof value.key==='string'&&i18n.global.te(value.key,'zh-CN')){
   const parameters=value.values;
   if(parameters===undefined||parameters&&typeof parameters==='object'&&!Array.isArray(parameters)&&Object.values(parameters).every(item=>item===null||['string','number','boolean','undefined'].includes(typeof item)))return new RuntimeDisplayError({key:value.key,values:parameters as MessageValues|undefined});
  }
 }
 return Error(typeof raw==='string'?raw:tr('common.status.failed'));
}
export function message(key:string,values?:MessageValues):MessageDescriptor{return {key,values};}
export function messageText(value:MessageDescriptor|string):string{return typeof value==='string'?value:tr(value.key,value.values);}
export function messageRef(initial:string|MessageDescriptor=''):WritableComputedRef<string,string|MessageDescriptor>{const value=ref(initial);return computed({get:()=>messageText(value.value),set:(next:string|MessageDescriptor)=>{value.value=next;}});}
export function missingMessages(){return [...missing].sort();}
export function languageLoadCounts(){return Object.fromEntries(loads);}
export function bindNativeLocale(update:(code:AppLocale)=>Promise<void|boolean>,initial?:AppLocale){nativeUpdate=update;nativeInitial=initial;}
async function loadLanguage(code:AppLocale):Promise<void>{
 if(loaded.has(code))return;
 let flight=flights.get(code);if(!flight){
  const loader=catalogueLoaders[code as keyof typeof catalogueLoaders];
  if(!loader)throw Error('unsupported_locale');
  flight=loader().then(rows=>{registerErrorMessages(rows);i18n.global.setLocaleMessage(code,messages(rows));loaded.add(code);loads.set(code,(loads.get(code)??0)+1);}).finally(()=>flights.delete(code));flights.set(code,flight);
 }
 await flight;
}
export async function setLocale(code:AppLocale):Promise<boolean>{
 if(!isAppLocale(code))return false;
 const request=++revision;pendingLocale.value=code;localeError.value='';
 try{
  await loadLanguage(code);if(request!==revision)return false;
  // No application reload, remount or business-state writes occur here.
  locale.value=code;
  if(store&&!store.write('uiLocale',code))localeError.value='common.language.saveFailed';
  if(!store)try{globalThis.localStorage?.setItem('ptb.v3.uiLocale',JSON.stringify(code));}catch{localeError.value='common.language.saveFailed';}
  if(nativeUpdate&&await nativeUpdate(code)===false)localeError.value='common.language.saveFailed';
  await Promise.all([...localeEffects].map(effect=>effect(code)));
  if(request!==revision)return false;
  return true;
 }catch{if(request===revision)localeError.value='common.language.failed';return false;}
 finally{if(request===revision)pendingLocale.value=null;}
}
export async function initializeLocale(preferences?:LocaleStore){
 // Bind a fixed owner for this session. Omitting a store restores the anonymous
 // device choice; an account without a choice always starts in Chinese.
 store=preferences;
 let saved:unknown='zh-CN';
 try{saved=store?store.read('uiLocale','zh-CN'):JSON.parse(globalThis.localStorage?.getItem('ptb.v3.uiLocale')??'null');}catch{}
 await setLocale(isAppLocale(saved)?saved:store?'zh-CN':nativeInitial??'zh-CN');
}
if(typeof document!=='undefined')watch(locale,code=>{document.documentElement.lang=code;document.documentElement.dir=supportedLocales.find(row=>row.code===code)?.direction??'ltr';document.documentElement.dataset.appLocale=code;},{immediate:true,flush:'sync'});
export function translateError(error:unknown,fallbackKey='common.status.failed'):string{
 const text=error instanceof Error?error.message:typeof error==='string'?error:'';
 return text&&i18n.global.te(text,locale.value)?tr(text):text||tr(fallbackKey);
}
