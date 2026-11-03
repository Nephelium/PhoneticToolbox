import {localized} from '../module-localization.ts';
import {locale,tr} from '../../i18n/index.ts';
import raw from './data/catalog.json' with {type:'json'};
import sources from './data/sources.json' with {type:'json'};
import type {Catalog, ChartSystem, SymbolEntry} from './types.ts';
export const catalog = raw as Catalog;
export const entries = new Map(catalog.entries.map(entry=>[entry.id,entry]));
export const systems: {id:ChartSystem; label:string; version:string}[] = localized(()=>[
  {id:'ipa',label:'IPA',version:tr('modules.ipa-plus.text.7e6986f516a0')},
  {id:'extipa',label:'extIPA',version:'ICPLA 2025'},
  {id:'voqs',label:'VoQS',version:tr('modules.ipa-plus.text.26b61bc00046')},
]);
export const sourceLinks:Record<string,{title:string;url:string}> = sources;
const symbolCache=new WeakMap<SymbolEntry,Map<string,SymbolEntry>>();
let symbolBuilds=0;
/** Localized display snapshot; insertion text, Unicode, aliases and sources stay authoritative. */
export function localizedSymbol(entry:SymbolEntry):SymbolEntry{
 const language=locale.value;
 let languages=symbolCache.get(entry);if(!languages){languages=new Map();symbolCache.set(entry,languages);}
 const saved=languages.get(language);if(saved)return saved;
 const prefix=`modules.ipa-plus.catalog.${entry.id}`;
 const result={...entry,nameZh:tr(prefix+'.nameZh'),descriptionZh:tr(prefix+'.descriptionZh'),usageZh:tr(prefix+'.usageZh'),contrastZh:tr(prefix+'.contrastZh'),
  ...(entry.representation?{representation:tr(prefix+'.representation')}:{})};
 languages.set(language,result);symbolBuilds++;return result;
}
export function symbolLocalizationBuilds(){return symbolBuilds;}
export function searchEntries(system:ChartSystem,query:string):SymbolEntry[]{
  locale.value;
  const q=query.trim().toLocaleLowerCase();
  return catalog.entries.filter(e=>e.system===system&&(!q||[e.display,e.insertText,e.nameZh,localizedSymbol(e).nameZh,e.nameEn,...e.aliases,...e.codePoints].join(' ').toLocaleLowerCase().includes(q))).map(localizedSymbol);
}
