import raw from './data/catalog.json';
import sources from './data/sources.json';
import type {Catalog, ChartSystem, SymbolEntry} from './types.ts';
export const catalog = raw as Catalog;
export const entries = new Map(catalog.entries.map(entry=>[entry.id,entry]));
export const systems: {id:ChartSystem; label:string; version:string}[] = [
  {id:'ipa',label:'IPA',version:'2026 重印 · 内容修订 2015/2005'},
  {id:'extipa',label:'extIPA',version:'ICPLA 2025'},
  {id:'voqs',label:'VoQS',version:'2016 修订表 · Ball 等 2018，图 2'},
];
export const sourceLinks:Record<string,{title:string;url:string}> = sources;
export function searchEntries(system:ChartSystem,query:string):SymbolEntry[]{
  const q=query.trim().toLocaleLowerCase();
  return catalog.entries.filter(e=>e.system===system&&(!q||[e.display,e.insertText,e.nameZh,e.nameEn,...e.aliases,...e.codePoints].join(' ').toLocaleLowerCase().includes(q)));
}
