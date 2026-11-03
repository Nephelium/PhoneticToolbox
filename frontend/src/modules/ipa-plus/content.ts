import {uiError as displayError,uiMessage as displayMessage} from '../module-localization.ts';
import {locale,tr} from '../../i18n/index.ts';
import raw from './data/symbol-content.json' with {type:'json'};
import type {SymbolEntry} from './types.ts';
import {entries,localizedSymbol} from './catalog.ts';
export interface SymbolContent {
 nameZh?:string;nameEn?:string;descriptionZh?:string;usageZh?:string;contrastZh?:string;notesZh?:string;
 media?:{audio?:string;video?:string;animation?:{renderer:string;version:number;config:Record<string,unknown>}};
}
export const symbolContent=raw as {version:1;entries:Record<string,SymbolContent>};
const contentCache=new WeakMap<SymbolEntry,Map<string,SymbolEntry&SymbolContent>>();
let contentBuilds=0;
export function contentFor(entry:SymbolEntry):SymbolEntry&SymbolContent{
 const source=entries.get(entry.id)??entry,language=locale.value;
 let languages=contentCache.get(source);if(!languages){languages=new Map();contentCache.set(source,languages);}
 const saved=languages.get(language);if(saved)return saved;
 const curated=symbolContent.entries[source.id],result={...localizedSymbol(source),...curated};
 if(curated)for(const field of ['nameZh','nameEn','descriptionZh','usageZh','contrastZh','notesZh'] as const){
  if(curated[field])result[field]=tr(`modules.ipa-plus.content.${source.id}.${field}`);
 }
 languages.set(language,result);contentBuilds++;return result;
}
export function contentLocalizationBuilds(){return contentBuilds;}
export function mediaUrl(path:string){
 // Only bundled/site-local assets. No absolute paths, traversal or external URLs.
 if(!/^(?:[\p{L}\p{N}_-]+\/)*[\p{L}\p{N}_ .-]+\.(?:mp3|wav|ogg|m4a|mp4|webm)$/u.test(path)||path.split('/').some(p=>p==='.'||p==='..'))throw displayError(displayMessage('modules.ipa-plus.text.640e0fdadb06'));
 return new URL(import.meta.env.BASE_URL+path,document.baseURI).href;
}
