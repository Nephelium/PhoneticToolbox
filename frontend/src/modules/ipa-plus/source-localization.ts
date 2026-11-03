import {locale,tr} from '../../i18n/index.ts';
import type {SymbolEntry} from './types.ts';
type SourceRef=SymbolEntry['sourceRefs'][number];
const cache=new WeakMap<SourceRef,Map<string,string>>();
/** Explanatory locators localize; source IDs, bibliography and coordinates stay intact. */
export function localizedSourceLocator(entryId:string,source:SourceRef):string{
 if(!/[\u3400-\u9fff]/u.test(source.locator))return source.locator;
 const language=locale.value;
 let languages=cache.get(source);if(!languages){languages=new Map();cache.set(source,languages);}
 const saved=languages.get(language);if(saved!==undefined)return saved;
 const value=tr(`modules.ipa-plus.locator.${entryId}.${source.sourceId}`);languages.set(language,value);return value;
}
