import raw from './data/symbol-content.json';
import type {SymbolEntry} from './types.ts';
export interface SymbolContent {
 nameZh?:string;nameEn?:string;descriptionZh?:string;usageZh?:string;contrastZh?:string;notesZh?:string;
 media?:{audio?:string;video?:string;animation?:{renderer:string;version:number;config:Record<string,unknown>}};
}
export const symbolContent=raw as {version:1;entries:Record<string,SymbolContent>};
export function contentFor(entry:SymbolEntry){return {...entry,...symbolContent.entries[entry.id]};}
export function mediaUrl(path:string){
 // Only bundled/site-local assets. No absolute paths, traversal or external URLs.
 if(!/^(?:[\p{L}\p{N}_-]+\/)*[\p{L}\p{N}_ .-]+\.(?:mp3|wav|ogg|m4a|mp4|webm)$/u.test(path)||path.split('/').some(p=>p==='.'||p==='..'))throw Error('演示素材路径无效。');
 return new URL(import.meta.env.BASE_URL+path,document.baseURI).href;
}
