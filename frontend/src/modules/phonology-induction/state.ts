import type {Settings,Options,Row} from './port.ts';
export const clone=<T>(v:T):T=>JSON.parse(JSON.stringify(v));
export function select(items:string[],selected:string[],anchor:string|undefined,value:string,ctrl:boolean,shift:boolean){
 if(shift&&anchor!==undefined){const a=items.indexOf(anchor),b=items.indexOf(value);if(a>=0&&b>=0)return {selected:items.slice(Math.min(a,b),Math.max(a,b)+1),anchor};}
 return {selected:ctrl?(selected.includes(value)?selected.filter(v=>v!==value):[...selected,value]):[value],anchor:value};
}
export function move(items:string[],selected:string[],target:string){
 const chosen=items.filter(v=>selected.includes(v));if(!chosen.length||chosen.includes(target))return items;
 const rest=items.filter(v=>!chosen.includes(v)),index=rest.indexOf(target);if(index<0)return items;
 rest.splice(index,0,...chosen);return rest;
}
export function swap(items:string[],source:string,target:string){const out=[...items],a=out.indexOf(source),b=out.indexOf(target);if(a>=0&&b>=0)[out[a],out[b]]=[out[b],out[a]];return out;}
export function merge(settings:Settings,kind:'initial'|'final',source:string,target:string){
 const values=settings[`${kind}_order`];if(source===target||!values.includes(source)||!values.includes(target))throw Error('请选择不同的有效目标音标。');
 settings[`${kind}_map`][source]=target;settings[`${kind}_order`]=values.filter(v=>v!==source);
}
export function resolve(value:string,map:Record<string,string>){const seen=new Set<string>();while(Object.hasOwn(map,value)){if(seen.has(value))throw Error('归并映射存在循环。');seen.add(value);value=map[value];}return value;}
export const label=(v:string)=>v===''?'空韵':v;
export const defaultOptions=():Options=>({skip_first_row:true,consonant_only_as_zero_initial:true,computation_revision:'m14/2',character_column:1,ipa_column:2,note_column:3,start_row:2,table_index:0,encoding:'auto',delimiter:'auto'});
export function undoMerge(settings:Settings,kind:'initial'|'final',source:string,original:string[]){
 const map=settings[`${kind}_map`];if(!Object.hasOwn(map,source))return;
 delete map[source];const order=settings[`${kind}_order`];
 const target=original.slice(original.indexOf(source)+1).find(v=>order.includes(v));
 const index=target===undefined?order.length:order.indexOf(target);order.splice(index,0,source);
}
export interface Entry extends Row {index:number;mapped_initial:string;mapped_final:string;tone_class:string}
export function entries(rows:Row[],settings:Settings):Entry[]{return rows.map((r,index)=>({...r,index,mapped_initial:resolve(r.initial,settings.initial_map),mapped_final:resolve(r.final,settings.final_map),tone_class:settings.tone_map[r.tone_value]||r.tone_value||'0'}));}
export const pairKey=(initial:string,final:string)=>JSON.stringify([initial,final]);
export function groupEntries(rows:Entry[],settings:Settings){
 const buckets=new Map<string,Entry[]>();for(const row of rows){const key=pairKey(row.mapped_initial,row.mapped_final);if(!buckets.has(key))buckets.set(key,[]);buckets.get(key)!.push(row);}
 const tones=[...new Set(settings.tone_order.map(t=>settings.tone_map[t]||t||'0'))];
 const known=new Set(tones);
 for(const [key,list] of buckets){const byTone=new Map<string,Entry[]>(),unknown:Entry[]=[];for(const row of list){if(!known.has(row.tone_class)){unknown.push(row);continue;}if(!byTone.has(row.tone_class))byTone.set(row.tone_class,[]);byTone.get(row.tone_class)!.push(row);}const ordered:Entry[]=[];for(const t of tones)ordered.push(...(byTone.get(t)??[]));ordered.push(...unknown);buckets.set(key,ordered);}
 return buckets;
}
export function orderedEntries(buckets:Map<string,Entry[]>,settings:Settings,mode:'initial'|'final'){
 const out:Entry[]=[];const outer=mode==='initial'?settings.initial_order:settings.final_order,inner=mode==='initial'?settings.final_order:settings.initial_order;
 for(const a of outer)for(const b of inner)out.push(...(buckets.get(mode==='initial'?pairKey(a,b):pairKey(b,a))??[]));return out;
}
export function searchEntries(rows:Entry[],query:string,scope:'all'|'character'|'ipa'){
 const normalize=(s:string)=>s.normalize('NFD').toLocaleLowerCase();const q=normalize(query.trim());if(!q)return [];
 return rows.filter(r=>{const ipa=[r.ipa,r.mapped_initial,r.mapped_final,(r.mapped_initial==='Ø'?'':r.mapped_initial)+r.mapped_final+r.tone_value];return (scope==='character'?[r.character]:scope==='ipa'?ipa:[r.character,...ipa]).some(v=>normalize(v).includes(q));}).map(r=>r.index);
}
