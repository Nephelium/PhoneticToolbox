import type {Settings} from './port.ts';
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
