import { reactive,markRaw } from 'vue';
import { platform } from '../platform/desktop.ts';
import parameters from '../generated/parameters.json';
import type { AudioAsset } from '../platform/types.ts';
export interface Workspace { asset:AudioAsset|null; start:number; end:number; channel:number; parameters:string[]; dirty:boolean; error:string; loading:boolean; zoom:number; offset:number }
export const host=platform();
export const states=reactive<Record<string,Workspace>>({});
export function workspace(id:string):Workspace {
  if(!states[id]) {
    const saved=host.projects.read<{parameters?:string[]}>('draft.'+id,{});
    states[id]={asset:null,start:0,end:0,channel:id==='M03'?1:0,parameters:parameters.map(p=>p.key).filter(k=>!Array.isArray(saved.parameters)||saved.parameters.includes(k)),dirty:false,error:'',loading:false,zoom:1,offset:0};
  }
  return states[id];
}
export function assignAsset(id:string,asset:AudioAsset) {const s=workspace(id);s.asset=markRaw(asset);s.start=0;s.end=asset.duration;s.channel=Math.min(s.channel,asset.channels.length-1);s.zoom=1;s.offset=0;s.error='';}
export function saveDraft(id:string) {const s=workspace(id); const ok=host.projects.write('draft.'+id,{parameters:[...s.parameters]});if(ok)s.dirty=false;return ok;}
