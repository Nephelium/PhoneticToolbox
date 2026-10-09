import type {SymbolEntry} from './types.ts';
export interface AnimationRequest {entry:SymbolEntry;version:number;config:Record<string,unknown>;container:HTMLElement;signal:AbortSignal}
export type AnimationRenderer=(request:AnimationRequest)=>void|(()=>void)|Promise<void|(()=>void)>;
const renderers=new Map<string,AnimationRenderer>();
export function registerSymbolAnimation(id:string,renderer:AnimationRenderer){
 if(renderers.has(id))throw Error('音标动画接口已登记：'+id);
 renderers.set(id,renderer);return ()=>{if(renderers.get(id)===renderer)renderers.delete(id);};
}
export function symbolAnimation(id:string){return renderers.get(id);}
