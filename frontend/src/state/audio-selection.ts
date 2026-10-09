import {inject,provide,ref,shallowRef,toRaw,watch,type InjectionKey} from 'vue';
import type {Workspace} from './workspace.ts';

export function createAudioSelectionGroup(){
 const getters=new Set<()=>Workspace>(),version=ref(0),focused=shallowRef<Workspace|null>(null);
 const known=new Map<Workspace,Workspace['asset']>();
 const tracks=()=>[...new Map([...getters].map(get=>{const state=get();return [toRaw(state),state] as const;})).values()];
 const same=(a:Workspace|null,b:Workspace)=>!!a&&toRaw(a)===toRaw(b);
 const clear=(state:Workspace)=>{state.start=0;state.end=0;};
 function select(state:Workspace){
  focused.value=state;
  if(tracks().length>1)for(const other of tracks())if(!same(other,state))clear(other);
 }
 function register(get:()=>Workspace){getters.add(get);version.value++;return ()=>{getters.delete(get);version.value++;};}
 const dispose=watch(()=>{version.value;return tracks().map(state=>({state,asset:state.asset}));},entries=>{
  const multi=entries.length>1;
  for(const {state,asset}of entries){
   const key=toRaw(state),previous=known.get(key),changed=asset!==previous;
   if(changed&&same(focused.value,state))focused.value=null;
   // New multi-audio inputs have no implicit full-length selection. Existing
   // deliberate selection is retained when another input becomes available.
   if(multi&&asset&&(changed||!same(focused.value,state)))clear(state);
   known.set(key,asset);
  }
  const keys=new Set(entries.map(e=>toRaw(e.state)));
  for(const key of known.keys())if(!keys.has(key))known.delete(key);
  if(focused.value&&!keys.has(toRaw(focused.value)))focused.value=null;
 },{flush:'post'});
 return {register,select,dispose,selected:(state:Workspace)=>same(focused.value,state),
  keyboard:(state:Workspace)=>same(focused.value,state)||(tracks().length===1&&state.end>state.start)};
}
type Group=ReturnType<typeof createAudioSelectionGroup>;
const key:InjectionKey<Group>=Symbol('module-audio-selection');
export function provideAudioSelection(){const group=createAudioSelectionGroup();provide(key,group);return group;}
export const useAudioSelection=()=>inject(key,null);
