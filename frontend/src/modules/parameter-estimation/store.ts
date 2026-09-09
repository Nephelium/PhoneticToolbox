import { reactive,shallowReactive } from 'vue';
import { createState,draftJson,type M01State } from './state.ts';
import { projects } from '../../platform/browser.ts';
const states=shallowReactive(new Map<string,M01State>());
export function m01State(key:string){let state=states.get(key);if(!state){state=reactive(createState(projects.read('m01.'+key,{})));states.set(key,state);}return state;}
export function saveM01(key:string){const state=m01State(key);const ok=projects.write('m01.'+key,JSON.parse(draftJson(state)));if(ok){state.saved=draftJson(state);state.wave.dirty=false;}return ok;}
export function forgetM01(key:string){const state=states.get(key);if(state){state.loadVersion++;state.listVersion++;state.wave.asset=null;}states.delete(key);}
export function clearM01Owner(owner:string){for(const key of states.keys())if(key.startsWith('server:'+owner+':'))forgetM01(key);}
