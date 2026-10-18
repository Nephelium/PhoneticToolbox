import test from 'node:test';
import assert from 'node:assert/strict';
import {webVocalPort} from '../src/platform/vocal.ts';
import {LocalAssets,MemoryAssetDatabase} from '../src/platform/local-assets.ts';
import {vocalProfileStore,type VocalProfileStore} from '../src/platform/vocal-profile.ts';

const originals=new Map(['window','AudioContext','sessionStorage'].map(k=>[k,Object.getOwnPropertyDescriptor(globalThis,k)]));
let legacy:string|null=null;
test.before(()=>{
 Object.defineProperty(globalThis,'window',{configurable:true,value:{addEventListener(){},removeEventListener(){}}});
 Object.defineProperty(globalThis,'AudioContext',{configurable:true,value:class {}});
 Object.defineProperty(globalThis,'sessionStorage',{configurable:true,value:{getItem(){return legacy;}}});
});
test.after(()=>{for(const [k,d] of originals)if(d)Object.defineProperty(globalThis,k,d);else Reflect.deleteProperty(globalThis,k);});
const fresh=()=>({version:1,active:'default',sets:[{id:'default',name:'默认构型',presets:[] as any[],favorites:['a']}]});
// Storage/queue fixture only. Real core validation and synthesis are covered in
// parity and owned Chrome/Qt acceptance, not inferred from this fake executor.
class Runtime {
 speaker='JD2';models:any={};audio:any={volume:.8,gain_db:0,device:'browser'};
 get model(){return this.models[this.speaker]??=( {library:fresh(),sequence:{frames:[],pitch_curve:[]}} );}
 async request(op:string,b:any={}){
  if(op==='meta')return {speaker:this.speaker,speaker_options:['JD2','W02'],audio_settings:{...this.audio}};
  if(op==='speaker/select'){this.speaker=b.speaker;return {speaker:this.speaker};}
  if(op==='library/load')return structuredClone(this.model.library);
  if(op==='library/save'){if(!b.sets)throw Error('invalid library');this.model.library=structuredClone(b);return structuredClone(b);}
  if(op==='presets/save'){this.model.library.sets[0].presets=structuredClone(b.presets);return b.presets;}
  if(op==='favorites/save'){this.model.library.sets[0].favorites=b.ids;return b.ids;}
  if(op==='keyframes/load')return structuredClone(this.model.sequence);
  if(op==='keyframes'){this.model.sequence=structuredClone(b);return {saved:b.frames.length};}
  if(op==='audio/settings'){this.audio={...this.audio,...b};return this.audio;}
  return {};
 }
 close(){}
}
const port=(store:VocalProfileStore,runtime=new Runtime())=>({runtime,port:webVocalPort(async()=>{throw Error('unexpected server');},'project','workspace',()=>runtime,store)});
const changed=(label:string)=>{const l=fresh();l.sets[0]!.name=label;return l;};

test('saved libraries survive disposal and a new workspace; account/project scopes stay separate',async()=>{
 const db=new MemoryAssetDatabase(),store=vocalProfileStore(new LocalAssets('owner','project',db));
 const a=port(store);await a.port.request('library/save',changed('甲'));await a.port.close(true);
 const b=port(store);await b.port.request('meta');assert.equal((await b.port.request('library/load')).sets[0].name,'甲');await b.port.close();
 for(const [owner,project] of [['other','project'],['owner','other']])assert.equal(await vocalProfileStore(new LocalAssets(owner!,project!,db)).read(),undefined);
});
test('quota failure rolls back native state and the following queued write still succeeds',async()=>{
 const base=vocalProfileStore(new LocalAssets('owner','project',new MemoryAssetDatabase()));let writes=0;
 const a=port({read:()=>base.read(),write:async(e,v)=>{if(++writes===1)throw Error('quota');await base.write(e,v);}});
 const first=assert.rejects(a.port.request('library/save',changed('失败')),/quota/);
 const second=a.port.request('library/save',changed('成功'));await first;await second;
 assert.equal(writes,2);assert.equal(JSON.parse((await base.read())!).models.JD2.library.sets[0].name,'成功');
 await a.port.close();
});
test('failed first audio write restores the native default and remains retryable',async()=>{
 let fail=true;const base=vocalProfileStore(new LocalAssets('owner','project',new MemoryAssetDatabase()));
 const a=port({read:()=>base.read(),write:async(e,v)=>{if(fail)throw Error('quota');await base.write(e,v);}});
 await assert.rejects(a.port.request('audio/settings',{volume:.3}),/quota/);assert.equal(a.runtime.audio.volume,.8);
 fail=false;await a.port.request('audio/settings',{volume:.4});assert.equal(a.runtime.audio.volume,.4);await a.port.close();
});
test('stale window cannot replace a newer library and rollback restores its previous view',async()=>{
 const store=vocalProfileStore(new LocalAssets('owner','project',new MemoryAssetDatabase())),a=port(store),b=port(store);
 await a.port.request('meta');await b.port.request('meta');await a.port.request('library/save',changed('新记录'));
 await assert.rejects(b.port.request('library/save',changed('过期覆盖')),/另一窗口/);
 assert.equal((await b.port.request('library/load')).sets[0].name,'默认构型');assert.equal(JSON.parse((await store.read())!).models.JD2.library.sets[0].name,'新记录');
 await a.port.close();await b.port.close();
});
test('legacy adoption never overrides a durable record; malformed source bytes remain intact',async()=>{
 const assets=new LocalAssets('owner','project',new MemoryAssetDatabase()),store=vocalProfileStore(assets);
 legacy=JSON.stringify({presets:[{id:'old',name:'a i'}],favorites:['a'],sequence:{frames:[]}});
 const a=port(store);await a.port.request('meta');assert.equal(a.runtime.model.library.sets[0].presets[0].name,'a i');await a.port.close();
 legacy='broken legacy';const b=port(store);await b.port.request('meta');assert.equal(b.runtime.model.library.sets[0].presets[0].name,'a i');await b.port.close();legacy=null;
 for(const raw of ['{broken',JSON.stringify({library:{bad:true}})]){
  await assets.setValue('vocal-profile-v1',raw);const c=port(store);await assert.rejects(c.port.request('meta'),/恢复副本/);assert.equal(await store.read(),raw);await c.port.close();
 }
});
test('speaker-specific sets and keyframes round trip through a new executor',async()=>{
 const store=vocalProfileStore(new LocalAssets('owner','project',new MemoryAssetDatabase())),a=port(store);
 await a.port.request('library/save',changed('男声'));await a.port.request('keyframes',{frames:[{f0:123}]});
 await a.port.request('speaker/select',{speaker:'W02'});await a.port.request('library/save',changed('女声'));await a.port.close();
 const b=port(store);await b.port.request('meta');assert.equal(b.runtime.speaker,'W02');assert.equal((await b.port.request('library/load')).sets[0].name,'女声');
 await b.port.request('speaker/select',{speaker:'JD2'});assert.equal((await b.port.request('library/load')).sets[0].name,'男声');assert.equal((await b.port.request('keyframes/load')).frames[0].f0,123);await b.port.close();
});
