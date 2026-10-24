import test from 'node:test';
import assert from 'node:assert/strict';
import {defaults,valid,preset,override,draw,configPreferences,restorePreferences,unavailableParameters} from '../src/modules/speech-synthesis/state.ts';
import {browserM06} from '../src/platform/browser-m06.ts';
import type {M06Port} from '../src/modules/speech-synthesis/port.ts';

test('M06 preference hydration scales curves and rebuilds consistent F0 state',()=>{
 const short=restorePreferences(defaults(),{...configPreferences(defaults()),duration:.5});
 valid(short);assert.equal(short.duration,.5);assert.equal(short.curves.F0.points.at(-1)![0],.5);
 for(const voice of ['假声','嘎裂']){
  const prior=defaults();preset(prior,voice);
  const reopened=restorePreferences(defaults(),configPreferences(prior));valid(reopened);
  assert.deepEqual(reopened.curves.F0,prior.curves.F0);assert.deepEqual(reopened.f0_transform,prior.f0_transform);
 }
});
test('M06 preference hydration preserves an existing full draft and rejects malformed settings atomically',()=>{
 const draft=defaults();draft.curves.F0.points=[[0,100],[1,160],[2,120]];preset(draft,'假声');
 const before=structuredClone(draft);assert.deepEqual(restorePreferences(draft,configPreferences(draft)),draft);
 assert.throws(()=>restorePreferences(draft,{...configPreferences(draft),duration:-1}));assert.deepEqual(draft,before);
 const narrowed=restorePreferences(defaults(),{...configPreferences(defaults()),f0_range:[200,400]});valid(narrowed);assert.equal(narrowed.curves.F0.points[0][1],200);
});
test('M06 new HNR edits respect the effective ceiling while legacy configurations stay readable',()=>{
 const c=defaults();assert.throws(()=>override(c,'HNR','100'),/60/);assert.throws(()=>override(c,'HNR','40,100'),/60/);
 override(c,'HNR','0,60');draw(c,'HNR',.2,.3,100);assert(c.curves.HNR.points.every(([,v])=>v<=60));
 c.curves.HNR.override=100;valid(c);
 for(const name of ['A1','A2','A3','A4','A5']){assert(unavailableParameters.has(name));assert.throws(()=>override(c,name,'60'),/串联/);assert.throws(()=>draw(c,name,0,.1,60),/串联/);}
});
test('M06 browser storage failure publishes a terminal state and retries saving without recomputing',async()=>{
 const states:string[]=[],cancels:string[]=[];let computes=0,saves=0,blocked=true;
 const remote={cancel:async(id:string)=>{cancels.push(id);}} as unknown as M06Port;
 const port=browserM06(remote,{project:'p',compute:async()=>{computes++;return {files:[{name:'m06.ptb.json',data:btoa(JSON.stringify({config:defaults()}))},{name:'parameters.csv',data:btoa('csv')}]};},save:async(name,raw)=>{saves++;if(blocked)throw Error('本机存储空间不足');return {id:name,sha256:'sha',size_bytes:raw.byteLength};}});
 let id='';await assert.rejects(port.run('generate',defaults(),undefined,new AbortController().signal,j=>{id=j.id;states.push(j.state);}),/存储/);
 assert.deepEqual(states,['running','running','failed']);await port.cancel(id);assert.deepEqual(cancels,[]);
 assert(port.canRetrySave?.(id));blocked=false;
 const result=await port.retrySave!(id,new AbortController().signal,j=>states.push(j.state));
 assert.equal(result.job,id);assert.equal(result.files.length,2);assert.equal(computes,1);assert.equal(saves,3);
 assert.deepEqual(states,['running','running','failed','running','succeeded']);assert.equal(port.canRetrySave?.(id),false);
});
test('M06 browser cancellation publishes a cancelled state and never cancels a remote UUID',async()=>{
 let release!:(v:never)=>void,id='';const states:string[]=[],cancels:string[]=[];
 const port=browserM06({cancel:async(id:string)=>{cancels.push(id);}} as unknown as M06Port,{project:'p',compute:async()=>new Promise(r=>release=r),save:async()=>{throw Error('unexpected');}});
 const pending=port.run('generate',defaults(),undefined,new AbortController().signal,j=>{id=j.id;states.push(j.state);});
 await new Promise(r=>setImmediate(r));await port.cancel(id);release(null as never);
 await assert.rejects(pending,{name:'AbortError'});assert.deepEqual(states,['running','cancelled']);await port.cancel(id);assert.deepEqual(cancels,[]);
});
test('M06 retry resumes after the last saved file and reports computation errors without a save retry',async()=>{
 let blocked=true,computes=0;const saved:string[]=[],states:string[]=[];let id='';
 const port=browserM06({cancel:async()=>{throw Error('unexpected remote');}} as unknown as M06Port,{project:'p',compute:async()=>{computes++;return {files:[{name:'m06.ptb.json',data:btoa('{}')},{name:'parameters.csv',data:btoa('csv')}]};},save:async(name,raw)=>{if(name.endsWith('parameters.csv')&&blocked)throw Error('disk full');saved.push(name);return {id:name,sha256:'sha',size_bytes:raw.byteLength};}});
 await assert.rejects(port.run('generate',defaults(),undefined,new AbortController().signal,j=>{id=j.id;states.push(j.state);}),/disk full/);
 assert.equal(saved.length,1);blocked=false;await port.retrySave!(id,new AbortController().signal,j=>states.push(j.state));assert.equal(saved.length,2);assert.equal(computes,1);assert.notEqual(saved[0],saved[1]);
 const broken=browserM06({} as M06Port,{project:'p',compute:async()=>{throw Error('invalid computation');},save:async()=>{throw Error('unexpected save');}});
 await assert.rejects(broken.run('generate',defaults(),undefined,new AbortController().signal,j=>{id=j.id;states.push(j.state);}),/invalid computation/);
 assert.equal(broken.canRetrySave!(id),false);assert.equal(states.at(-1),'failed');
});
