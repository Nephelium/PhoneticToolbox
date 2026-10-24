import test from 'node:test';
import assert from 'node:assert/strict';
import {crc32} from 'node:zlib';
import {m08Archive} from '../src/platform/m08Archive.ts';
import {m08Port} from '../src/platform/m08.ts';
import {defaults} from '../src/modules/pitch-manipulation/state.ts';
import {sha256} from '../src/platform/research.ts';

test('M08-R1 ZIP keeps UTF-8 names, bytes, CRC and independent entries',()=>{
 const files=[{name:'原音 ɑ̃˥.wav',raw:new Uint8Array([1,2,3,4]).buffer},{name:'原音 ɑ̃˥ (2).wav',raw:new Uint8Array([5,6]).buffer}];
 const raw=m08Archive(files),v=new DataView(raw);let offset=0;
 for(const file of files){
  assert.equal(v.getUint32(offset,true),0x04034b50);assert.equal(v.getUint16(offset+6,true),0x800);
  const length=v.getUint32(offset+18,true),n=v.getUint16(offset+26,true);
  assert.equal(new TextDecoder().decode(raw.slice(offset+30,offset+30+n)),file.name);
  const bytes=new Uint8Array(raw.slice(offset+30+n,offset+30+n+length));assert.deepEqual(bytes,new Uint8Array(file.raw));assert.equal(v.getUint32(offset+14,true),crc32(bytes));offset+=30+n+length;
 }
 assert.equal(v.getUint32(offset,true),0x02014b50);assert.equal(v.getUint16(raw.byteLength-12,true),2);
});

test('M08-R1 ZIP rejects duplicate/path names and bounded export excess',()=>{
 const raw=new ArrayBuffer(0);
 assert.throws(()=>m08Archive([{name:'../x.wav',raw}]));assert.throws(()=>m08Archive([{name:'x.wav',raw},{name:'X.wav',raw}]));
 assert.throws(()=>m08Archive([{name:'x.wav',raw:new ArrayBuffer(64_000_001)}]));
 assert.throws(()=>m08Archive(Array.from({length:257},(_,i)=>({name:i+'.wav',raw}))));
});

test('M08-R1 native bulk export uses existing results and returns cancellation/partial failures verbatim',async()=>{
 const calls:string[]=[];const file={id:'input',name:'input.wav',kind:'audio' as const,size:1};
 const result={id:'r',name:'output.wav',source_id:'asset',job_id:'job',saved:false,sha256:'hash',start:0,end:1,config:defaults(),times:[0,.5],original_f0:[100,120]};
 let outcome={saved:[] as {id:string;name:string}[],failed:[] as {id:string;error:string}[],cancelled:true};
 const port=m08Port({project:'p',source:async()=>({asset_id:'asset',sha256:'source-hash'}),request:async<T>(action:string)=>{calls.push(action);return [result] as T;},job:async()=>{throw Error('must not create copy jobs')},read:async()=>{throw Error('must use native verified export')},cancel:async()=>{},exportMany:async values=>{assert.equal(values[0].id,'r');assert.equal(values[0].source_id,'asset');return outcome;}});
 const history=await port.history(file);assert.equal(history[0].source_id,'input');
 assert.equal(await port.saveMany!(history),outcome);
 outcome={saved:[],failed:[{id:'r',error:'disk full'}],cancelled:false};assert.equal(await port.saveMany!(history),outcome);assert.deepEqual(calls,['history']);
 await assert.rejects(port.saveMany!([]));await assert.rejects(port.saveMany!([history[0],history[0]]));
});

test('browser save chooses location before reading, preserves cancel and aborts failed writes',async()=>{
 const raw=new Uint8Array([1,2,3]).buffer,hash=await sha256(raw),events:string[]=[];
 const result={id:'r',name:'output.wav',source_id:'a',job_id:'j',saved:false,sha256:hash,start:0,end:1,config:defaults(),times:[],original_f0:[]};
 const port=m08Port({project:'p',source:async()=>({asset_id:'a',sha256:'h'}),request:async<T>()=>[result] as T,job:async()=>{throw Error('no copy')},read:async()=>{events.push('read');return raw},cancel:async()=>{}});
 const values=await port.history({id:'input',name:'a.wav',kind:'audio',size:1});
 const old=(globalThis as any).showSaveFilePicker;
 try{
  (globalThis as any).showSaveFilePicker=async()=>{events.push('picker');throw new DOMException('cancel','AbortError')};
  assert.equal((await port.saveMany!(values)).cancelled,true);assert.deepEqual(events,['picker']);events.length=0;
  (globalThis as any).showSaveFilePicker=async()=>{events.push('picker');return {name:'chosen.wav',createWritable:async()=>({write:async(blob:Blob)=>{assert.equal(blob.size,3);events.push('write')},close:async()=>{events.push('close')},abort:async()=>{events.push('abort')}})}};
  assert.equal((await port.saveMany!(values)).destination,'chosen.wav');assert.deepEqual(events,['picker','read','write','close']);
  events.length=0;(globalThis as any).showSaveFilePicker=async()=>({createWritable:async()=>({write:async()=>{throw Error('disk full')},abort:async()=>{events.push('abort')}})});
  await assert.rejects(port.saveMany!(values),/disk full/);assert.deepEqual(events,['read','abort']);
 }finally{if(old)(globalThis as any).showSaveFilePicker=old;else delete (globalThis as any).showSaveFilePicker;}
});


test('optional sequence prefixes affect native and browser exports only, with stable retry numbers',async()=>{
 const raw=new Uint8Array([1,2,3]).buffer,hash=await sha256(raw);
 const results=['a','b'].map(id=>({id,name:'原音.wav',source_id:'asset',job_id:'j',saved:false,sha256:hash,start:0,end:1,config:defaults(),times:[],original_f0:[]}));
 const transport={project:'p',source:async()=>({asset_id:'asset',sha256:'h'}),request:async<T>()=>results as T,job:async()=>{throw Error('no copy')},read:async()=>raw,cancel:async()=>{}};
 const native=m08Port({...transport,exportMany:async values=>({saved:values.map(r=>({id:r.id,name:r.name})),failed:[]})});
 const values=await native.history({id:'input',name:'source.wav',kind:'audio',size:1});
 assert.deepEqual((await native.saveMany!(values)).saved.map(r=>r.name),['原音.wav','原音.wav']);
 assert.deepEqual((await native.saveMany!(values,{a:1,b:2})).saved.map(r=>r.name),['1_原音.wav','2_原音.wav']);
 assert.equal((await native.saveMany!([values[1]],{a:1,b:2})).saved[0].name,'2_原音.wav');
 assert.deepEqual(values.map(r=>r.name),['原音.wav','原音.wav']);
 assert.deepEqual((await native.saveMany!(values)).saved.map(r=>r.name),['原音.wav','原音.wav']);
 await assert.rejects(native.saveMany!(values,{a:0}),/序号/);
 const web=m08Port(transport),selected=await web.history({id:'input',name:'source.wav',kind:'audio',size:1});
 const old=(globalThis as any).showSaveFilePicker;let blob:Blob|undefined,suggestion='';
 try{
  (globalThis as any).showSaveFilePicker=async(options:any)=>{suggestion=options.suggestedName;return {name:suggestion,createWritable:async()=>({write:async(value:Blob)=>{blob=value},close:async()=>{},abort:async()=>{}})}};
  const receipt=await web.saveMany!(selected,{a:1,b:2});
  assert.deepEqual(receipt.saved.map(r=>r.name),['1_原音.wav','2_原音.wav']);
  const bytes=new Uint8Array(await blob!.arrayBuffer()),view=new DataView(bytes.buffer);let offset=0;const names=[];
  while(view.getUint32(offset,true)===0x04034b50){const n=view.getUint16(offset+26,true),size=view.getUint32(offset+18,true);names.push(new TextDecoder().decode(bytes.slice(offset+30,offset+30+n)));offset+=30+n+size;}
  assert.deepEqual(names,['1_原音.wav','2_原音.wav']);
  await web.saveMany!([selected[1]],{a:1,b:2});assert.equal(suggestion,'2_原音.wav');
  await web.saveMany!([selected[1]]);assert.equal(suggestion,'原音.wav');
 }finally{if(old)(globalThis as any).showSaveFilePicker=old;else delete (globalThis as any).showSaveFilePicker;}
});
