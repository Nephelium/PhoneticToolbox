import test from 'node:test';
import assert from 'node:assert/strict';
import {crc32} from 'node:zlib';
import {m08Archive} from '../src/platform/m08Archive.ts';
import {m08Port} from '../src/platform/m08.ts';
import {defaults} from '../src/modules/pitch-manipulation/state.ts';

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
