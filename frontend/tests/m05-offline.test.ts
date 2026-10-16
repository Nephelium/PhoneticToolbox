import test from 'node:test';
import assert from 'node:assert/strict';
import {desktopM05} from '../src/platform/m05.ts';
import type {LipResult} from '../src/modules/lip-extraction/port.ts';
const result=(id:string):LipResult=>({id,name:id+'.mp4',backend:'test',rows:[],metadata:{timing:{lip_manual_offset:.12}},files:[]});

test('M05 batch chooses one directory and acknowledges each completed result, with distinct offsets',async()=>{
 const calls:any[]=[];let choices=0;const saved:string[]=[];
 const port=await desktopM05(async body=>{calls.push(body);return {available:true,saved:true} as any;},async()=>{choices++;return {id:'output'};});
 assert(await port.saveBatch!([{result:result('a'),offset:.12},{result:result('b'),offset:-.04}],'apply',{video:true,animation:true,quality:'high'},id=>saved.push(id)));
 assert.equal(choices,1);assert.deepEqual(saved,['a','b']);
 assert.deepEqual(calls.slice(1).map(c=>[c.job,c.offset,c.options,c.directory]),[['a',.12,{video:true,animation:true,quality:'high'},'output'],['b',-.04,{video:true,animation:true,quality:'high'},'output']]);
});
test('M05 cancelled directory writes nothing; failure preserves acknowledgement of completed items only',async()=>{
 const saved:string[]=[];let writes=0;
 const transport=async(body:any)=>{if(body.op==='m05_save'){if(++writes===2)throw Error('disk failure');return {saved:true} as any;}return {available:true} as any;};
 const cancelled=await desktopM05(transport,async()=>null);
 assert.equal(await cancelled.saveBatch!([{result:result('a'),offset:0}],'apply',{video:true,animation:true,quality:'high'},id=>saved.push(id)),false);assert.equal(writes,0);
 const port=await desktopM05(transport,async()=>({id:'output'}));
 await assert.rejects(port.saveBatch!([{result:result('a'),offset:0},{result:result('b'),offset:0}],'save_without_offset',{video:false,animation:true,quality:'high'},id=>saved.push(id)),/disk failure/);
 assert.deepEqual(saved,['a']);
});
test('M05 standalone GIF/MP4 exports use existing results, without submitting analysis or saving a bundle',async()=>{
 const calls:any[]=[];const port=await desktopM05(async body=>{calls.push(body);return {available:true,saved:true} as any;},async()=>({id:'output'}));
 for(const format of ['gif','mp4'] as const)assert(await port.exportAnimation!(result('a'),format,'high',new AbortController().signal));
 assert(calls.slice(1).every(c=>typeof c.export_id==='string'));assert.deepEqual(calls.slice(1).map(({export_id,...body})=>body),['gif','mp4'].map(format=>({op:'m05_export_animation',job:'a',directory:'output',format,quality:'high',offset:.12})));
});
