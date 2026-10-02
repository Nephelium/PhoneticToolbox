import test from 'node:test';
import assert from 'node:assert/strict';
import {LipReplay,atTime} from '../src/modules/lip-extraction/replay.ts';
import type {LipRow,LipResult,LipPort} from '../src/modules/lip-extraction/port.ts';
import {lipParameterTable} from '../src/modules/parameter-display/lip.ts';
import {matchingTable} from '../src/modules/parameter-display/state.ts';
import type {ResearchFiles,ResearchFile} from '../src/platform/research.ts';

test('M05-R1 M02 uses all four lip tracks, stable sorting, missing values and offset exactly once',async()=>{
 const vector=(values:(number|null)[])=>({values,nonfinite:values.map(v=>v===null?1:0)});
 const wire={schema:'ptb.lip/1',data:{relative_times:vector([.3,.1,.2]),open:vector([3,1,null]),outer_width:vector([6,2,4]),area:vector([9,3,null]),circularity:vector([.6,.2,.4]),metadata:{time_alignment_mode:'anchored_audio_start',lip_manual_offset:.125}}};
 const file:ResearchFile={id:'lip',name:'audio_recording.lip.json',kind:'lip',size:200};
 const files={annotation:{lip:async()=>({wire,sha256:'1'.repeat(64)})}} as unknown as ResearchFiles;
 const table=await lipParameterTable(files,file);
 assert.deepEqual(table.columns,['Time_s','LipArea','LipWidth','LipOpen','LipCirc']);
 assert.deepEqual(table.rows,[[.225,3,2,1,.2],[.325,null,4,null,.4],[.425,9,6,3,.6]]);
 const audio={...file,id:'a',name:'audio_recording.wav',kind:'audio' as const};
 assert.equal(matchingTable(audio,[file])?.id,'lip');
 const old={...file,id:'v2',name:'audio_recording.pkl',kind:'lip_pickle' as const};
 assert.equal(matchingTable(audio,[old])?.id,'v2');
 assert.equal(matchingTable(audio,[file,old])?.id,'lip');
});

test('M05-R1 full playback is independent of the 241-point plot and supports VFR/seek/missing frames',async()=>{
 const source:LipRow[]=Array.from({length:1200},(_,index)=>({index,time_s:index/30+(index>=430?2:0),detected:index!==400,points:null,metrics:index===400?null:{open:index}}));
 const result:LipResult={id:'test',name:'VFR',backend:'fixture',files:[],rows:source.filter((_,i)=>i%5===0||i===1199),metadata:{timing:{decoded_frames:source.length}}};
 const calls:number[]=[];
 const port={replay:async(_:LipResult,start:number)=>{calls.push(start);return {rows:source.slice(start,start+90),complete:start+90>=source.length};}} as LipPort;
 const playback=new LipReplay(result,port);
 for(const index of [0,1,89,90,241,399,400,401,429,430,1199,10]){
  assert.deepEqual(await playback.time(source[index].time_s),source[index]);
  assert.deepEqual(await playback.frame(index),source[index]);
 }
 assert.equal((await playback.time(source[429].time_s+1))?.index,429);
 assert(calls.every(i=>i%90===0));
 assert((playback as any).pages.size<=3);
 assert.equal(atTime(source,source[400].time_s),400);
});

test('M05-R1 a failed frame read is retryable and shared pending requests are bounded',async()=>{
 let fail=true,calls=0;
 const row:LipRow={index:0,time_s:0,detected:false,points:null,metrics:null};
 const result={id:'x',rows:[row],metadata:{timing:{decoded_frames:1}}} as LipResult;
 const port={replay:async()=>{calls++;if(fail)throw Error('read failed');return {rows:[row],complete:true};}} as unknown as LipPort;
 const playback=new LipReplay(result,port);
 await assert.rejects(playback.frame(0),/read failed/);fail=false;
 const values=await Promise.all([playback.frame(0),playback.frame(0)]);
 assert.deepEqual(values,[row,row]);assert.equal(calls,2);
});
