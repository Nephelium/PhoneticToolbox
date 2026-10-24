import test from 'node:test';import assert from 'node:assert/strict';
import {createEditor} from '../src/modules/annotation/editor.mjs';
import {serializeGrid} from '../src/modules/annotation/format.ts';

function fixture(){
 const messages:string[]=[],reads:number[]=[],e=createEditor({message:m=>messages.push(m)});
 const silent=new Float32Array(48000),voiced=Float32Array.from({length:48000},(_,i)=>i>=6000&&i<17000?.35*Math.sin(i*.091):0);
 e.state.audioBuffer={duration:3,sampleRate:16000,numberOfChannels:2,getChannelData:i=>{reads.push(i);return [silent,voiced][i];}};
 e.state.textgrid={xmin:0,xmax:3,tiers:[{name:'words',intervals:[{xmin:0,xmax:.35,text:''},{xmin:.35,xmax:1.12,text:'a1'},{xmin:1.12,xmax:3,text:''}]}]};
 e.controls.fitStart.value='0';e.controls.fitEnd.value='2';return {e,messages,reads};
}
test('M12 intensity uses the explicitly selected stereo channel without reading the first channel',()=>{
 const {e,messages,reads}=fixture(),before=serializeGrid(e.state.textgrid!);
 e.fitIntensityRange(0);assert.equal(serializeGrid(e.state.textgrid!),before);assert.equal(e.state.undoStack.length,0);assert.match(messages.at(-1)!,/第 1 声道/);
 reads.length=0;e.fitIntensityRange(1);assert.notEqual(serializeGrid(e.state.textgrid!),before);assert(reads.length>0&&reads.every(i=>i===1));assert.match(messages.at(-1)!,/第 2 声道/);assert.equal(e.state.undoStack.length,1);
 e.undo();assert.equal(serializeGrid(e.state.textgrid!),before);
});
test('M12 missing or invalid intensity channel never falls back or consumes undo',()=>{
 const {e,reads}=fixture(),before=serializeGrid(e.state.textgrid!);e.saveUndoState();const history=[...e.state.undoStack];
 for(const channel of [undefined,-1,2,.5,NaN]){
  assert.throws(()=>e.fitIntensityRange(channel as number),/声道/);assert.equal(serializeGrid(e.state.textgrid!),before);assert.deepEqual(e.state.undoStack,history);assert.equal(e.state.dirty,false);
 }
 assert.deepEqual(reads,[]);
});
test('M12 selected-channel RMS retains the original mono numeric result',()=>{
 const {e}=fixture(),env=e.localIntensityEnvelope(.2,1.3,1)!;assert(env&&env.times.length>4);assert.equal(e.localIntensityEnvelope(.2,1.3,0),null);
 const reference=createEditor();reference.state.audioBuffer={duration:3,sampleRate:16000,numberOfChannels:1,getChannelData:()=>e.state.audioBuffer!.getChannelData(1)};
 assert.deepEqual(env,reference.localIntensityEnvelope(.2,1.3,0));
});
