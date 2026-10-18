import test from 'node:test';import assert from 'node:assert/strict';
import {createEditor,type Grid} from '../src/modules/annotation/editor.mjs';
import {serializeGrid,parseGrid} from '../src/modules/annotation/format.ts';
import {manualDoubleClick,sequenceDoubleClick} from '../src/modules/annotation/sequence.ts';
import {endpointClickTime} from '../src/platform/waveform.ts';

const setup=()=>{const e=createEditor();e.state.textgrid={xmin:0,xmax:4,tiers:[{name:'words',xmin:0,xmax:4,intervals:[{xmin:0,xmax:4,text:'当前'}]}]};return e;};
for(const [mode,expected] of [
 ['inside',[[0,1,'当前'],[1,3,'参考'],[3,4,'当前']]],
 ['outside',[[0,.5,'当前'],[.5,1,'参考'],[1,3,'当前'],[3,3.5,'参考'],[3.5,4,'当前']]],
 ['before',[[0,.5,'当前'],[.5,1,'参考'],[1,4,'当前']]],
 ['after',[[0,1,'当前'],[1,3.5,'参考'],[3.5,4,'当前']]],
] as const)test(`M12 reference ${mode} clips to overlap, preserves uncovered audio and supports undo`,async()=>{
 const e=setup(),before=serializeGrid(e.state.textgrid!);
 e.state.referenceTextGrid={xmin:.5,xmax:3.5,tiers:[{name:'words',intervals:[{xmin:.5,xmax:3.5,text:'参考'}]}]};
 const source=structuredClone(e.state.referenceTextGrid);e.controls.spliceMode.value=mode;e.controls.spliceStart.value='1';e.controls.spliceEnd.value='3';
 await e.applyReferenceSplice();assert.deepEqual(e.wordTier()!.intervals.map(i=>[i.xmin,i.xmax,i.text]),expected);
 assert.deepEqual(e.state.referenceTextGrid,source);assert.equal(parseGrid(serializeGrid(e.state.textgrid!)).xmax,4);e.undo();assert.equal(serializeGrid(e.state.textgrid!),before);
});
test('M12 longer reference crossing both ends retains current file and partial-tier domains',async()=>{
 const e=setup();e.state.textgrid!.tiers.push({name:'partial',xmin:1,xmax:3,intervals:[{xmin:1,xmax:3,text:'旧'}]});
 e.state.referenceTextGrid={xmin:0,xmax:6,tiers:[{name:'words',intervals:[{xmin:0,xmax:6,text:'新'}]},{name:'partial',xmin:2,xmax:5,intervals:[{xmin:2,xmax:5,text:'局部'}]}]};
 e.controls.spliceMode.value='inside';e.controls.spliceStart.value='0';e.controls.spliceEnd.value='6';await e.applyReferenceSplice();
 assert.deepEqual(e.wordTier()!.intervals,[{xmin:0,xmax:4,text:'新'}]);assert.deepEqual(e.state.textgrid!.tiers[1],{name:'partial',xmin:1,xmax:3,intervals:[{xmin:1,xmax:2,text:'旧'},{xmin:2,xmax:3,text:'局部'}]});
 assert.doesNotThrow(()=>serializeGrid(e.state.textgrid!));
});
test('M12 shorter point layer replaces only the covered points including overlap endpoints',async()=>{
 const e=setup();e.state.textgrid!.tiers.push({name:'marks',xmin:0,xmax:4,points:[{number:.2,mark:'保留左'},{number:1,mark:'旧'},{number:3,mark:'旧'},{number:3.8,mark:'保留右'}]});
 e.state.referenceTextGrid={xmin:1,xmax:3,tiers:[{name:'marks',points:[{number:1,mark:'左'},{number:2,mark:'中'},{number:3,mark:'右'}]}]};
 e.controls.spliceMode.value='inside';e.controls.spliceStart.value='0';e.controls.spliceEnd.value='4';await e.applyReferenceSplice();
 assert.deepEqual(e.state.textgrid!.tiers[1].points,[{number:.2,mark:'保留左'},{number:1,mark:'左'},{number:2,mark:'中'},{number:3,mark:'右'},{number:3.8,mark:'保留右'}]);
});
test('M12 reference precision difference no longer rejects the whole selected interval',async()=>{
 const e=setup();e.state.referenceTextGrid={xmin:0,xmax:4.000001,tiers:[{name:'words',intervals:[{xmin:0,xmax:4.000001,text:'参考'}]}]};
 e.controls.spliceMode.value='after';e.controls.spliceStart.value='2';await e.applyReferenceSplice();
 assert.deepEqual(e.wordTier()!.intervals,[{xmin:0,xmax:2,text:'当前'},{xmin:2,xmax:4,text:'参考'}]);
});
test('M12 no overlap and late tier mismatch preserve all tiers and do not add undo entries',async()=>{
 const e=setup(),before=serializeGrid(e.state.textgrid!);e.state.referenceTextGrid={xmin:5,xmax:6,tiers:[{name:'words',intervals:[{xmin:5,xmax:6,text:'不可见'}]}]};
 e.controls.spliceMode.value='inside';e.controls.spliceStart.value='0';e.controls.spliceEnd.value='4';await assert.rejects(e.applyReferenceSplice(),/没有重叠/);
 assert.equal(e.state.undoStack.length,0);assert.equal(serializeGrid(e.state.textgrid!),before);assert.equal(e.state.dirty,false);
 e.state.textgrid!.tiers.push({name:'marks',points:[{number:1,mark:'保留'}]});const original=serializeGrid(e.state.textgrid!);
 e.state.referenceTextGrid={xmin:0,xmax:4,tiers:[{name:'words',intervals:[{xmin:0,xmax:4,text:'不可部分写入'}]},{name:'marks',intervals:[{xmin:0,xmax:4,text:'类型不同'}]}]};
 await assert.rejects(e.applyReferenceSplice(),/类型不同/);assert.equal(serializeGrid(e.state.textgrid!),original);assert.equal(e.state.undoStack.length,0);
});
test('M12 double-click targets visible recording endpoints within six CSS pixels only',()=>{
 assert.equal(endpointClickTime(.004,0,4,4,1000),0);assert.equal(endpointClickTime(3.996,0,4,4,1000),4);
 assert.equal(endpointClickTime(.03,0,4,4,1000),.03);assert.equal(endpointClickTime(1.002,1,1,4,1000),1.002);
 assert.equal(endpointClickTime(1.998,1,1,4,1000),1.998);assert.equal(endpointClickTime(3.999,3,1,4,500),4);
 assert.equal(endpointClickTime(.004,0,4,4,2000),0);assert.equal(endpointClickTime(3.996,0,4,4,2000),4);
});
for(const mode of ['manual','sequence'])for(const end of [2.000062,2.000063,2.0000625])test(`M12 ${mode} can define both recording endpoints despite sub-microsecond WAV/grid rounding (${end})`,()=>{
 const e=createEditor(),grid:Grid={xmin:0,xmax:end,tiers:['words','phones'].map(name=>({name,xmin:0,xmax:end,intervals:[{xmin:0,xmax:end,text:''}]}))};
 e.state.textgrid=grid;e.state.audioBuffer={duration:2.0000625,sampleRate:16000,getChannelData:()=>new Float32Array()};e.state.labSequence=['a1'];
 const click=mode==='manual'?manualDoubleClick:sequenceDoubleClick;click(e,0);assert.equal(e.state.sequenceStart,0);click(e,2.0000625);
 assert.equal(e.state.sequenceStart,null);assert.deepEqual([e.wordTier()!.intervals[e.state.selected!.index].xmin,e.wordTier()!.intervals[e.state.selected!.index].xmax],[0,end]);
 e.undo();assert.equal(e.state.dirty,true);assert.deepEqual(e.wordTier()!.intervals,[{xmin:0,xmax:end,text:''}]);
});
