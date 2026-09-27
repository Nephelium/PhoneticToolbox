import test from 'node:test';import assert from 'node:assert/strict';
import {createEditor} from '../src/modules/annotation/editor.mjs';
import {manualDoubleClick,splitInitialPhones,hasUnsplitPhones} from '../src/modules/annotation/sequence.ts';
import {serializeGrid,parseGrid,preferredGrid} from '../src/modules/annotation/format.ts';
import {sampleWavePath} from '../src/platform/waveform.ts';
const setup=()=>{const e=createEditor();e.state.textgrid={xmin:0,xmax:2,tiers:[{name:'words',intervals:[{xmin:0,xmax:.5,text:'a'},{xmin:.5,xmax:.502,text:'b'},{xmin:.502,xmax:2,text:''}]},{name:'phones',intervals:[{xmin:0,xmax:.5,text:'a'},{xmin:.5,xmax:.502,text:'b'},{xmin:.502,xmax:2,text:''}]}]};return e;};
test('M12-R3 word/phone boundaries can move to 0.1ms apart and serialize without collapsing',()=>{
 const e=setup();e.state.selectedBoundary={tier:'words',time:.5};e.moveBoundary({tier:'words',index:0,edge:'end',time:.5},.5019);
 assert.equal(e.state.selectedBoundary.time,.5019,'keyboard boundary selection follows the actual drag');
 assert.equal(e.wordTier()!.intervals[1].xmin,.5019);assert.equal(e.phoneTier()!.intervals[1].xmin,.5019);
 e.moveBoundary({tier:'phones',index:0,edge:'end',time:.5019},.5018);assert.equal(e.phoneTier()!.intervals[1].xmin,.5018);assert.equal(e.wordTier()!.intervals[1].xmin,.5019);
 assert.equal(parseGrid(serializeGrid(e.state.textgrid!)).tiers[1].intervals![1].xmax,.502);
 e.moveBoundary({tier:'words',index:0,edge:'end',time:.5019},.5017);assert.equal(e.phoneTier()!.intervals[1].xmin,.5018,'nearby but distinct phone must stay independent');
});
test('M12-R3 close phone split and exact boundary deletion preserve neighbouring tiny intervals',()=>{
 const e=setup();e.splitPhoneAt(.501);assert.equal(e.phoneTier()!.intervals.length,4);
 e.state.selectedBoundary={tier:'phones',time:.501};e.deleteSelectedBoundary();assert.equal(e.phoneTier()!.intervals.length,3);assert.equal(e.phoneTier()!.intervals[1].xmin,.5);assert.equal(e.phoneTier()!.intervals[1].xmax,.502);
});
test('M12-R3 existing document blank supports endpoint creation without word list and undo',()=>{
 const e=setup(),before=serializeGrid(e.state.textgrid!);manualDoubleClick(e,.8);manualDoubleClick(e,.8001);
 assert.equal(e.wordTier()!.intervals[e.state.selected!.index].xmax,.8001);assert(e.phoneTier()!.intervals.some(i=>i.xmin===.8&&i.xmax===.8001));e.undo();assert.equal(serializeGrid(e.state.textgrid!),before);
});
test('M12-R3 manual endpoints reject overlap atomically and scan prefers Chinese autosave with legacy fallback',()=>{
 const e=setup(),before=serializeGrid(e.state.textgrid!);assert.throws(()=>manualDoubleClick(e,.1));manualDoubleClick(e,.7);assert.throws(()=>manualDoubleClick(e,.6));assert.equal(serializeGrid(e.state.textgrid!),before);
 const audio={name:'中文.wav'},old={name:'中文_webedit.TextGrid'},fresh={name:'中文_自动保存.TextGrid'};
 assert.equal(preferredGrid(audio,[old,fresh]),fresh);assert.equal(preferredGrid(audio,[old]),old);
});
test('M12-R3 hit test selects the closest dense boundary instead of the first in the hit radius',()=>{
 const e=setup();e.state.visibleDuration=2;globalThis.window={devicePixelRatio:1} as Window&typeof globalThis;
 e.setCanvas({width:1000,height:150,getBoundingClientRect:()=>({left:0,top:0})} as HTMLCanvasElement);
 const hit=e.hitTest({clientX:250.95,clientY:30} as MouseEvent)!;assert.equal(hit.index,1);assert.equal(hit.edge,'end');
});
test('M12-R3 detailed waveform connects actual signed samples at their original fractional-window times',()=>{
 const samples=new Float32Array([0,.5,-.5,1,0]),before=samples.slice();
 const path=sampleWavePath(samples,.5,3.5,1,800)!;
 assert.equal(path.split('M').length-1,1);assert.equal(path.split('L').length-1,4);assert(!path.includes('V'));
 const coords=path.slice(1).split(' L').map(p=>p.split(',').map(Number));
 for(let i=0;i<coords.length;i++){assert.equal(coords[i][0],(i-.5)/3*1000);assert.equal(coords[i][1],45-samples[i]*37);}
 assert.deepEqual(samples,before);
});
test('M12-R3 detailed waveform remains bounded and leaves long-view peak aggregation available',()=>{
 const samples=new Float32Array(100000);samples[80]=2;
 assert.equal(sampleWavePath(samples,0,100000,2,800),null);
 assert(!sampleWavePath(samples,20.5,30.5,1,800)!.includes('NaN'));
 assert(sampleWavePath(samples,99998.5,100000,1,800)!.split('L').length<=2);
});
test('M12-R3 first phone boundary follows cursor or equal option; remaining phones share the remaining duration',()=>{
 const e=setup(),word=e.wordTier()!.intervals[0],before=serializeGrid(e.state.textgrid!);
 splitInitialPhones(e,word,.1,'cursor',['a','b','c']);assert.deepEqual(e.phoneTier()!.intervals.slice(0,3).map(i=>[i.xmin,i.xmax]),[[0,.1],[.1,.3],[.3,.5]]);
 const split=serializeGrid(e.state.textgrid!);assert(!hasUnsplitPhones(e,word));assert.throws(()=>splitInitialPhones(e,word,.2,'equal',['a','b','c']));assert.equal(serializeGrid(e.state.textgrid!),split);e.undo();assert.equal(serializeGrid(e.state.textgrid!),before);
 splitInitialPhones(e,e.wordTier()!.intervals[0],.1,'equal',['a','b','c']);assert.deepEqual(e.phoneTier()!.intervals.slice(0,3).map(i=>i.xmax),[.166667,.333333,.5]);
});
test('M12-R3 initial phone split preserves surrounding blank and rejects an invalid endpoint before mutation',()=>{
 const e=setup(),word=e.wordTier()!.intervals[1];e.phoneTier()!.intervals=[{xmin:0,xmax:2,text:''}];const before=serializeGrid(e.state.textgrid!);
 assert.throws(()=>splitInitialPhones(e,word,.502,'cursor',['b','a']));assert.equal(serializeGrid(e.state.textgrid!),before);
 splitInitialPhones(e,word,.5001,'cursor',['b','a']);assert.deepEqual(e.phoneTier()!.intervals.map(i=>[i.xmin,i.xmax]),[[0,.5],[.5,.5001],[.5001,.502],[.502,2]]);
});
