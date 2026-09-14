import test from 'node:test';import assert from 'node:assert/strict';import fs from 'node:fs';import vm from 'node:vm';
import {createEditor} from '../src/modules/annotation/editor.mjs';
import {parseGrid,serializeGrid,validateEditingAudio} from '../src/modules/annotation/format.ts';
import {lipCurve} from '../src/modules/annotation/display.ts';import {spectrum} from '../src/modules/annotation/spectrum.ts';import {hann,fft} from '../src/modules/annotation/fft.mjs';
// Frozen V2 source from this checkout, hash-equal to the independently inspected V2.
const legacy=fs.readFileSync(new URL('../../phonetic_toolbox/gui/resources/web_praat_editor/app.js',import.meta.url),'utf8');
function old(){const context=vm.createContext({structuredClone,localStorage:{getItem:()=>null},document:{getElementById:()=>({value:'',textContent:''})}});vm.runInContext(legacy.slice(0,legacy.indexOf('document.addEventListener("keydown"'))+'\nglobalThis.api={state,els,fillGaps,pinyinToPhones,replacementWindows,spliceTier,detectIntensityBoundsForWord,localIntensityEnvelope,incrementTone,labelsForIntervalCount,drawLipOpenness,drawIntensity,hann,fft};',context);return context.api;}
const clean=(value:unknown)=>JSON.parse(JSON.stringify(value));
test('M12 V2 fill gaps / all four reference modes / Unicode dictionary fallback',()=>{
  const a=old(),b=createEditor();const intervals=[{xmin:.1,xmax:.6,text:'汉语 æ'},{xmin:.8,xmax:1.4,text:'ba1'}];
  assert.deepEqual(b.fillGaps(intervals,2),clean(a.fillGaps(intervals,2)));
  for(const mode of ['outside','inside','before','after']){
    const windows=b.replacementWindows(mode,.4,1.1,2);assert.deepEqual(windows,clean(a.replacementWindows(mode,.4,1.1,2)));
    const current={name:'words',intervals:b.fillGaps(intervals,2)},reference={name:'words',intervals:[{xmin:0,xmax:2,text:'参考'}]};
    assert.deepEqual(b.spliceTier(current,reference,windows,2),clean(a.spliceTier(current,reference,windows,2)));
  }
  for(const label of ['ba1','zhuang1','iong3','nv3','lü4','汉语','æ',''])assert.deepEqual(b.pinyinToPhones(label),clean(a.pinyinToPhones(label)));
});
test('M12 original RMS envelope and inward/outward boundary estimates are exact',()=>{
  const a=old(),b=createEditor(),samples=Float32Array.from({length:32000},(_,i)=>i>=6000&&i<17000?.35*Math.sin(i*.091):0);
  const audio={duration:2,sampleRate:16000,getChannelData:()=>samples};a.state.audioBuffer=audio;b.state.audioBuffer=audio;
  assert.deepEqual(b.localIntensityEnvelope(.2,1.3),clean(a.localIntensityEnvelope(.2,1.3)));
  // Compare the complete fit result through the V2 envelope-derived boundary path.
  for(const trim of [-.05,-.01,0,.01,.08]){
    const word={xmin:.35,xmax:1.12,text:'ba1'};
    const expected=a.detectIntensityBoundsForWord(word,0,2,trim);
    const actual=(b as any).detectIntensityBoundsForWord(word,0,2,trim);
    assert.deepEqual(actual,clean(expected));assert.ok(actual);
  }
});
test('M12 TextGrid Chinese/IPA/quotes/multiline and point tiers survive save',()=>{
  const grid={xmin:0,xmax:2,tiers:[{name:'词层',intervals:[{xmin:0,xmax:2,text:'井井 "æ"\n第二行'}]},{name:'事件',points:[{number:.5,mark:'ʔ'}]}]};
  const result=parseGrid(serializeGrid(grid));assert.equal(result.tiers[0].intervals![0].text,grid.tiers[0].intervals[0].text);assert.equal(result.tiers[1].points![0].mark,'ʔ');assert.equal(serializeGrid(result),serializeGrid(grid));
});
test('M12 malformed, overlapping, duplicate and excessive grids fail explicitly',()=>{
  assert.throws(()=>parseGrid(''));assert.throws(()=>parseGrid('File type = "ooTextFile"\n"TextGrid" 0 1 <exists> 1 "IntervalTier" "words" 0 1 2 0 .8 "a" .5 1 "b"'));
  const asset={name:'x',sampleRate:16000,frames:32000,duration:2,channels:[new Float32Array()],peaks:[]};assert.equal(validateEditingAudio(asset),asset);
  for(const values of [{sampleRate:4294967295},{sampleRate:1},{frames:8000001},{channels:Array(9).fill(new Float32Array())}])assert.throws(()=>validateEditingAudio({...asset,...values}));
});
test('M12 cannot drag the outer domain or paste beyond a short blank',()=>{
 const e=createEditor();e.state.textgrid={xmin:0,xmax:1,tiers:[{name:'words',intervals:[{xmin:0,xmax:.97,text:'a'},{xmin:.97,xmax:1,text:''}]}]};
 e.moveBoundary({tier:'words',index:0,edge:'start',time:0},.1);assert.equal(e.wordTier()!.intervals[0].xmin,0);
 e.state.selected={tier:'words',index:1};e.state.copiedWord='ba1';e.pasteCopiedWord();assert.equal(e.wordTier()!.intervals.at(-1)!.xmax,1);assert.equal(e.wordTier()!.intervals.at(-1)!.text,'');
});
test('M12 reference splice retains and splices point tiers without losing unrelated content',async()=>{
 const e=createEditor();e.state.textgrid={xmin:0,xmax:2,tiers:[{name:'words',intervals:[{xmin:0,xmax:2,text:'a'}]},{name:'marks',points:[{number:.2,mark:'keep'},{number:1,mark:'old'}]}]};
 e.state.referenceTextGrid={xmin:0,xmax:2,tiers:[{name:'marks',points:[{number:1,mark:'new'}]}]};e.controls.spliceMode.value='inside';e.controls.spliceStart.value='.5';e.controls.spliceEnd.value='1.5';
 await e.applyReferenceSplice();assert.deepEqual(e.state.textgrid.tiers[1].points,[{number:.2,mark:'keep'},{number:1,mark:'new'}]);
});
test('M12 editing and reference reuse preserve unrelated tiers and their own domains',async()=>{
 const e=createEditor(),other={name:'notes',xmin:.1,xmax:1.9,intervals:[{xmin:.2,xmax:.5,text:'keep'}]};
 e.state.textgrid={xmin:0,xmax:2,tiers:[{name:'words',intervals:[{xmin:0,xmax:2,text:'a'}]},structuredClone(other)]};
 e.normalizeTextGrid(e.state.textgrid);assert.deepEqual(e.state.textgrid.tiers[1],other);
 e.state.referenceTextGrid={xmin:0,xmax:2,tiers:[{name:'words',intervals:[{xmin:0,xmax:2,text:'ref'}]}]};
 e.controls.spliceMode.value='after';e.controls.spliceStart.value='1';await e.applyReferenceSplice();assert.deepEqual(e.state.textgrid.tiers[1],other);assert.doesNotThrow(()=>serializeGrid(e.state.textgrid!));
 e.state.wordTierName='notes';assert.equal(e.wordTier(),null);
});
test('M12 accepts original TextGrid microsecond rounding while preserving its domain',()=>{
 const e=createEditor();e.state.textgrid={xmin:0,xmax:3.919796,tiers:[{name:'words',xmin:0,xmax:3.919796,intervals:[{xmin:0,xmax:3.919796,text:'a'}]}]};e.state.audioBuffer={duration:3.9197959183673468,sampleRate:22050,getChannelData:()=>new Float32Array()};
 assert.ok(e.wordTier());e.normalizeTextGrid(e.state.textgrid);assert.equal(e.wordTier()!.intervals[0].xmax,3.919796);
});
test('M12 lip visible neighbours, gaps and display scaling match original coordinates',()=>{
 const a=old(),times=[0,.1,.2,.3,.4],values=[1,2,NaN,4,3],commands:{x:number;y:number;move:boolean}[]=[];
 const ctx={beginPath(){},stroke(){},fillText(){},moveTo(x:number,y:number){commands.push({x,y,move:true});},lineTo(x:number,y:number){commands.push({x,y,move:false});}};
 for(const start of [-.2,.05,.14,.5]){commands.length=0;Object.assign(a.state,{lipData:{times,lipOpen:values},lipOffset:.007,visibleStart:start,visibleDuration:.08});a.drawLipOpenness(ctx,{width:900,height:175});assert.deepEqual(lipCurve(times,values.map(v=>Number.isFinite(v)?v:null),.007,start,.08,900,175),commands);}
});
test('M12 FFT/Hann and 900-point relative intensity display match V2',()=>{
 const a=old();assert.deepEqual([...hann(1024)],[...a.hann(1024)]);const samples=Float32Array.from({length:16000},(_,i)=>i<3000?0:.3*Math.sin(i*.072)),re=Float64Array.from(samples.slice(4000,5024)),im=new Float64Array(1024),expectedRe=re.slice(),expectedIm=im.slice();fft(re,im);a.fft(expectedRe,expectedIm);assert.deepEqual(re,expectedRe);assert.deepEqual(im,expectedIm);
 Object.assign(a.state,{audioBuffer:{sampleRate:16000,getChannelData:()=>samples},visibleStart:0,visibleDuration:1,showLipOpen:false,showLipWidth:false});
 const points:{x:number;y:number}[]=[];const ctx={beginPath(){},stroke(){},fillText(){},moveTo(x:number,y:number){points.push({x,y});},lineTo(x:number,y:number){points.push({x,y});}};a.drawIntensity(ctx,{width:1000,height:175});
 const data=spectrum(samples,16000,0,1,700,175,false,900);assert.equal(points.length,900);data.intensity.forEach((v,i)=>assert.equal(175-v*175,points[i].y));
});
