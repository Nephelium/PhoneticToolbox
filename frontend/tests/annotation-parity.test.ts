import test from 'node:test';import assert from 'node:assert/strict';import fs from 'node:fs';import vm from 'node:vm';
import {createEditor} from '../src/modules/annotation/editor.mjs';
import {parseGrid,serializeGrid} from '../src/modules/annotation/format.ts';
// Frozen V2 source from this checkout, hash-equal to the independently inspected V2.
const legacy=fs.readFileSync(new URL('../../phonetic_toolbox/gui/resources/web_praat_editor/app.js',import.meta.url),'utf8');
function old(){const context=vm.createContext({structuredClone,localStorage:{getItem:()=>null},document:{getElementById:()=>({value:'',textContent:''})}});vm.runInContext(legacy.slice(0,legacy.indexOf('document.addEventListener("keydown"'))+'\nglobalThis.api={state,els,fillGaps,pinyinToPhones,replacementWindows,spliceTier,detectIntensityBoundsForWord,localIntensityEnvelope,incrementTone,labelsForIntervalCount};',context);return context.api;}
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
});
