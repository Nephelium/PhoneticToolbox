import test from 'node:test';import assert from 'node:assert/strict';
import {createEditor} from '../src/modules/annotation/editor.mjs';
import {loadEditingGrid,parseGrid,serializeGrid,preferredGrid,validateEditingAudio} from '../src/modules/annotation/format.ts';
import {sequenceDoubleClick,sequenceParts,resetSequence} from '../src/modules/annotation/sequence.ts';
import {newIntervalTiers} from '../src/modules/annotation/layers.ts';
function setup(){const e=createEditor();e.state.textgrid=loadEditingGrid(undefined,2).grid;e.state.textgrid.tiers.push(...newIntervalTiers(e.state.textgrid,['words','phones']));e.state.labSequence=['zhe4','shi4','shang4'];resetSequence(e);return e;}
const words=(e:ReturnType<typeof setup>)=>e.wordTier()!.intervals.filter(i=>i.text);
const phones=(e:ReturnType<typeof setup>)=>e.phoneTier()!.intervals.filter(i=>i.text);
test('M12-R2 missing, whitespace and valid zero-tier TextGrid allow explicitly named empty tiers',()=>{
 for(const text of [undefined,'',' \r\n\t','"ooTextFile short" "TextGrid" 0 2 <exists> 0']){
  const {grid,created}=loadEditingGrid(text,2);assert.equal(created,true);assert.equal(grid.tiers.length,0);grid.tiers.push(...newIntervalTiers(grid,['音节','音素']));
  assert.deepEqual(parseGrid(serializeGrid(grid)).tiers[0].intervals,[{xmin:0,xmax:2,text:''}]);
 }
 assert.throws(()=>loadEditingGrid('malformed content',2));
 assert.throws(()=>newIntervalTiers(loadEditingGrid(undefined,2).grid,['same','same']));
});
test('M12-R1 endpoints create aligned syllable and phone then split at clicked time',()=>{
 const e=setup();sequenceDoubleClick(e,.2);assert.equal(e.state.sequenceIndex,0);assert.equal(words(e).length,0);
 sequenceDoubleClick(e,.7);assert.deepEqual(words(e),[{xmin:.2,xmax:.7,text:'zhe4'}]);assert.deepEqual(phones(e),words(e));
 sequenceDoubleClick(e,.35);assert.deepEqual(phones(e),[{xmin:.2,xmax:.35,text:'zh'},{xmin:.35,xmax:.7,text:'e4'}]);assert.equal(e.state.sequenceIndex,1);
 const before=serializeGrid(e.state.textgrid!);assert.throws(()=>sequenceDoubleClick(e,.5),/已切分/);assert.equal(serializeGrid(e.state.textgrid!),before);
});
test('M12-R1 Ctrl uses the live previous end; ordinary endpoints preserve gaps; undo restores progress',()=>{
 const e=setup();sequenceDoubleClick(e,.2);sequenceDoubleClick(e,.7);sequenceDoubleClick(e,1.1,true);
 assert.deepEqual(words(e)[1],{xmin:.7,xmax:1.1,text:'shi4'});
 e.undo();assert.equal(e.state.sequenceIndex,1);assert.equal(words(e).length,1);
 sequenceDoubleClick(e,.9);sequenceDoubleClick(e,1.4);assert.deepEqual(words(e)[1],{xmin:.9,xmax:1.4,text:'shi4'});
 assert(e.wordTier()!.intervals.some(i=>i.xmin===.7&&i.xmax===.9&&i.text===''));
 sequenceDoubleClick(e,2,true);assert.equal(words(e)[2].text,'shang4');assert.equal(words(e)[2].xmax,2);
 assert.throws(()=>sequenceDoubleClick(e,1.9,true));assert.equal(e.state.sequenceIndex,3);
});
test('M12-R1 failed/overlapping or reversed endpoints do not consume labels or alter tiers',()=>{
 const e=setup();assert.throws(()=>sequenceDoubleClick(e,.3,true),/起点/);sequenceDoubleClick(e,.4);
 const before=serializeGrid(e.state.textgrid!);assert.throws(()=>sequenceDoubleClick(e,.2));assert.equal(e.state.sequenceIndex,0);assert.equal(serializeGrid(e.state.textgrid!),before);
 sequenceDoubleClick(e,.8);sequenceDoubleClick(e,.1);assert.throws(()=>sequenceDoubleClick(e,1));assert.equal(e.state.sequenceIndex,1);
 assert.equal(e.state.sequenceStart,.1);assert.equal(words(e).length,1);
});
test('M12-R1 onset/rime keeps the whole final and neutral tones; zero initial stays one interval',()=>{
 const e=setup();for(const [label,expected] of [['zhe4',['zh','e4']],['shang4',['sh','ang4']],['ding4',['d','ing4']],['men5',['m','en5']],['yi1',['yi1']],['ai4',['ai4']],['nv3',['n','v3']]] as const)assert.deepEqual(sequenceParts(e,label),expected);
 e.state.phoneDict=new Map([['æ',['a','e']]]);assert.deepEqual(sequenceParts(e,'æ'),['a','e']);
});
test('M12-R1 loading a word list resumes a matching prefix and undo cannot restore another list cursor',()=>{
 const e=setup();sequenceDoubleClick(e,.2);sequenceDoubleClick(e,.7);resetSequence(e);assert.equal(e.state.sequenceIndex,1);
 e.state.labSequence=['new1'];resetSequence(e);e.undo();assert.equal(e.state.sequenceIndex,0);assert.equal(e.state.sequenceStart,null);
});
test('M12-R1 Chinese recovery suffix matches uniquely and long recording stays within shared sample budget',()=>{
 const wav={name:'中文/女1_六条原始音频拼接.wav'},grid={name:'中文/女1_六条原始音频拼接_初始定位.TextGrid'};
 assert.equal(preferredGrid(wav,[wav,grid]),grid);assert.throws(()=>preferredGrid(wav,[grid,{name:'中文/女1_六条原始音频拼接_其他.TextGrid'}]),/多个/);
 const asset={name:wav.name,sampleRate:44100,frames:13784432,duration:13784432/44100,channels:[new Float32Array()]};
 assert.equal(validateEditingAudio(asset),asset);assert.throws(()=>validateEditingAudio({...asset,channels:Array(3).fill(new Float32Array())}));
});
test('M12-R1 group 1ms nudge keeps every selected phone duration and unrelated tier, and is undoable',()=>{
 const e=setup();sequenceDoubleClick(e,.2);sequenceDoubleClick(e,.7);sequenceDoubleClick(e,.35);sequenceDoubleClick(e,1.1,true);
 e.state.textgrid!.tiers.push({name:'events',points:[{number:.6,mark:'keep'}]});const before=serializeGrid(e.state.textgrid!);
 e.state.selectedIndices=e.wordTier()!.intervals.flatMap((w,i)=>w.text?[i]:[]);const oldPhones=phones(e).map(p=>({...p}));
 e.moveSelected(.001);assert.equal(words(e)[0].xmin,.201);assert.equal(words(e)[1].xmax,1.101);
 phones(e).forEach((p,i)=>{assert.ok(Math.abs(p.xmin-oldPhones[i].xmin-.001)<1e-12);assert.ok(Math.abs((p.xmax-p.xmin)-(oldPhones[i].xmax-oldPhones[i].xmin))<1e-12);});
 assert.deepEqual(e.state.textgrid!.tiers[2].points,[{number:.6,mark:'keep'}]);e.undo();assert.equal(serializeGrid(e.state.textgrid!),before);
});
test('M12-R1 moving an adjacent syllable collision or past recording end rejects the entire edit',()=>{
 const e=setup();sequenceDoubleClick(e,.2);sequenceDoubleClick(e,.7);sequenceDoubleClick(e,1.1,true);
 const before=serializeGrid(e.state.textgrid!),history=e.state.undoStack.length;e.state.selected={tier:'words',index:1};e.state.selectedIndices=[];
 assert.throws(()=>e.moveSelected(.001),/重叠/);assert.equal(serializeGrid(e.state.textgrid!),before);assert.equal(e.state.undoStack.length,history);
 e.state.selectedIndices=[1,2];assert.throws(()=>e.moveSelected(-.201),/范围/);assert.equal(serializeGrid(e.state.textgrid!),before);
});
test('M12-R1 editing a sequential syllable label preserves its manually placed internal split',()=>{
 const e=setup();sequenceDoubleClick(e,.2);sequenceDoubleClick(e,.7);sequenceDoubleClick(e,.35);
 e.editText('zhe5',sequenceParts(e,'zhe5'));assert.deepEqual(phones(e),[{xmin:.2,xmax:.35,text:'zh'},{xmin:.35,xmax:.7,text:'e5'}]);e.undo();assert.equal(words(e)[0].text,'zhe4');
});
