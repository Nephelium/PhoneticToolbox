import {test} from 'node:test';
import assert from 'node:assert/strict';
import {matchingTable,visibleParameters,assignParameters,removeGroup,series,annotationRuns,overlayPlot} from '../src/modules/parameter-display/state.ts';
import type {ParameterTable,ResearchFile} from '../src/platform/research.ts';
const audio:ResearchFile={id:'a',name:'阴平.wav',kind:'audio',size:12};
test('M02 paired legacy formats prefer SQLite without guessing duplicates',()=>{const files:ResearchFile[]=[{id:'x',name:'阴平.xlsx',kind:'parameter',size:1},{id:'s',name:'阴平.ptb.sqlite',kind:'parameter',size:1}];assert.equal(matchingTable(audio,files)?.id,'s');assert.equal(matchingTable(audio,[...files,{...files[1],id:'duplicate'}]),undefined);});
test('M02 independent reaper/correction filters do not mutate selection',()=>{const names=['pF0','rF0','H1*-H2*','H1-H2'];assert.deepEqual(visibleParameters(names,'',false,false),['pF0','H1-H2']);assert.deepEqual(visibleParameters(names,'',false,true),['pF0','H1*-H2*','H1-H2']);assert.equal(names.length,4);});
test('M02 bulk assignment and merge preserve every parameter exactly once',()=>{const initial=[{id:1,title:'图1',parameters:['F0','Intensity','TextGrid']},{id:2,title:'图2',parameters:[]}];const groups=assignParameters(initial,['F0','TextGrid','F0'],2);assert.deepEqual(groups.map(g=>g.parameters),[['Intensity'],['F0','TextGrid']]);assert.deepEqual(initial[0].parameters,['F0','Intensity','TextGrid']);assert.deepEqual(removeGroup(groups,1)[0].parameters,['F0','TextGrid','Intensity']);assert.throws(()=>assignParameters(groups,['F0'],3));});
const table:ParameterTable={schema_version:'m02/1',sha256:'a'.repeat(64),columns:['Time_s','pF0','TextGrid'],kinds:['number','number','text'],rows:[[0,100,'a'],[.01,110,'a'],[.02,null,'b'],[.03,120,'b'],[.04,'Infinity','']]};
test('M02 original time and missing values remain explicit gaps',()=>{assert.deepEqual(series(table,'pF0',0,.05,1),[{time:0,value:100},{time:.01,value:110},{time:.02,value:null},{time:.03,value:120},{time:.04,value:null}]);assert.deepEqual(annotationRuns(table,'TextGrid',.005,.035),[{start:.005,end:.02,text:'a'},{start:.02,end:.035,text:'b'}]);});
test('M02 extrema retained in pixel decimation',()=>{const data={...table,rows:[[0,100,''],[.001,500,''],[.002,1,''],[.003,100,'']]};assert.deepEqual(series(data,'pF0',0,1,1).map(p=>p.value),[100,500,1,100]);});
test('M02 manual 2.2 multiple curves share one coordinate scale, without normalization',()=>{
  const data:ParameterTable={...table,columns:['Time_s','pF0','rF0'],kinds:['number','number','number'],rows:[[0,100,100],[.01,150,200],[.02,null,300]]};
  const plot=overlayPlot(data,['pF0','rF0'],0,.03);
  assert.equal(plot.dual,false);assert.deepEqual(plot.curves.map(c=>c.axis),['left','left']);
  assert.deepEqual(plot.left,{min:90,max:310});assert.deepEqual(plot.curves[0].points.map(p=>p.value),[100,150,null]);
});
test('M02 automatic right axis preserves v2 strict 50 ratio and 100 magnitude thresholds',()=>{
  const data:ParameterTable={...table,columns:['Time_s','small','large'],kinds:['number','number','number'],rows:[[0,2,100],[.01,2,100]]};
  assert.equal(overlayPlot(data,['small','large'],0,.02).dual,false);
  data.rows=[[0,2,101],[.01,2,101]];assert.deepEqual(overlayPlot(data,['small','large'],0,.02).curves.map(c=>c.axis),['left','right']);
  data.rows=[[0,0,1000],[.01,0,1000]];assert.equal(overlayPlot(data,['small','large'],0,.02).dual,false);
  data.rows=[[0,1,100],[.01,1,100]];assert.equal(overlayPlot(data,['small','large'],0,.02).dual,false);
});
test('M02 axis classification uses original visible frames, independent of display decimation',()=>{
  const data:ParameterTable={...table,columns:['Time_s','a','b'],kinds:['number','number','number'],rows:[[0,1,1],[.001,1,300],[.002,1,1],[.003,1,1],[.5,1,900]]};
  const tiny=overlayPlot(data,['a','b'],0,.004,1),wide=overlayPlot(data,['a','b'],0,.004,900);
  assert.equal(tiny.curves[1].mean,75.75);assert.deepEqual(tiny.curves.map(c=>c.axis),wide.curves.map(c=>c.axis));assert.deepEqual(tiny.left,wide.left);
});
