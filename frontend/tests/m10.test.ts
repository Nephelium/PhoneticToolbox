import test from 'node:test';
import assert from 'node:assert/strict';
import {frameIntervals,silentAt} from '../public/vocal-tract/keyframes.js';
import {moveControl,PoseHistory} from '../public/vocal-tract/controls.mjs';
import {areaReadout} from '../public/vocal-tract/area-view.mjs';
import {readFileSync} from 'node:fs';
import {registerNose} from '../public/vocal-tract/anatomy.mjs';
import {planeSegments,stitch,sectionPaths} from '../public/vocal-tract/geometry.mjs';
import {presetSymbol,presetInputName} from '../public/vocal-tract/presets.js';
import {sectionView} from '../public/vocal-tract/section-view.mjs';
import {canvasFont} from '../public/vocal-tract/fonts.js';
import {buildPresetChart} from '../src/modules/vocal-tract/preset-chart.ts';
import type {Catalog} from '../src/modules/ipa-plus/types.ts';

test('M10-R11 preserves lateral air at midline contact and reports complete closure',()=>{
  const upper=Array(96).fill(0),lower=Array(96).fill(0);upper[20]=.5;upper[21]=.5;
  const state={section:0,centerline:[[1,2,3,0,1]],airway_sections:[{upper,lower}],tube_lengths:[2,2],tube_areas:[2,.12]};
  const lateral=areaReadout(state);assert.equal(lateral.midlineClosed,true);assert.equal(lateral.closed,false);
  assert.ok(lateral.geometric>0);assert.equal(lateral.acoustic,.12);assert.equal(lateral.position,3);
  upper.fill(0);state.tube_areas[1]=.0001;
  const sealed=areaReadout(state);assert.equal(sealed.closed,true);assert.equal(sealed.geometric,0);assert.equal(sealed.acoustic,.0001);
});

test('M10-R11 undo/redo records an independent larynx edit without mutating hyoid parameters',()=>{
  const h=new PoseHistory(),before={params:[.13,-3.9],larynx_height:0},after={...before,larynx_height:.65};
  h.begin(before);h.commit(after);
  assert.deepEqual(h.undo(after),before);assert.deepEqual(h.redo(before),after);
  assert.deepEqual(before.params,[.13,-3.9]);
});

test('M10-R9 includes the entire Plus matrix, literal examples and the original 107 symbols',()=>{
  const catalog=JSON.parse(readFileSync(new URL('../src/modules/ipa-plus/data/catalog.json',import.meta.url),'utf8')) as Catalog;
  const chart=buildPresetChart(catalog),matrix=catalog.charts.ipa.find(s=>s.id==='extended')!;
  assert.equal(chart.consonants.columns.length,14);assert.equal(chart.consonants.rows.length,13);
  assert.deepEqual(chart.consonants.columns,matrix.columns);
  assert.deepEqual(chart.consonants.rows,matrix.rows!.map(row=>({label:row.label,cells:row.cells.map(cell=>cell.ids)})));
  const ids=chart.consonants.rows.flatMap(row=>row.cells.flat());assert.equal(ids.length,186);
  const entries=new Map(chart.entries.map(e=>[e.id,e]));
  for(const id of ids){const e=catalog.entries.find(e=>e.id===id)!;assert.equal(e.insertionMode,'literal');assert.deepEqual(entries.get(id),{id,display:e.display,insertText:e.insertText,nameZh:e.nameZh});}
  const symbols=new Set(chart.entries.map(e=>e.insertText));
  for(const symbol of ['pʰ','m̥','t͡s','ɓ','pʼ','ʘ'])assert.ok(symbols.has(symbol),symbol);
  const base=catalog.entries.filter(e=>e.system==='ipa'&&!e.isExample&&e.insertionMode==='literal'&&e.display!=='ʼ'&&['pulmonic','vowels','nonpulmonic','other'].includes(e.section));
  assert.equal(base.length,107);for(const e of base)assert.ok(entries.has(e.id));
  assert.deepEqual(chart.vowels,catalog.charts.ipa.find(s=>s.id==='vowels')!.points);
  assert.deepEqual(chart.extras,base.filter(e=>['nonpulmonic','other'].includes(e.section)).map(e=>e.id));
  for(const name of ['pʰ','/pʰ/','[m̥]','t͡s'])assert.ok(symbols.has(presetInputName(name,symbols)));
});

test('M10-R8 cross-section centimetres are isotropic and fixed at native zero',()=>{
  for(const [width,height] of [[320,240],[200,120],[700,400]]){
    const view=sectionView(width,height);
    assert.ok(view.scale>0);
    assert.equal(view.x(48),width/2);
    assert.equal(view.y(0),view.centerY);
    assert.ok(Math.abs((view.x(48+96/7)-view.x(48))-(view.y(0)-view.y(1)))<1e-10);
    assert.ok(view.x(0)>=16&&view.x(96)<=width-16);
    assert.ok(view.y(3.5)>=12-1e-10&&view.y(-3.5)<=height-26+1e-10);
  }
  assert.equal(sectionView(0,0).scale,0);
});

test('M10-R8 canvas captions respect the 12 px floor including fixed IPA labels',()=>{
  for(const ipa of [false,true])for(const size of [8,9,10,11,12,14,24])assert.ok(parseFloat(canvasFont(size,ipa))>=12);
  assert.equal(parseFloat(canvasFont(24)),24);
});

test('M10-R7 custom IPA preserves combining sequences and validates storage names',()=>{
  const symbols=new Set(['a','ɡ']);
  for(const name of ['t͡s','ã','n̥','pʰ','[t͡s]','𝼆'])assert.equal(presetInputName(' '+name+' ',symbols),name);
  assert.equal(presetInputName(' /a/ ',symbols),'a');
  assert.equal(presetInputName('𝼆'.repeat(80),symbols),'𝼆'.repeat(80));
  for(const name of ['', '   ', 't\ns','a'.repeat(81),'a\u0000'])assert.throws(()=>presetInputName(name,symbols));
});

test('M10-R6 maps old bare, slash and bracket IPA names without guessing free names',()=>{
  const symbols=new Set(['a','ɡ','ɧ']);
  for(const name of ['a','/a/','[a]',' /a/ '])assert.equal(presetSymbol(name,symbols),'a');
  assert.equal(presetSymbol('[ɧ]',symbols),'ɧ');
  for(const name of ['自存 /a/','g','al','',undefined])assert.equal(presetSymbol(name,symbols),null);
});
test('M10 interval names and exact consecutive boundaries',()=>{
  const frames=[{name:'/n/',duration:.6},{name:'/a/',duration:.3},{preset:'i',duration:1}];
  const out=frameIntervals(frames);assert.equal(out[0].name,'/n/');assert.equal(out[1].start,.6);assert.equal(out[2].end,1.9);
  frames[0].duration=1;assert.equal(frameIntervals(frames)[1].start,1);
});
test('M10 side drag reaches native lateral bracing range',()=>{
  const meta={parameters:[{name:'TS3',min:-1,max:1}]};
  assert.ok(moveControl(meta,[0],'side3',0,-1).params[0]<-.8);
  assert.equal(moveControl(meta,[0],'side3',0,.5).params[0],.15);
  assert.equal(moveControl(meta,[0],'side3',0,10).params[0],1);
});
test('M10 silent intervals exclude F0 at exact start and include next pose at end',()=>{
  const frames=[{duration:.2},{duration:.05,silent:true},{duration:.25}];
  assert.equal(silentAt(frames,.3999),false);
  assert.equal(silentAt(frames,.4),true);
  assert.equal(silentAt(frames,.4999),true);
  assert.equal(silentAt(frames,.5),false);
  assert.equal(frameIntervals(frames)[1].name,'静音');
  assert.equal(silentAt([{duration:.05,silent:true}],1),true);
});
test('M10 tongue root and hyoid have two-axis control',()=>{
  const meta={parameters:['TRX','TRY','HX','HY'].map(name=>({name,min:-10,max:10}))};
  assert.deepEqual(moveControl(meta,[0,0,0,0],'root',.2,-.3).params,[.2,-.3,0,0]);
  assert.deepEqual(moveControl(meta,[0,0,0,0],'hyoid',.1,-.1).params,[0,0,.1,-.1]);
});
test('M10 nasal sections stay within the reference head without artificial floor chords',()=>{
  const load=(name:string)=>JSON.parse(readFileSync(new URL('../public/vocal-tract/assets/'+name+'.json',import.meta.url),'utf8'));
  const original=load('nasal'),nose=registerNose(original),head=load('head');
  assert.equal(nose.triangles,original.triangles);
  assert.equal(nose.vertices.length,original.vertices.length);
  for(let i=0;i<head.vertices.length;i+=3){const [x,y,z]=head.vertices.slice(i,i+3);head.vertices.splice(i,3,(z-.127)*100+6.5,(y-1.85)*85-1,x*80);}
  const envelope=stitch(planeSegments(head.vertices,head.triangles))[0];
  const inside=(p:number[])=>{let value=false;for(let i=0,j=envelope.length-1;i<envelope.length;j=i++){const a=envelope[i],b=envelope[j];if((a[1]>p[1])!==(b[1]>p[1])&&p[0]<(b[0]-a[0])*(p[1]-a[1])/(b[1]-a[1])+a[0])value=!value;}return value;};
  for(const z of [-.5,.5]){
    const sections=sectionPaths(nose,z);
    for(const path of sections){
      assert.ok(Math.hypot(...path[0].map((v:number,k:number)=>v-path.at(-1)[k]))<.002,'slice already closed before SVG rendering');
      assert.ok(path.every(inside),'nasal reference protrudes beyond the head');
      for(let i=1;i<path.length;i++)assert.ok(Math.hypot(path[i][0]-path[i-1][0],path[i][1]-path[i-1][1])<2,'spurious long closing chord');
    }
  }
});
