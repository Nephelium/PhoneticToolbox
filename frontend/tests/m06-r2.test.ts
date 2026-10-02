import test from 'node:test';
import assert from 'node:assert/strict';
import {defaults,preset,draw,valid,override,resize,exportParams,importParams,parameters} from '../src/modules/speech-synthesis/state.ts';
const approx=(a:number,b:number)=>assert(Math.abs(a-b)<1e-9,`${a} vs ${b}`);
const curved=()=>{const c=defaults();c.curves.F0.points=[[0,100],[1,160],[2,120]];return c;};
test('M06 R2 F0 translates the full contour and reverses all three neutral presets',()=>{
 for(const voice of ['假声','嘎裂'])for(const neutral of ['常态浊声','耳语','气声']){
  const c=curved(),original=structuredClone(c.curves.F0.points);preset(c,voice);
  assert.equal(c.curves.F0.override,null);assert.equal(c.f0_transform.preset,voice);
  original.forEach(([t,v],i)=>{assert.equal(c.curves.F0.points[i][0],t);approx(c.curves.F0.points[i][1]-v,c.f0_transform.offset_hz);});
  const changed=JSON.stringify(c);preset(c,voice);assert.equal(JSON.stringify(c),changed);
  preset(c,neutral);assert.equal(c.f0_transform.offset_hz,0);original.forEach(([,v],i)=>approx(c.curves.F0.points[i][1],v));
 }
});
test('M06 R2 edited shifted curve survives return and a different shifted preset',()=>{
 const c=curved();preset(c,'假声');const shift=c.f0_transform.offset_hz;draw(c,'F0',.4,.6,shift+150);
 preset(c,'嘎裂');const lowShift=c.f0_transform.offset_hz;approx(c.curves.F0.points.find(([t])=>t===.4)![1],150+lowShift);
 preset(c,'常态浊声');approx(c.curves.F0.points.find(([t])=>t===.4)![1],150);valid(c);
});
test('M06 R2 scalar F0 becomes directly editable; neutral presets do not overwrite it',()=>{
 const c=defaults();override(c,'F0','180');preset(c,'气声');assert.equal(c.curves.F0.override,180);
 preset(c,'假声');assert.equal(c.curves.F0.override,null);assert.deepEqual(c.curves.F0.points,[[0,300],[2,300]]);
 draw(c,'F0',.5,1,340);preset(c,'常态浊声');approx(c.curves.F0.points.find(([t])=>t===.5)![1],220);
});
test('M06 R2 wide contours clamp only the offset, preserving point differences',()=>{
 const c=defaults();c.f0_range=[20,1000];c.curves.F0.points=[[0,30],[1,700],[2,300]];preset(c,'嘎裂');
 assert.equal(c.f0_transform.offset_hz,-10);assert.equal(c.curves.F0.points[1][1]-c.curves.F0.points[0][1],670);valid(c);
});
test('M06 R2 shift metadata survives CSV, JSON and resize',()=>{
 let c=curved();preset(c,'假声');const shift=c.f0_transform.offset_hz;c=importParams(exportParams(c));resize(c,4);
 assert.equal(c.f0_transform.offset_hz,shift);assert.equal(c.curves.F0.points[2][0],4);
 c=importParams(JSON.stringify(c));preset(c,'耳语');approx(c.curves.F0.points[1][1],160);
});
test('M06 R2 graph limits and transactional override validation',()=>{
 assert.deepEqual(parameters.AV.slice(0,3),[60,0,80]);const c=defaults();const before=JSON.stringify(c);
 assert.throws(()=>override(c,'AV','60,200'));assert.equal(JSON.stringify(c),before);
 assert.throws(()=>override(c,'AV','200'));preset(c,'假声');draw(c,'F0',.5,1,-999);valid(c);
 preset(c,'气声');valid(c);assert(c.curves.F0.points.every(([,v])=>v>=1));
});
