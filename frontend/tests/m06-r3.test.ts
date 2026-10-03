import test from 'node:test';
import assert from 'node:assert/strict';
import {defaults,preset,valid,exportParams,importParams} from '../src/modules/speech-synthesis/state.ts';

test('M06 R3 requested voice levels keep source presets and F0 independent',()=>{
 for(const [voice,av,ah] of [['气声',65,35],['常态浊声',70,0],['嘎裂',70,0],['假声',70,0],['耳语',0,40]] as const){
  const c=defaults();preset(c,voice);
  assert.equal(c.curves.AV.override,av);assert.equal(c.curves.AH.override,ah);
  assert.equal(importParams(exportParams(c)).curves.AV.override,av);
 }
});
test('M06 R3 F0 choice survives parameter roundtrip and old v2 files default to CC',()=>{
 for(const method of ['praat_cc','praat_ac','reaper'] as const){
  const c=defaults();c.f0_method=method;
  assert.equal(importParams(exportParams(c)).f0_method,method);
  assert.equal(importParams(JSON.stringify(c)).f0_method,method);
 }
 const old=JSON.parse(JSON.stringify(defaults()));delete old.f0_method;
 assert.equal(importParams(JSON.stringify(old)).f0_method,'praat_cc');
 assert.throws(()=>valid({...defaults(),f0_method:'unknown' as never}));
 assert.throws(()=>valid({...defaults(),f0_method:null as never}));
});
