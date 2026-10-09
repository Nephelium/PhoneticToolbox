import test from 'node:test';
import assert from 'node:assert/strict';
import {defaults,valid,exportParams,importParams,f0Methods} from '../src/modules/speech-synthesis/state.ts';
test('M06 R5 keeps R4 Klatt snapshots and rejects retired methods without silently converting',()=>{
 const old={...defaults(),render:{method:'klatt',pitch:'original',spectral_ratio:1,aperiodicity_ratio:1,source_sha256:null}};
 assert.deepEqual(importParams(exportParams(old)),defaults());assert(!Object.hasOwn(old,'render'));
 for(const method of ['world','psola'])assert.throws(()=>valid({...defaults(),render:{method}} as never),/已移除/);
 assert.throws(()=>valid({...defaults(),f0_method:'harvest'} as never),/Harvest 已移除/);
 assert.deepEqual(Object.keys(f0Methods),['praat_cc','praat_ac','reaper']);
});
