import test from 'node:test';
import assert from 'node:assert/strict';
import {defaults,valid,exportParams,importParams} from '../src/modules/speech-synthesis/state.ts';
test('M06 R4 old settings retain Klatt and new methods preserve explicit controls',()=>{
 const old=defaults();delete (old as Partial<typeof old>).render;
 assert.equal(valid(old).render.method,'klatt');
 for(const method of ['world','psola'] as const){const c=defaults();c.render={method,pitch:'curve',spectral_ratio:1.2,aperiodicity_ratio:.5,source_sha256:'a'.repeat(64)};c.f0_method='harvest';assert.deepEqual(importParams(exportParams(c)),c);}
});
test('M06 R4 refuses invalid method, non-finite ratios and invalid source identity',()=>{
 for(const patch of [{method:'bad'},{pitch:'bad'},{spectral_ratio:NaN},{spectral_ratio:0},{aperiodicity_ratio:3},{source_sha256:'x'}]){const c=defaults();Object.assign(c.render,patch);assert.throws(()=>valid(c));}
});
