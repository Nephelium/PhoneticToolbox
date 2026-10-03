import {test} from 'node:test';
import assert from 'node:assert/strict';
import {normalizeButtons} from '../src/design/buttons.ts';
test('P19-R3 malformed button preferences recover without discarding independently valid choices',()=>{
 for(const saved of [null,undefined,42,'all',[],{mode:'unknown',effects:'false'}])assert.deepEqual(normalizeButtons(saved),{mode:'auto',effects:true});
 assert.deepEqual(normalizeButtons({mode:'plain',effects:false}),{mode:'plain',effects:false});
 assert.deepEqual(normalizeButtons({mode:'invalid',effects:false}),{mode:'auto',effects:false});
 assert.deepEqual(normalizeButtons({mode:'all',effects:null,unrelated:'keep'}),{mode:'all',effects:true});
 for(const mode of ['auto','all','plain'] as const)for(const effects of [true,false])assert.deepEqual(normalizeButtons(JSON.parse(JSON.stringify({mode,effects}))),{mode,effects});
});
