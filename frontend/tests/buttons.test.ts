import {test} from 'node:test';
import assert from 'node:assert/strict';
import {normalizeButtons} from '../src/design/buttons.ts';
test('P19-R3 malformed button preferences recover without discarding independently valid choices',()=>{
 for(const saved of [null,undefined,42,'all',[],{mode:'unknown',effects:'false'}])assert.deepEqual(normalizeButtons(saved),{mode:'auto',effects:true,material:'solid',glassOpacity:35,glassBlur:12});
 assert.deepEqual(normalizeButtons({mode:'plain',effects:false}),{mode:'plain',effects:false,material:'solid',glassOpacity:35,glassBlur:12});
 assert.deepEqual(normalizeButtons({mode:'invalid',effects:false}),{mode:'auto',effects:false,material:'solid',glassOpacity:35,glassBlur:12});
 assert.deepEqual(normalizeButtons({mode:'all',effects:null,unrelated:'keep'}),{mode:'all',effects:true,material:'solid',glassOpacity:35,glassBlur:12});
 for(const mode of ['auto','all','plain'] as const)for(const effects of [true,false])for(const material of ['solid','glass'] as const)assert.deepEqual(normalizeButtons(JSON.parse(JSON.stringify({mode,effects,material}))),{mode,effects,material,glassOpacity:35,glassBlur:12});
 assert.equal(normalizeButtons({material:'invalid'}).material,'solid');
 assert.deepEqual(normalizeButtons({glassOpacity:0,glassBlur:0}),{mode:'auto',effects:true,material:'solid',glassOpacity:0,glassBlur:0});
 assert.equal(normalizeButtons({glassOpacity:200}).glassOpacity,100);
 assert.equal(normalizeButtons({glassBlur:99}).glassBlur,40);
 assert.equal(normalizeButtons({glassBlur:NaN,glassOpacity:'20'}).glassOpacity,35);
});
