import {test} from 'node:test';
import assert from 'node:assert/strict';
import {setLocale} from '../src/i18n/index.ts';
import {parameterLabel} from '../src/i18n/parameters.ts';
import parameters from '../src/generated/parameters.json' with {type:'json'};
test('parameter display labels change language while all 80 scientific keys and source labels remain unchanged',async()=>{
 const source=structuredClone(parameters),keys=parameters.map(row=>row.key);
 await setLocale('en');assert.equal(parameterLabel('LipArea','唇面积'),'Lip area');
 await setLocale('ja');assert.equal(parameterLabel('LipOpen','唇开度'),'口唇開口度');
 for(const row of parameters)if(!['LipArea','LipWidth','LipOpen','LipCirc'].includes(row.key))assert.equal(parameterLabel(row.key,row.label),row.label);
 await setLocale('zh-CN');assert.equal(parameterLabel('LipCirc','唇圆度'),'唇圆度');
 assert.deepEqual(parameters,source);assert.deepEqual(parameters.map(row=>row.key),keys);assert.equal(keys.length,80);
});
