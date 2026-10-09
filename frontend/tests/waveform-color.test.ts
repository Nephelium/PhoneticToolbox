import {test} from 'node:test';
import assert from 'node:assert/strict';
import {normalizeHex,normalizeWaveformAppearance,waveformCss} from '../src/design/waveform-color.ts';
test('P19-R4 colors normalize valid hex and reject CSS expressions or transparent colors',()=>{
 assert.equal(normalizeHex(' #A3F '),'#aa33ff');assert.equal(normalizeHex('#AABBCC'),'#aabbcc');
 for(const invalid of [null,42,'red','transparent','#12345678','#12','var(--accent)','url(x)',';color:red'])assert.equal(normalizeHex(invalid),null);
});
test('P19-R4 malformed preferences recover while preserving independently valid choices',()=>{
 for(const invalid of [null,undefined,[],42,'custom'])assert.deepEqual(normalizeWaveformAppearance(invalid),{mode:'theme',custom:'#2463eb'});
 assert.deepEqual(normalizeWaveformAppearance({mode:'broken',custom:'#f08'}),{mode:'theme',custom:'#ff0088'});
 assert.deepEqual(normalizeWaveformAppearance({mode:'blue',custom:'#f08'}),{mode:'blue',custom:'#ff0088'});
 assert.deepEqual(normalizeWaveformAppearance({mode:'custom',custom:'transparent'}),{mode:'custom',custom:'#2463eb'});
 assert.equal(waveformCss({mode:'blue',custom:'#aa33ff'}),'var(--wave)');assert.equal(waveformCss({mode:'theme',custom:'#aa33ff'}),'var(--accent)');assert.equal(waveformCss({mode:'custom',custom:'#aa33ff'}),'#aa33ff');
});
