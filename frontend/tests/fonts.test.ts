import {test} from 'node:test';
import assert from 'node:assert/strict';
import {defaults,normalizeFonts,fontKey,figureFonts,quoteFamily} from '../src/design/fonts.ts';
test('F01/F02 fixed IPA survives imported preferences and English changes',()=>{
 const p=normalizeFonts({...defaults(),latin:'Times New Roman',ipa:'Arial'});
 assert.equal(p.ipa,'Doulos SIL');assert.equal(p.latin,'Times New Roman');
});
test('F03 corrupt or future preferences fall back without accepting CSS or paths',()=>{
 assert.deepEqual(normalizeFonts(null),defaults());
 assert.deepEqual(normalizeFonts({version:99,zh:'SimSun'}),defaults());
 assert.equal(normalizeFonts({...defaults(),zh:'bad";color:red',size:999}).zh,defaults().zh);
 assert.equal(normalizeFonts({...defaults(),mono:'C:/font.ttf'}).mono,defaults().mono);
 assert.equal(quoteFamily('Times New Roman'),'"Times New Roman"');
});
test('F05 figure inheritance and independent fonts do not mutate preferences',()=>{
 const p=defaults();p.zh='SimSun';p.latin='Times New Roman';p.figure.zh='KaiTi';
 assert.equal(figureFonts(p).zh,'SimSun');p.figure.follow=false;
 assert.equal(figureFonts(p).zh,'KaiTi');assert.equal(p.zh,'SimSun');assert.equal(figureFonts(p).ipa,'Doulos SIL');
});
test('F09 preferences are scoped to owner and not project or shared session',()=>{
 assert.notEqual(fontKey('alice'),fontKey('bob'));assert.notEqual(fontKey(),fontKey('alice'));
});
