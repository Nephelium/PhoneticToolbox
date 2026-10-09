import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {addToneMark,convertText,createDraft,createPageDraft,dataInventory,displayEntry,effectiveIpaSize,mappingStandards,moveVariantChoices,plainOutput,restoreDraft,snapshotDraft,standards,variantsFor} from '../src/modules/mandarin-ipa/state.ts';

test('M13-F01 freezes all eleven legacy columns and the default standard',()=>{
  assert.equal(standards.length,11);assert.deepEqual(standards,[...mappingStandards,'汉语拼音']);
  assert.equal(createDraft().standard,'Standard Chinese (Beijing)');
  const expected=['ma˥','ma̠˥','mᴀ˥','mᴀ˥','mᴀ˥','mᴀ˥','mᴀ˥','mᴀ˥','ma˥','mä˥','mā'];
  assert.deepEqual(standards.map(standard=>plainOutput(convertText('妈',standard))),expected);
});

test('M13-F01 keeps the legacy pinyin tone placement and neutral tone rules',()=>{
  assert.equal(addToneMark('hao',3),'hǎo');assert.equal(addToneMark('liu',2),'liú');assert.equal(addToneMark('ou',4),'òu');assert.equal(addToneMark('hua',0),'hua');
});

test('M13-F01 exposes ambiguity per position without pretending to resolve context',()=>{
  const first=convertText('银行','Standard Chinese (Beijing)严');const token=first[1];assert.equal(token.kind,'mapped');if(token.kind!=='mapped')return;
  assert.equal(token.value,'ɕiŋ˧˥');assert.equal(token.variants.length,4);
  const options=variantsFor(token,'Standard Chinese (Beijing)严');assert.deepEqual(options.map(option=>option.pinyin),['xing','hang','hang','heng']);
  assert.equal(plainOutput(convertText('银行','Standard Chinese (Beijing)严',{'行_1':1})).endsWith('xa̝ŋ˧˥'),true);
  assert.equal(plainOutput(convertText('行','汉语拼音',{'行_0':1})),'háng');
});

test('M13-F01 preserves punctuation, Latin text, whitespace and explicit newlines',()=>{
  const text='妈 A1，\n花';const tokens=convertText(text,'Standard Chinese (Beijing)严');
  assert.equal(plainOutput(tokens),'ma̠˥ A1，\nxu̟a˥');assert.equal(tokens.filter(token=>token.kind==='literal'&&token.newline).length,1);
});

test('M13-F02/F03 restores only bounded layout state and keeps the IPA-only legacy default size',()=>{
  const defaults=createDraft();assert.equal(effectiveIpaSize(defaults),16);defaults.display='ipa-only';assert.equal(effectiveIpaSize(defaults),28);defaults.ipaSizeUserSet=true;assert.equal(effectiveIpaSize(defaults),16);
  const restored=restoreDraft({...defaults,hanziSize:999,ipaSize:9,gap:-99,lineHeight:9,standard:'invented',selectedVariants:{'行_0':2,'bad':9}});
  assert.equal(restored.hanziSize,24);assert.equal(restored.ipaSize,16);assert.equal(restored.gap,0);assert.equal(restored.lineHeight,1.8);assert.equal(restored.standard,'Standard Chinese (Beijing)');assert.deepEqual(restored.selectedVariants,{'行_0':2});
});

test('M13-R2 hides only tone letters and preserves segmental diacritics and pinyin ü',()=>{
  assert.equal(plainOutput(convertText('妈麻马骂吗','Standard Chinese (Beijing)',{},false)),'mamamamama');
  assert.equal(plainOutput(convertText('妈','Standard Chinese (Beijing)严',{},false)),'ma̠');
  assert.equal(plainOutput(convertText('妈','UntPhesoca严',{},false)),'mä');
  assert.equal(plainOutput(convertText('妈女略','汉语拼音',{},false)),'manülüe');
  assert.equal(plainOutput(convertText('妈女略','汉语拼音')),'mānǚlüè');
  const token=convertText('行','Standard Chinese (Beijing)严',{'行_0':1},false)[0];assert.equal(token.kind,'mapped');if(token.kind!=='mapped')return;
  assert.equal(token.value,'xa̝ŋ');assert.equal(token.entry.tone,2);assert.equal(token.selectedVariant,1);
  assert.equal(variantsFor(token,'Standard Chinese (Beijing)严',false)[1].value,'xa̝ŋ');
  assert.equal(plainOutput(convertText('A1，\n妈','Standard Chinese (Beijing)',{},false)),'A1，\nma');
});

test('M13-R2 starts empty with Beijing while explicit restore preserves the complete old draft',()=>{
  const saved={...createDraft(),text:'银行',standard:'UntPhesoca严',selectedVariants:{'行_1':1},hanziSize:32,ipaColor:'#123abc',hanziColor:'#fedcba',hanziFont:'KaiTi',showTone:false};
  const fresh=createPageDraft(saved);assert.equal(fresh.text,'');assert.equal(fresh.standard,'Standard Chinese (Beijing)');assert.deepEqual(fresh.selectedVariants,{});assert.equal(fresh.hanziSize,32);assert.equal(fresh.showTone,false);
  assert.equal(restoreDraft(saved).text,'银行');assert.deepEqual(restoreDraft(saved).selectedVariants,{'行_1':1});
  const roundtrip=restoreDraft(snapshotDraft(restoreDraft(saved)));assert.deepEqual(roundtrip,saved);
  const malformed=restoreDraft({...saved,ipaColor:'red;',hanziColor:'<bad>',hanziFont:'bad;family',showTone:'false'});
  assert.equal(malformed.ipaColor,'');assert.equal(malformed.hanziColor,'');assert.equal(malformed.hanziFont,'');assert.equal(malformed.showTone,true);
  const legacy=restoreDraft({version:1,text:'妈',standard:'UntPhesoca严'});assert.equal(legacy.showTone,true);assert.equal(legacy.hanziFont,'');
});

test('M13-R2 moves choices with unchanged characters and discards replaced positions',()=>{
  assert.deepEqual(moveVariantChoices('行行','行',{'行_0':1,'行_1':2}),{'行_0':1});
  assert.deepEqual(moveVariantChoices('银行','😀银行',{'行_1':1}),{'行_2':1});
  assert.deepEqual(moveVariantChoices('银行','河行',{'行_1':1}),{'行_1':1});
  assert.deepEqual(moveVariantChoices('银行','行行',{'行_0':1,'行_1':2}),{'行_1':2});
  assert.deepEqual(moveVariantChoices('银行','新的文本',{'行_1':1}),{});
});

test('M13 long text conversion is complete and deterministic',()=>{
  const text='妈行花。\n'.repeat(500),first=convertText(text,'UntPhesoca严'),second=convertText(text,'UntPhesoca严');
  assert.equal(first.length,[...text].length);assert.deepEqual(first,second);assert.equal(first.filter(token=>token.kind==='mapped').length,1500);
});

test('M13 local mapping and Doulos asset match the reviewed legacy resources',()=>{
  assert.deepEqual(dataInventory,{schemaVersion:'m13-ipa-data/1',sourcePath:'phonetic_toolbox/gui/resources/ipa_trans/ipa_converter.html',sourceSha256:'11a5c8cc7315d04bc2beece64519247ab86f80e7e9fb37510704b8d1e4ec6451',rows:21572,uniqueCharacters:20771,duplicateCharacters:681,maxVariants:7});
  const hash=(url:URL)=>createHash('sha256').update(readFileSync(url)).digest('hex');
  assert.equal(hash(new URL('../src/assets/DoulosSIL-Regular.ttf',import.meta.url)),hash(new URL('../../phonetic_toolbox/gui/resources/ipa_trans/DoulosSIL-Regular.ttf',import.meta.url)));
  assert.ok(readFileSync(new URL('../src/assets/Doulos-OFL.txt',import.meta.url),'utf8').includes('SIL OPEN FONT LICENSE'));
});

test('M13 runtime sources have no CDN or external text-processing endpoint',()=>{
  for(const name of ['state.ts','export.ts','MandarinIpaPage.vue']){
    const source=readFileSync(new URL('../src/modules/mandarin-ipa/'+name,import.meta.url),'utf8');assert.doesNotMatch(source,/https?:\/\//i,name);
  }
});
