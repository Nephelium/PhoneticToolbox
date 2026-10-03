import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import type {Catalog} from '../src/modules/ipa-plus/types.ts';
const data=JSON.parse(fs.readFileSync(new URL('../src/modules/ipa-plus/data/catalog.json',import.meta.url),'utf8')) as Catalog;
const ipa=(value:string)=>data.entries.find(e=>e.system==='ipa'&&e.insertText===value&&!e.isExample)!;

test('M17 R2 names match independently transcribed user chart 4 terms',()=>{
 const expected:Record<string,string>={
  'p':'清双唇爆发音','ɾ':'浊龈拍音或闪音','ʔ':'喉爆发音','ʘ':'双唇啧音','ǃ':'龈(后)啧音','ɗ':'齿/龈浊内爆音','ʼ':'喷音记号',
  'ʍ':'唇-软腭清擦音','ɧ':'同时发ʃ和x','ɺ':'龈边浊闪音','ʡ':'会厌爆发音',
  '̹':'更圆','̜':'略展','̟':'偏前','̠':'偏后','̽':'中-央化','̯':'不成音节','˞':'r音色','̤':'气声性','̰':'嘎裂声性','̼':'舌唇',
  '̝':'偏高','̞':'偏低','̘':'舌根偏前','̙':'舌根偏后','̺':'舌尖性','̻':'舌叶性','ⁿ':'鼻除阻','ˡ':'边除阻','̚':'无闻除阻',
  'ː':'长','ˑ':'半长','̆':'超短','|':'小(音步)组块','‖':'大(语调)组块','.':'音节间隔','‿':'连接(间隔不出现)',
  '˥':'超高调（调符）','˩':'超低调（调符）','ꜜ':'降阶','ꜛ':'升阶','↗':'整体上升','↘':'整体下降',
  'i':'闭前不圆唇元音','e':'半闭前不圆唇元音','ɛ':'半开前不圆唇元音','a':'开前不圆唇元音',
 };
 for(const [symbol,name] of Object.entries(expected))assert.equal(ipa(symbol)?.nameZh,name,symbol);
 assert.equal(data.charts.ipa.find(s=>s.id==='tones')!.title,'声调与词重调');
 assert(ipa('p').aliases.includes('清双唇塞音'));
 for(const e of data.entries){
  for(const key of ['nameZh','descriptionZh','usageZh','contrastZh'] as const)assert(!e[key].includes('嗒音'),e.id+':'+key);
  assert(e.sourceRefs.every(r=>!r.locator.includes('嗒音')),e.id);
  assert(e.aliases.every(a=>!a.includes('嗒音')),e.id);
 }
});

test('M17 R2 merged consonant placement preserves scope and all original IDs',()=>{
 const matrix=data.charts.ipa.find(s=>s.id==='extended')!;
 assert.equal(matrix.rows!.length,13);assert.equal(matrix.columns!.length,14);
 assert(matrix.columns!.includes('龈'));assert(matrix.columns!.includes('喉'));
 assert(matrix.subtitle.includes('空格不表示构音不可能'));
 assert(matrix.rows!.every(r=>r.cells.every(c=>!c.shaded&&!c.rightHalfShaded)));
 const values=(row:string,column:string)=>matrix.rows!.find(r=>r.label===row)!.cells[matrix.columns!.indexOf(column)]!.ids.map(id=>data.entries.find(e=>e.id===id)!.insertText);
 assert(values('喷音','龈后').includes('t͡ʃʼ'));assert(values('内爆音','小舌').includes('ʛ̥'));
 assert(values('爆发音','龈后').includes('t̠ʲ'));assert(!values('爆发音','龈-腭').includes('t̠ʲ'));
 assert(values('啧音','龈后').includes('ǃ'));
 assert(data.entries.find(e=>e.id==='ipa-pulmonic-003')?.insertText);
 const ids=data.charts.ipa.flatMap(s=>s.ids);assert.equal(ids.length,351);assert.equal(new Set(ids).size,351);
});

test('M17 R2 explanations cite exact editions and preserve the VoQS and extIPA distinctions',()=>{
 for(const e of data.entries.filter(e=>e.system==='ipa')){
  assert(e.sourceRefs.some(r=>r.sourceId==='ipa-chart-zh-2007'));
  assert(e.sourceRefs.some(r=>r.sourceId==='ipa-handbook-jiang-2008'&&r.locator.includes('PDF')));
 }
 const voqs=(v:string)=>data.entries.find(e=>e.system==='voqs'&&e.insertText===v)!;
 assert(voqs('V̤').descriptionZh.includes('相对开放'));assert(voqs('Ṿ').descriptionZh.includes('杓会厌'));
 assert(voqs('V̤').sourceRefs.some(s=>s.sourceId==='voice-quality-esling-2019'&&s.locator.includes('56–58')));
 assert(voqs('V‼').sourceRefs.some(s=>s.locator.includes('71–73')));
 assert(voqs('V͊').nameZh==='去鼻化声');
 const dento=data.entries.find(e=>e.system==='extipa'&&e.section==='articulatory'&&e.insertText==='͆')!;
 assert(dento.descriptionZh.startsWith('上唇与下齿'));assert(dento.sourceRefs.some(s=>s.sourceId==='extipa-ball-2018'));
 const denasal=data.entries.find(e=>e.system==='extipa'&&e.insertText==='͊')!;
 assert.equal(denasal.nameZh,'部分去鼻化');assert(denasal.sourceRefs.some(s=>s.sourceId==='extipa-ball-2024'));
 for(const e of data.entries)for(const r of e.sourceRefs)assert(!/[A-Z]:[\\/]/.test(r.locator));
});
