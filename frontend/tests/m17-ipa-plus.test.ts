import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {insertSymbol,EditorHistory} from '../src/modules/ipa-plus/editor.ts';
import {graphemes,safeSelection,codePoints,suspiciousCharacters} from '../src/modules/ipa-plus/unicode.ts';
import type {Catalog,SymbolEntry} from '../src/modules/ipa-plus/types.ts';
const catalog=JSON.parse(fs.readFileSync(new URL('../src/modules/ipa-plus/data/catalog.json',import.meta.url),'utf8')) as Catalog;
test('M17 catalogue has complete declared counts and unique, attributable entries',()=>{
 assert.equal(catalog.entries.length,625);assert.equal(new Set(catalog.entries.map(e=>e.id)).size,625);
 assert.deepEqual(Object.fromEntries(['ipa','extipa','voqs'].map(s=>[s,catalog.entries.filter(e=>e.system===s).length])),{ipa:351,extipa:209,voqs:65});
 for(const e of catalog.entries){assert(e.nameZh&&e.nameEn&&e.descriptionZh&&e.usageZh&&e.contrastZh&&e.sourceRefs.length);assert.deepEqual(e.codePoints,codePoints(e.insertText));assert(!/[\u0000-\u001f\ue000-\uf8ff]/u.test(e.insertText));assert(!e.insertText.includes('◌'));}
});
test('M17 every occurrence resolves exactly once in its system',()=>{
 for(const system of ['ipa','extipa','voqs'] as const){const ids=catalog.charts[system].flatMap(s=>s.ids);assert.equal(new Set(ids).size,ids.length);assert.deepEqual(ids.slice().sort(),catalog.entries.filter(e=>e.system===system).map(e=>e.id).sort());
  for(const section of catalog.charts[system]){if(section.kind==='matrix')assert.deepEqual(section.rows!.flatMap(r=>r.cells.flatMap(c=>c.ids)).sort(),section.ids.slice().sort());if(section.kind==='vowels')assert.deepEqual(section.points!.flatMap(p=>p.ids).sort(),section.ids.slice().sort());}
 }
});
test('M17 every symbol produces exact text at selection and one undoable transaction',()=>{
 for(const e of catalog.entries){const h=new EditorHistory({text:'甲𝼆乙',start:1,end:e.insertionMode==='bridge'?1:3});const next=insertSymbol(h.current,e);const changed=h.commit(next);assert.equal(h.current.text,e.insertionMode==='paired-span'?'甲'+e.prefix+'𝼆'+e.suffix+'乙':e.insertionMode==='bridge'?'甲'+e.insertText+'𝼆乙':'甲'+e.insertText+'乙');if(changed){assert.equal(h.undo()!.text,'甲𝼆乙');assert.equal(h.redo()!.text,next.text);}else assert.equal(h.canUndo,false);}
});
test('M17 supplementary and combining selection boundaries never split graphemes',()=>{
 assert.deepEqual(safeSelection('a𝼆ãb',2,2),{start:3,end:3});assert.deepEqual(safeSelection('a𝼆ãb',4,5),{start:3,end:5});assert.equal(graphemes('𝼆ã').length,2);
});
test('M17 bridge requires exactly two graphemes and preserves their diacritics',()=>{
 const e={insertText:'͡',insertionMode:'bridge'} as SymbolEntry;
 assert.equal(insertSymbol({text:'t̼θ̼',start:0,end:4},e).text,'t̼͡θ̼');
 assert.throws(()=>insertSymbol({text:'abc',start:0,end:3},e),/两个字素/);
});
test('M17 empty paired spans place caret inside; combining input adds no placeholder',()=>{
 assert.deepEqual(insertSymbol({text:'x',start:1,end:1},{insertText:'{}',insertionMode:'paired-span',prefix:'{V! ',suffix:' V!}'}),{text:'x{V!  V!}',start:5,end:5});
 assert.equal(insertSymbol({text:'',start:0,end:0},{insertText:'̤',insertionMode:'combining'}).text,'̤');
});
test('M17 mixed native and command history drops redo only after new edit',()=>{
 const h=new EditorHistory();h.commit({text:'汉字',start:2,end:2});h.commit({text:'汉字𝼆',start:4,end:4});assert.equal(h.undo()!.text,'汉字');h.select(0,1);h.commit({text:'甲字',start:1,end:1});assert.equal(h.canRedo,false);assert.deepEqual(h.undo(),{text:'汉字',start:0,end:1});
});
test('M17 suspicious user text is reported without mutation',()=>{
 const text='a\uf267\n\u202e';assert.deepEqual(suspiciousCharacters(text).map(s=>s.code),['U+F267','U+202E']);assert.equal(text,'a\uf267\n\u202e');
});
test('M17 VoQS actual 2016 base inventory retains 56 table entries',()=>{
 assert.equal(catalog.entries.filter(e=>e.system==='voqs'&&e.section!=='scope').length,56);
 assert(catalog.entries.some(e=>e.insertText==='V𐞀'&&e.nameEn==='aryepiglottic phonation'));
 assert(catalog.entries.some(e=>e.insertText==='Vꟸ'&&e.nameEn==='faucalized voice'));
});
test('M17 all 56 VoQS names exactly match the user-specified UntPhesoca translation',()=>{
 const terms=JSON.parse(fs.readFileSync(new URL('../src/modules/ipa-plus/data/voqs-zh-names.json',import.meta.url),'utf8')).terms;
 const expected=['颊气流','肺部吸气言语','食管气流','气管–食管言语','常态浊声','假声','耳语','嘎裂','耳语声','嘎裂声','气声','糙声','耳语假声','嘎裂假声','糙假声','糙嘎裂','糙耳语声','糙嘎裂声','耳语嘎裂声','糙耳语嘎裂声','耳语嘎裂假声','糙耳语嘎裂假声','弛/松声','挤喉发声/紧声','室襞性发声','复音','耳语室襞性发声','杓状会厌襞性发声','痉挛性发声障碍','电子喉发声','喉位偏高声','喉位偏低声','唇化声（开圆唇）','唇化声（闭圆唇）','展唇声','唇齿化声','舌尖化声','舌叶化声','卷舌声','齿化声','龈化声','腭–龈化声','腭化声','软腭化声','小舌化声','咽化声','喉–咽化声','咽门化声','鼻化声','去鼻化声','颌位偏开声','颌位偏闭声','颌位偏右声','颌位偏左声','颌位突出声','舌突出声'];
 assert.deepEqual(terms.map((t:{nameZh:string})=>t.nameZh),expected);
 for(const term of terms){const entry=catalog.entries.find(e=>e.id===term.symbolId)!;assert.equal(entry.nameZh,term.nameZh);assert.equal(entry.nameEn,term.nameEn);assert(entry.sourceRefs.some(s=>s.sourceId==='voqs-zhihu-203037479'));assert(entry.aliases.includes(term.nameZh.split('/')[0]));}
});
test('M17 extIPA partial parentheses, uncertainty and text downgrade are explicit',()=>{
 for(const value of ['̥᪽','̥᫃','̥᫄','̊᪻','̊᫁','̊᫂','̬᪽','̬᫃','̬᫄','C⃝','Ȼ⃝','Ṽ⃝','Ʞ⃝','σ⃝','⟅n̥ã⟆'])assert(catalog.entries.some(e=>e.insertText===value),value);
 assert(catalog.entries.filter(e=>e.insertText.includes('⟅')).every(e=>!!e.representation));
});
test('M17 R1 independently checked screenshot combinations remain distinct from basic symbols',()=>{
 const additions=catalog.entries.filter(e=>e.section==='combinations');
 assert.equal(additions.filter(e=>e.system==='ipa').length,107);
 assert.equal(additions.filter(e=>e.system==='extipa').length,4);
 for(const value of ['pʰ','t̪ʰ','r̪','s̠','ɻ̊','t͡ɕ','ɖ͡ʐ','ɓ̥','q͡χʼ','p͆͡f͆']){
  const e=additions.find(e=>e.insertText===value)!;assert(e,value);assert.equal(e.notationStatus,'combination');assert.equal(e.isExample,true);
 }
 for(const value of ['r̪','r̠','ⱱ̟'])assert(!additions.find(e=>e.insertText===value)!.descriptionZh.includes('清化圈'));
});
test('M17 R1 group rows cover every listed entry exactly once',()=>{
 for(const system of ['ipa','extipa'] as const)for(const section of catalog.charts[system]){
  if(section.kind!=='list')continue;assert(section.groups?.length,section.id);
  assert.deepEqual(section.groups!.flatMap(g=>g.ids).sort(),section.ids.slice().sort(),section.id);
  assert(section.groups!.every(g=>g.label&&g.hint));
 }
});
test('M17 R1 old extIPA direction is explicitly sourced to the 2002 chart discussion',()=>{
 const historical=catalog.entries.filter(e=>e.notationStatus==='historical');
 assert.deepEqual(historical.map(e=>e.insertText),['↑','t↑']);
 for(const e of historical){assert.equal(e.section,'historical');assert(e.descriptionZh.includes('2002'));assert(e.descriptionZh.includes('2025'));assert(e.sourceRefs.some(s=>s.sourceId==='extipa-lv-jiang-2013'));}
 const denasal=catalog.entries.find(e=>e.system==='extipa'&&e.insertText==='͊')!;
 assert.equal(denasal.nameZh,'部分去鼻化');assert(denasal.aliases.includes('非鼻化'));
});
test('M17 R1 all source references resolve to available attributed links',()=>{
 const sources=JSON.parse(fs.readFileSync(new URL('../src/modules/ipa-plus/data/sources.json',import.meta.url),'utf8'));
 for(const e of catalog.entries)for(const source of e.sourceRefs){assert(sources[source.sourceId]?.title,source.sourceId);assert(sources[source.sourceId]?.url.startsWith('https://'));}
 assert(sources['extipa-lv-jiang-2013'].title.includes('江荻'));
});
