import test from 'node:test';
import assert from 'node:assert/strict';
import {addTable,replaceTable,removeTable,CustomTableStore,emptyLibrary,parseDelimited,restoreLibrary,syllableKey,tableName,tableRows,type CustomTable} from '../src/modules/mandarin-ipa/custom-tables.ts';
import {convertText,plainOutput,restoreDraft,restoreTableDraft,snapshotDraft,createDraft,variantsFor} from '../src/modules/mandarin-ipa/state.ts';
import {MemoryAssetDatabase} from '../src/platform/local-assets.ts';
import {readTableFile} from '../src/modules/mandarin-ipa/table-file.ts';
import * as XLSX from '../src/modules/perception/vendor/xlsx.mjs';
import {suggestColumns,columnLabel} from '../src/modules/mandarin-ipa/custom-tables.ts';
import {defaultToneSettings,toneClasses,toneSuffix,toneSymbols,tonePreview,transcribeTone} from '../src/modules/mandarin-ipa/tones.ts';

const hanzi:CustomTable={id:'custom:hanzi',name:'调查字音',kind:'hanzi',rows:[['行','hɑŋ˧˥'],['行','ɕiŋ˧˥'],['𠮷','kiʔ']]};
test('M13 imports quoted UTF-8 two-column tables, headers and exact IPA without flattening polyphones',()=>{
  const rows=tableRows(parseDelimited('\uFEFF汉字,国际音标\r\n行,"hɑŋ˧˥"\r\n行,ɕiŋ˧˥\r\n行,ɕiŋ˧˥\r\n𠮷,kiʔ\r\n\r\n'),'hanzi');assert.deepEqual(rows,hanzi.rows);
  assert.deepEqual(parseDelimited('汉字\t国际音标\n妈\tmã\n'),[['汉字','国际音标'],['妈','mã']]);
  assert.deepEqual(parseDelimited('"行","a,b"'),[['行','a,b']]);
  const tokens=convertText('行𠮷妈A😀\n',hanzi.id,{'行_0':1},true,{},hanzi);
  assert.equal(plainOutput(tokens),'ɕiŋ˧˥kiʔ?A😀\n');assert.equal(tokens.length,6);
  assert.equal(tokens[0].kind,'mapped');if(tokens[0].kind==='mapped')assert.deepEqual(variantsFor(tokens[0],hanzi.id).map(o=>o.value),['hɑŋ˧˥','ɕiŋ˧˥']);
  assert.equal(plainOutput(convertText('行',hanzi.id,{},false,{},hanzi)),'hɑŋ');
  const draft={...createDraft(),standard:hanzi.id,selectedVariants:{'行_0':8}};assert.deepEqual(restoreDraft(snapshotDraft(draft)),draft);
});
test('M13 rejects partial, malformed or oversized records before saving',()=>{
  for(const rows of [[['银行','a']],[['妈','']],[['妈','a','oops']],[['妈','a\nb']]])assert.throws(()=>tableRows(rows,'hanzi'),/第 1 行/);
  for(const text of ['汉字,IPA\n"妈,a','a"b,c'])assert.throws(()=>parseDelimited(text),/引号/);
  assert.throws(()=>tableRows([['mǎ4','a']],'syllable'),/第 1 行/);assert.throws(()=>tableRows([],'hanzi'),/没有/);
  for(const name of ['', ' '.repeat(3),'长'.repeat(61)])assert.throws(()=>tableName(name));
  assert.throws(()=>addTable({version:1,tables:[hanzi],selected:hanzi.id},hanzi),/同名/);
  assert.throws(()=>restoreLibrary('{bad'),/无法读取/);
});
test('M13 syllable lookup normalizes tone marks, numeric tones, ü and neutral tone with explicit precedence',()=>{
  for(const value of ['nǚ','nü3','nv3','nu:3'])assert.equal(syllableKey(value),'nü3');
  assert.equal(syllableKey('ma5'),'ma0');assert.equal(syllableKey('ma0'),'ma0');assert.equal(syllableKey('Mā'),'ma1');
  const table:CustomTable={id:'custom:syllables',name:'音节',kind:'syllable',rows:tableRows([['ma','mɑ'],['mā','ma˥'],['ma5','mə'],['nǚ','ny˨˩˦'],['hang2','hɑŋ'],['xing2','ɕiŋ']],'syllable')};
  assert.equal(plainOutput(convertText('妈麻马骂吗女',table.id,{},true,{},table)),'ma˥mɑmɑmɑməny˨˩˦');
  const tokens=convertText('行',table.id,{'行_0':1},true,{},table);assert.equal(plainOutput(tokens),'hɑŋ');
  assert.equal(plainOutput(convertText('花',table.id,{},true,{},table)),'?');
});
test('M13 saved tables reopen independently of drafts, isolate accounts, survive project changes and reject stale writes',async()=>{
  const db=new MemoryAssetDatabase(),a=new CustomTableStore('server:a:project-one:M13',db),b=new CustomTableStore('server:a:project-two:M13',db),other=new CustomTableStore('server:b:project-one:M13',db);
  await a.load();await b.load();await other.load();const next=addTable(emptyLibrary(),hanzi);await a.save(next);
  assert.deepEqual(await b.load(),next);assert.deepEqual(await other.load(),emptyLibrary());
  const stale=new CustomTableStore('server:a:project-one:M13',db);await stale.load();await b.save({...next,selected:null});
  await assert.rejects(()=>stale.save(addTable(next,{...hanzi,id:'custom:second',name:'第二套'})),/另一窗口/);
  assert.deepEqual(await a.load(),{...next,selected:null});
});
test('M13 reads real XLSX sheets and rejects formulas, invalid archives and non-UTF8 text',async()=>{
  const book=XLSX.utils.book_new();XLSX.utils.book_append_sheet(book,XLSX.utils.aoa_to_sheet([['汉字','IPA'],['女','ny˨˩˦']]),'字表');
  const formula=XLSX.utils.aoa_to_sheet([['汉字','IPA'],['女','text']]);formula.B2={t:'s',f:'"ny"',v:'ny'};XLSX.utils.book_append_sheet(book,formula,'含公式');
  const raw=XLSX.write(book,{type:'array',bookType:'xlsx',compression:true});
  const parsed=await readTableFile(new File([raw],'字表.xlsx'));assert.deepEqual(parsed.names,['字表','含公式']);assert.deepEqual(await parsed.rows(0),[['汉字','IPA'],['女','ny˨˩˦']]);await assert.rejects(()=>parsed.rows(1),/公式/);
  await assert.rejects(()=>readTableFile(new File(['bad'],'字表.xlsx')),/有效的 Excel/);
  await assert.rejects(()=>readTableFile(new File([new Uint8Array([0xff])],'字表.csv')),/UTF-8/);
});
test('M13 replacement keeps table identity, revisions stale draft choices, and deletion retains the draft text',()=>{
  const first=addTable(emptyLibrary(),hanzi),saved={...createDraft(),text:'行𠮷',standard:hanzi.id,selectedVariants:{'行_0':1},hanziSize:32};
  assert.equal(restoreTableDraft(saved,first).draft.selectedVariants['行_0'],1);
  const replaced=replaceTable(first,hanzi.id,{name:'新版字音',kind:'hanzi',rows:[['行','a˥'],['行','b˩']]});
  assert.equal(replaced.tables.length,1);assert.equal(replaced.tables[0].id,hanzi.id);assert.equal(replaced.tables[0].revision,2);assert.equal(first.tables[0].name,'调查字音');
  const restored=restoreTableDraft(saved,restoreLibrary(JSON.stringify(replaced)));assert.deepEqual(restored.draft.selectedVariants,{});assert.equal(restored.draft.text,'行𠮷');assert.equal(restored.draft.hanziSize,32);assert.match(restored.notice,/已替换/);
  const current={...restored.draft,selectedVariants:{'行_0':1}};assert.equal(restoreTableDraft(current,replaced).draft.selectedVariants['行_0'],1);
  const removed=removeTable(replaced,hanzi.id);assert.equal(removed.selected,null);assert.equal(removed.tables.length,0);
  const recovered=restoreTableDraft(current,removed);assert.equal(recovered.draft.standard,'Standard Chinese (Beijing)');assert.equal(recovered.draft.text,'行𠮷');assert.deepEqual(recovered.draft.selectedVariants,{});assert.match(recovered.notice,/已删除/);
  assert.throws(()=>removeTable(removed,hanzi.id),/不存在/);
  const two=addTable(first,{...hanzi,id:'custom:other',name:'另一套'});assert.throws(()=>replaceTable(two,hanzi.id,{...hanzi,name:'另一套'}),/同名/);assert.equal(removeTable(two,hanzi.id).selected,'custom:other');
});
test('M13 replacement and deletion persist only after a successful CAS and never revive a removed table',async()=>{
  class RejectingDatabase extends MemoryAssetDatabase{reject=false;override async compare(key:string,expected:unknown,value:unknown){if(this.reject)throw Error('owned write failure');return super.compare(key,expected,value);}}
  const db=new RejectingDatabase(),store=new CustomTableStore('server:a:project:M13',db);await store.load();const first=addTable(emptyLibrary(),hanzi);await store.save(first);
  db.reject=true;await assert.rejects(()=>store.save(removeTable(first,hanzi.id)),/保存失败/);assert.deepEqual(await store.load(),first);
  db.reject=false;const replacement=replaceTable(first,hanzi.id,{...hanzi,rows:[['行','new']]});await store.save(replacement);
  const stale=new CustomTableStore('server:a:other:M13',db);await stale.load();await store.save(removeTable(replacement,hanzi.id));
  await assert.rejects(()=>stale.save(replacement),/另一窗口/);assert.deepEqual(await store.load(),emptyLibrary());
});
test('M13 selects arbitrary columns and exact start row without importing notes, numbers or titles',()=>{
  const raw=[['调查表'],[],['序号','备注','IPA','汉字'],['1','例字','ma55','妈'],['2','另一例','ma35','麻'],['3','空行']];
  assert.deepEqual(suggestColumns(raw),{keyColumn:3,ipaColumn:2,startRow:4});
  assert.deepEqual(tableRows(raw,'hanzi',suggestColumns(raw)),[['妈','ma55'],['麻','ma35']]);
  assert.deepEqual([0,25,26,255].map(columnLabel),['A','Z','AA','IV']);
  assert.throws(()=>tableRows(raw,'hanzi',{keyColumn:2,ipaColumn:2,startRow:4}),/不同/);
  for(const startRow of [0,1.5,8])assert.throws(()=>tableRows(raw,'hanzi',{keyColumn:3,ipaColumn:2,startRow}),/起始行/);
  assert.throws(()=>tableRows([...raw,['4','','bad','银行']],'hanzi',suggestColumns(raw)),/第 7 行/);
  assert.deepEqual(tableRows([['1','ma1','ma55']],'syllable',{keyColumn:1,ipaColumn:2,startRow:1}),[['ma1','ma55']]);
});
test('M13 XLSX keeps leading empty rows and columns so selectors use actual spreadsheet coordinates',async()=>{
  const book=XLSX.utils.book_new(),sheet=XLSX.utils.aoa_to_sheet([['妈','ma55']]);
  const offset=XLSX.utils.aoa_to_sheet([[],[],['','','汉字','IPA'],['','','妈','ma55']]);
  XLSX.utils.book_append_sheet(book,offset,'偏移');XLSX.utils.book_append_sheet(book,sheet,'普通');
  const raw=XLSX.write(book,{type:'array',bookType:'xlsx',compression:true}),read=await readTableFile(new File([raw],'偏移.xlsx')),rows=await read.rows(0);
  assert.equal(rows.length,4);assert.deepEqual(suggestColumns(rows),{keyColumn:2,ipaColumn:3,startRow:4});assert.deepEqual(tableRows(rows,'hanzi',suggestColumns(rows)),[['妈','ma55']]);
});
test('M13 explicit tone values convert all five levels, superscripts and full width without guessing numeric classes',()=>{
  const settings={...defaultToneSettings(),mode:'values' as const};
  assert.equal(toneSymbols('12345'),'˩˨˧˦˥');assert.equal(toneSymbols('⁵⁵'),'˥˥');assert.equal(toneSymbols('２１４'),'˨˩˦');
  assert.equal(transcribeTone('ma 35',settings),'ma˧˥');assert.equal(transcribeTone('ma⁵⁵',settings),'ma˥˥');assert.equal(transcribeTone('ma５１',settings),'ma˥˩');
  assert.equal(transcribeTone('ma˧˥',settings),'ma˧˥');assert.equal(transcribeTone('ma2',defaultToneSettings()),'ma2');
  assert.equal(transcribeTone('ma55',settings,false),'ma');assert.equal(transcribeTone('mɑ̃ʔ214',settings),'mɑ̃ʔ˨˩˦');
  assert.throws(()=>transcribeTone('ma6',settings),/1–5/);assert.throws(()=>toneSymbols('35x'),/调值/);assert.throws(()=>tonePreview([['妈','ma6']],settings),/第 1 条记录/);
});
test('M13 tone classes retain original labels or use explicit user mapping with no Mandarin default',()=>{
  const rows=[['妈','ma¹'],['麻','ma2'],['马','ma３'],['入','jaʔ1a'],['另','jaʔ1b']];
  assert.deepEqual(toneClasses(rows),['1','1a','1b','2','3']);assert.deepEqual(toneSuffix('jaʔ¹A'),{base:'jaʔ',original:'¹A',key:'1a'});
  const keep={...defaultToneSettings(),mode:'classes' as const};assert.equal(transcribeTone('ma¹',keep),'ma¹');assert.equal(transcribeTone('ma¹',keep,false),'ma');
  const mapped={...keep,classOutput:'values' as const,mapping:{'1':'33','2':'21','3':'˨˩˦','1a':'5','1b':'2'}};
  assert.deepEqual(tonePreview(rows,mapped).map(r=>r[1]),['ma˧˧','ma˨˩','ma˨˩˦','jaʔ˥','jaʔ˨']);assert.equal(transcribeTone('ma0',keep),'ma0');assert.throws(()=>transcribeTone('ma0',mapped),/调类 0/);
});
test('M13 tone edits preserve original IPA and survive replacement, CAS persistence, reopen and old library compatibility',async()=>{
  const table:CustomTable={...hanzi,rows:[['妈','ma1'],['麻','ma2']],tones:{mode:'classes',classOutput:'values',mapping:{'1':'55','2':'35'}}};
  const db=new MemoryAssetDatabase(),store=new CustomTableStore('server:tone:project:M13',db);await store.load();let library=addTable(emptyLibrary(),table);await store.save(library);
  const reopened=new CustomTableStore('server:tone:other:M13',db);library=await reopened.load();assert.deepEqual(library.tables[0].rows,table.rows);
  assert.equal(plainOutput(convertText('妈麻',table.id,{},true,{},library.tables[0])),'ma˥˥ma˧˥');assert.equal(plainOutput(convertText('妈麻',table.id,{},false,{},library.tables[0])),'mama');
  library=replaceTable(library,table.id,{...library.tables[0],tones:{...table.tones!,classOutput:'original'}});await reopened.save(library);
  assert.equal(plainOutput(convertText('妈麻',table.id,{},true,{},(await store.load()).tables[0])),'ma1ma2');
  assert.deepEqual(restoreLibrary(JSON.stringify({version:1,tables:[hanzi],selected:hanzi.id})).tables[0].rows,hanzi.rows);
  assert.throws(()=>restoreLibrary(JSON.stringify({version:1,tables:[{...table,tones:{...table.tones,mapping:{'1':'6'}}}],selected:table.id})),/无法读取/);
});
