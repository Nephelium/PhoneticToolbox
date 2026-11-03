import test from 'node:test';
import assert from 'node:assert/strict';
import {messageRef,setLocale,tr} from '../src/i18n/index.ts';
import {uiErrorMessage} from '../src/modules/module-localization.ts';
import {readTableFile} from '../src/modules/mandarin-ipa/table-file.ts';
import {createDraft,restoreDraft,persistDraft} from '../src/modules/ipa-plus/storage.ts';

test('one failed M13 file read and invalid M17 restoration relocalize without a retry or data change',async()=>{
 await setLocale('en');
 let reads=0;const file={name:'原始字表.xlsx',size:24,arrayBuffer:async()=>{reads++;return new ArrayBuffer(24);}} as File;
 let failure:unknown;try{await readTableFile(file);}catch(error){failure=error;}
 assert(failure instanceof Error);const imported=messageRef();imported.value=uiErrorMessage(failure);
 const saved={version:9,text:'用户𝼆内容',revision:4,system:'ipa'},snapshot=JSON.stringify(saved);
 const restored=[restoreDraft(null,'writer'),restoreDraft(saved,'writer')];
 const notices=restored.map(value=>{const state=messageRef();state.value=value.messageDisplay??value.message;return state;});
 for(const language of ['en','ja','zh-CN','en'] as const){
  await setLocale(language);
  assert.equal(imported.value,tr('modules.mandarin-ipa.text.082077c9b644'));
  assert.deepEqual(notices.map(value=>value.value),['ed3270fa2210','7e63187713a9'].map(key=>tr('modules.ipa-plus.text.'+key)));
  assert.equal(reads,1);assert.equal(JSON.stringify(saved),snapshot);assert.equal(restored[1].draft.text,saved.text);
 }
 assert(restored.every(value=>value.blocked));await setLocale('zh-CN');
});

test('one IndexedDB draft conflict retains its descriptor through each locale',async()=>{
 const original=globalThis.indexedDB;let previous:unknown,putFailure=false;
 const database={transaction:()=>{
  const tx:any={objectStore:()=>({get:()=>{const request:any={};queueMicrotask(()=>{request.result=previous;request.onsuccess();});return request;},put:()=>{if(putFailure)throw Error('quota');queueMicrotask(()=>tx.oncomplete());}}),abort:()=>queueMicrotask(()=>tx.onabort())};return tx;
 },close:()=>{}};
 globalThis.indexedDB={open:()=>{const request:any={};queueMicrotask(()=>{request.result=database;request.onsuccess();});return request;}} as unknown as IDBFactory;
 try{
  for(const [stored,revision,throwPut,key] of [
   [{version:1,revision:2},1,false,'7d3ef6f7e3e5'],
   [undefined,1,false,'3ab191fcc964'],
   [undefined,0,true,'82ca8642a75c']
  ] as const){
   previous=stored;putFailure=throwPut;await setLocale('en');
   let failure:unknown;try{await persistDraft('unchanged-key',createDraft('same-writer'),revision);}catch(error){failure=error;}
   assert(failure instanceof Error);const value=messageRef();value.value=uiErrorMessage(failure);
   for(const language of ['en','ja','zh-CN'] as const){await setLocale(language);assert.equal(value.value,tr('modules.ipa-plus.text.'+key));}
  }
 }finally{globalThis.indexedDB=original;await setLocale('zh-CN');}
});


test('one failed M14 job retains the error code descriptor and current column-mapping guidance',async()=>{
 const {m14Port,m14Error,m14ErrorMessage}=await import('../src/platform/m14.ts');
 let jobs=0;const port=m14Port({project:'same-project',upload:async()=>{throw Error('unused');},create:async()=>{jobs++;return {state:'failed',error_code:'m14_no_valid_rows'}as any;},job:async()=>{throw Error('no retry');},cancel:async()=>{},read:async()=>new ArrayBuffer(0)});
 await setLocale('en');let failure:unknown;
 try{await port.inspectSource({asset_id:'same-file',sha256:'same-hash',name:'原始字表.xlsx'},{}as any,new AbortController().signal);}catch(error){failure=error;}
 assert(failure instanceof Error);const shown=messageRef();shown.value=uiErrorMessage(failure);
 for(const language of ['en','ja','zh-CN']as const){await setLocale(language);assert.equal(shown.value,tr('platform.no_valid_records_check_the_column_numbers_start_row'));assert.equal(m14Error('m14_unsupported_format'),tr('platform.xlsx_xls_csv_tsv_txt_and_docx_tables_are'));}
 assert.equal(jobs,1);assert.equal(m14ErrorMessage('unknown 原始错误'),'unknown 原始错误');await setLocale('zh-CN');
});
