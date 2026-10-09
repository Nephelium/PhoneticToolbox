import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {modules} from '../src/app/registry.ts';
const root=new URL('../../',import.meta.url);
const read=(path:string)=>JSON.parse(readFileSync(new URL(path,root),'utf8'));

test('every displayed module source belongs to the global acknowledgement catalogue once',()=>{
 const registry=read('third_party/source-registry.json');
 const sources=read('frontend/src/generated/sources.json');
 const coverage=read('third_party/acknowledgement-coverage.json');
 const ids=sources.map((s:any)=>s.id);
 assert.equal(new Set(ids).size,ids.length);
 assert.equal(registry.total_records,registry.sources.length);
 assert.deepEqual(ids,registry.sources.filter((s:any)=>!s.retired&&s.show_in_acknowledgements!==false).map((s:any)=>s.id));
 assert.deepEqual(coverage.modules.map((m:any)=>m.id).sort(),modules.map(m=>m.id).sort());
 for(const module of coverage.modules){
  assert.deepEqual(module.source_ids,sources.filter((s:any)=>s.modules.includes(module.id)).map((s:any)=>s.id));
 }
 for(const entry of coverage.supplementary_links){
  const source=sources.find((s:any)=>s.id===entry.source_id);
  assert(source?.modules.includes(entry.module),entry.source_id);
  assert(Object.values(source.urls).includes(entry.url),entry.url);
 }
 const local=read('frontend/src/modules/ipa-plus/data/sources.json');
 for(const [key,source]of Object.entries(local) as [string,{url:string}][]){
  assert(coverage.supplementary_links.some((s:any)=>s.module==='M17'&&s.local_source===key&&s.url===source.url),key);
 }
 const model=readFileSync(new URL('frontend/public/vocal-tract/index.html',root),'utf8');
 for(const [,url]of model.matchAll(/href="(https:[^"]+)"/g)){
  assert(coverage.supplementary_links.some((s:any)=>s.module==='M10'&&s.url===url),url);
 }
 const author=sources.find((s:any)=>s.id==='SRC-ZAIWA');
 for(const field of ['attribution','permission_date','adaptation_note','adaptation_caution'])assert(author[field],field);
 assert.equal(author.permission_date,'2026-09-10');
 for(const id of ['REF-M10-JORDAN-2017-MRI','REF-M10-OLIVEIRA-2012-MRI'])assert.equal(sources.find((s:any)=>s.id===id).license,'');
});
