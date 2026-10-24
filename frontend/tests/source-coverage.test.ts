import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {modules} from '../src/app/registry.ts';
import {standards} from '../src/modules/mandarin-ipa/state.ts';
const root=new URL('../../',import.meta.url);
const read=(path:string)=>JSON.parse(readFileSync(new URL(path,root),'utf8'));

test('academic acknowledgements copy complete citations and keep private lookup provenance internal',()=>{
 const registry=read('third_party/source-registry.json');
 const sources=read('frontend/src/generated/sources.json');
 const academic=sources.filter((s:any)=>s.acknowledgement_group==='academic');
 assert.equal(academic.length,61);
 for(const id of ['M19-ZENODO','M19-PAPER','M19-SHAOZI','M19-ATLAS','M19-LANGUAGE-ATLAS'])assert(academic.some((s:any)=>s.id===id),id);
 assert(academic.some((s:any)=>s.id==='REF-M15-R-QUANTILE-TYPE7'));
 for(const id of ['WEBSITE-AOKI-SHORT-WINDOWS','REF-FAST-GRIFFINLIM','REF-NAFNET','REF-BIGVGAN'])assert(academic.some((s:any)=>s.id===id),id);
 for(const id of ['REF-SHUE-VOICE-SOURCE-2010','REF-VOICESAUCE-2011','REF-LING-VOICESAUCE-2019'])assert(academic.some((s:any)=>s.id===id),id);
 assert(!sources.some((s:any)=>s.id==='SRC-M16-VOICEVISTA'));
 assert(registry.sources.find((s:any)=>s.id==='SRC-M16-VOICEVISTA').evidence,'recording source audit retained');
 for(const source of academic){
  const original=registry.sources.find((s:any)=>s.id===source.id);
  assert(source.citation?.trim(),source.id);
  assert.equal(original.citation_format,'GB/T 7714-2015',source.id);
  assert.match(source.citation,/\[(?:J|M|C|D|S|R|Z|DS|EB|DB)(?:\/OL)?\]/,source.id);
  assert.doesNotMatch(source.version,/link only|no implementation|according to existing|本轮|references checked|delivered separately/i,source.id);
 }
 assert(!sources.some((s:any)=>s.id==='M17-CIN-REFERENCE'));
 const cin=registry.sources.find((s:any)=>s.id==='M17-CIN-REFERENCE');
 assert.equal(cin.show_in_acknowledgements,false);
 assert(cin.evidence,'private alias provenance retained');
 const praat=sources.find((s:any)=>s.id==='SRC-PRAAT');
 assert.match(praat.authors,/Paul Boersma/);
 assert.match(praat.authors,/Yannick Jadoul/);
 assert.equal(praat.version,registry.sources.find((s:any)=>s.id==='SRC-PRAAT').acknowledgement_version);
 assert.match(registry.sources.find((s:any)=>s.id==='SRC-PRAAT').actual_included_version,/not probed/,'original audit preserved');
});

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

test('M13 conversion choices retain visible academic citations and edition boundaries',()=>{
 const registry=read('third_party/source-registry.json');
 const sources=read('frontend/src/generated/sources.json');
 const academic=registry.sources.filter((s:any)=>s.modules.includes('M13')&&s.acknowledgement_group==='academic'&&s.show_in_acknowledgements!==false);
 assert.deepEqual(academic.flatMap((s:any)=>s.m13_standards).sort(),[...standards].sort());
 const bib=readFileSync(new URL('third_party/references.bib',root),'utf8');
 for(const source of academic){
  const displayed=sources.find((s:any)=>s.id===source.id);
  assert.equal(displayed.acknowledgement_group,'academic');
  assert.equal(displayed.citation,source.citation);
  assert.equal(displayed.reference_note,source.reference_note);
  assert.equal(source.citation_format,'GB/T 7714-2015');
  assert.match(displayed.citation,/\[(J|M|EB\/OL|S\/OL|DB\/OL)\]/);
  assert(bib.includes('{'+source.id.replaceAll('-','_')+','),source.id);
  assert(source.bibliographic_status&&source.mapping_status,source.id);
 }
 const unt=academic.find((s:any)=>s.id==='REF-M13-UNTPHESOCA-2019');
 assert.deepEqual(unt.m13_standards,['UntPhesoca宽','UntPhesoca严']);
 assert.equal(unt.urls['知乎原文'],'https://zhuanlan.zhihu.com/p/38258415');
 assert.equal(unt.publication_date,'2019-04-22');
 assert.equal(unt.access_date,'2026-10-11');
 assert.match(unt.citation,/\[EB\/OL\]\. \(2019-04-22\)\[2026-10-11\]/);
 assert.match(unt.reference_note,/支持自动或手动组合儿化及逐项拆分/);
 assert.doesNotMatch(unt.reference_note,/版本未锁定|尚不自动组合/);
 assert.match(academic.find((s:any)=>s.id==='REF-M13-WU-1992').reference_note,/尚未确认具体依据/);
});
