import test from 'node:test';
import assert from 'node:assert/strict';
import sources from '../src/generated/sources.json' with {type:'json'};
import notes from '../src/i18n/locales/zh-CN/source-notes.json' with {type:'json'};
import {sourceText,type SourceDisplayField} from '../src/components/source-display.ts';
import {setLocale,tr} from '../src/i18n/index.ts';

test('source explanations change language while original registry and citations remain intact',async()=>{
 const before=JSON.stringify(sources),fields:SourceDisplayField[]=['reference_note','license','attribution','adaptation_note','adaptation_caution'];
 let count=0;
 for(const source of sources)for(const field of fields){
  const raw=source[field as keyof typeof source];if(typeof raw==='string'&&/[\u4e00-\u9fff]/.test(raw)){
   const key='sources.metadata.'+source.id+'.'+field;assert.equal((notes as Record<string,string>)[key],raw);count++;
  }
 }
 assert.equal(count,Object.keys(notes).length);
 const mediapipe=sources.find(row=>row.id==='SRC-MEDIAPIPE')!,zaiwa=sources.find(row=>row.id==='SRC-ZAIWA')!,quantile=sources.find(row=>row.id==='REF-M15-R-QUANTILE-TYPE7')!;
 await setLocale('en');assert.equal(sourceText(mediapipe,'license'),'Apache-2.0 (code)');assert.match(sourceText(quantile,'reference_note'),/type 7/);assert.match(sourceText(zaiwa,'adaptation_note'),/LPC/);
 assert.equal(tr('components.sourceUsedBy',{modules:'Lip tracking'}),'Used by: Lip tracking');assert.equal(tr('components.sourceVersion',{version:'literal-{V}'}),'Version: literal-{V}');
 await setLocale('ja');assert.equal(sourceText(mediapipe,'license'),'Apache-2.0（コード）');assert.match(sourceText(quantile,'reference_note'),/四分位数/);assert.match(sourceText(zaiwa,'attribution'),/陆尧/);
 assert.equal(sourceText({id:'unknown',license:'研究者原文 {V}'},'license'),'研究者原文 {V}');assert.equal(sourceText({...mediapipe,license:'changed original'},'license'),'changed original');
 assert.equal(JSON.stringify(sources),before);
 await setLocale('zh-CN');assert.equal(sourceText(mediapipe,'license'),mediapipe.license);
});
