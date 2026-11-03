import {test,afterEach} from 'node:test';
import assert from 'node:assert/strict';
import {setLocale} from '../src/i18n/index.ts';
import {emptyProject,classPath,categoryColor,categoryCounts,observationText,volumeNames,type AtlasFeature} from '../src/modules/geolinguistics/model.ts';
import {displayCodes,displayText,displayValue,displayCounts} from '../src/modules/geolinguistics/groups.ts';
afterEach(async()=>{await setLocale('zh-CN');});

test('map UI translations preserve raw classifications, missing-value sentinels and output descriptions',async()=>{
 const project=emptyProject(),feature:AtlasFeature={id:'F1_P',volume:'P',number:1,page:2,title:'用户保留的原书题名',titleReviewed:true,categories:['A1'],labels:{A1:'原始释义'},legend:'原图图例'};
 const path=classPath(['','','','','','','甲区','', '乙片']),color=categoryColor(path),sourceValue=displayValue('A9',feature,project,'fine',undefined,true),snapshot=JSON.stringify(project);
 for(const language of ['en','ja','zh-CN'] as const){
  await setLocale(language);
  assert.equal(classPath(['','','','','','','甲区','','乙片']),path);assert.equal(categoryColor(path),color);
  assert.deepEqual(displayCodes(null,feature,'fine'),['待核']);assert.deepEqual(categoryCounts([null]),[{code:'待核',count:1}]);
  assert.deepEqual(displayCounts([{value:null}],feature,'fine'),[{code:'待核',count:1}]);
  assert.equal(displayValue('A9',feature,project,'fine',undefined,true),sourceValue);assert.equal(sourceValue,'A9 · 释义待核');
  assert.equal(observationText(null,feature,project,false,true),'原值待核');
  assert.equal(displayValue('A1',feature,project,'fine'),'原始释义');assert.equal(feature.title,'用户保留的原书题名');assert.equal(JSON.stringify(project),snapshot);
  assert.equal(volumeNames.P,language==='en'?'Phonetics':language==='ja'?'音声':'语音');
  assert.equal(displayText('待核',feature,project,'fine'),language==='en'?'Original value pending verification':language==='ja'?'元の値は確認待ち':'原值待核');
 }
});
