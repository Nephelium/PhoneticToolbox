import test from 'node:test';
import assert from 'node:assert/strict';
import {setLocale,missingMessages} from '../src/i18n/index.ts';
import {catalog,localizedSymbol,symbolLocalizationBuilds,searchEntries} from '../src/modules/ipa-plus/catalog.ts';
import {contentFor,contentLocalizationBuilds,symbolContent} from '../src/modules/ipa-plus/content.ts';
import {localizedChartSections,chartLocalizationBuilds} from '../src/modules/ipa-plus/chart-localization.ts';
import {defaults as perceptionDefaults,copy} from '../src/modules/perception/model.ts';
import {standards,convertText} from '../src/modules/mandarin-ipa/state.ts';
import {fieldLabels,autoMapping} from '../src/modules/recording/task-import.ts';

test('all IPA descriptions and charts switch language once per source and retain scientific data',async()=>{
 const before=JSON.stringify(catalog),snapshots=new Map<string,ReturnType<typeof contentFor>[]>();
 for(const language of ['zh-CN','en','ja'] as const){
  assert.equal(await setLocale(language),true);
  const localized=catalog.entries.map(entry=>{
   const symbol=localizedSymbol(entry),content=contentFor(entry);
   assert.equal(symbol.id,entry.id);assert.equal(symbol.display,entry.display);assert.equal(symbol.insertText,entry.insertText);
   assert.equal(symbol.codePoints,entry.codePoints);assert.equal(symbol.sourceRefs,entry.sourceRefs);
   for(const field of ['nameZh','descriptionZh','usageZh','contrastZh'] as const)assert.ok(content[field].trim());
   if(language!=='zh-CN')for(const field of ['descriptionZh','usageZh','contrastZh'] as const)assert.notEqual(content[field],symbolContent.entries[entry.id]?.[field]??entry[field]);
   return content;
  });
  snapshots.set(language,localized);
  for(const system of ['ipa','extipa','voqs'] as const){
   const sections=localizedChartSections(system,catalog.charts[system]);
   for(let i=0;i<sections.length;i++){assert.equal(sections[i].ids,catalog.charts[system][i].ids);assert.equal(sections[i].points,catalog.charts[system][i].points);}
   assert.equal(localizedChartSections(system,catalog.charts[system]),sections);
  }
  const builds=[symbolLocalizationBuilds(),contentLocalizationBuilds(),chartLocalizationBuilds()];
  for(let repeat=0;repeat<5;repeat++)catalog.entries.forEach(entry=>{localizedSymbol(entry);contentFor(entry);});
  assert.deepEqual([symbolLocalizationBuilds(),contentLocalizationBuilds(),chartLocalizationBuilds()],builds);
 }
 const builds=[symbolLocalizationBuilds(),contentLocalizationBuilds(),chartLocalizationBuilds()];
 await setLocale('en');assert.equal(contentFor(catalog.entries[0]),snapshots.get('en')![0]);
 for(const system of ['ipa','extipa','voqs'] as const)localizedChartSections(system,catalog.charts[system]);
 assert.deepEqual([symbolLocalizationBuilds(),contentLocalizationBuilds(),chartLocalizationBuilds()],builds);
 assert.equal(JSON.stringify(catalog),before);assert.deepEqual(missingMessages(),[]);
 await setLocale('zh-CN');
});

test('language changes preserve authored experiment text, mapping identities and recording import aliases',async()=>{
 await setLocale('zh-CN');const project=perceptionDefaults(),chineseInstructions=project.config.introText;project.config.introText='User-authored 教示';project.trials=[{id:'fixed-trial',instruction:'Original 刺激提示',stimuli:{}}];
 const saved=copy(project),ids=[...standards],result=convertText('春天','Standard Chinese (Beijing)',{},true),mapping=autoMapping(['任务编号','任务名称','录音内容','文件名','次数','分组','备注','启用']);
 const chineseLabel=fieldLabels.prompt;
 await setLocale('en');assert.deepEqual(project,saved);assert.deepEqual([...standards],ids);assert.deepEqual(convertText('春天','Standard Chinese (Beijing)',{},true),result);
 assert.notEqual(fieldLabels.prompt,chineseLabel);assert.deepEqual(autoMapping(['任务编号','任务名称','录音内容','文件名','次数','分组','备注','启用']),mapping);
 const englishDefaults=perceptionDefaults();assert.notEqual(englishDefaults.config.introText,chineseInstructions);assert.notEqual(englishDefaults.config.title,project.config.title);
 await setLocale('ja');assert.deepEqual(project,saved);assert.ok(searchEntries('ipa','無声').length);
 await setLocale('zh-CN');
});
