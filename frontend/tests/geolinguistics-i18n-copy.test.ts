import test from 'node:test';
import assert from 'node:assert/strict';
import {createSSRApp,h} from 'vue';
import {renderToString} from 'vue/server-renderer';
import {setLocale,tr} from '../src/i18n/index.ts';

test('geolinguistics counts and explanations render as complete sentences in three languages',async()=>{
  for(const language of ['zh-CN','en','ja'] as const){
    await setLocale(language);
    const volume=tr('modules.geolinguistics.phonetics'),level=tr('modules.geolinguistics.broad_categories');
    const html=await renderToString(createSSRApp({render:()=>h('section',[
      h('p',tr('modules.geolinguistics.catalogSummary',{count:1,volume,scope:tr('modules.geolinguistics.readable_legend')})),
      h('p',tr('modules.geolinguistics.selectedSitesCount',{count:0})),
      h('p',tr('modules.geolinguistics.comparisonSummary',{total:205,same:62,different:143,missing:0,shown:143})),
      h('p',tr('modules.geolinguistics.comparisonRecordsExplanation')),
      h('p',tr('modules.geolinguistics.comparisonEditedRecordsExplanation')),
      h('p',tr('modules.geolinguistics.comparisonMeaningLevel',{level})),
      h('p',tr('modules.geolinguistics.comparisonPagination',{page:1,pages:3,pageSize:60})),
    ])}));
    assert(!html.includes('{count}')&&!html.includes('{level}'));
    assert(!/Phoneticsvolume|variants\.Click|theBroad|categorieslevel|\b1 items\b/.test(html));
    if(language==='en'){
      assert(html.includes('Items: 1 · Phonetics volume · Readable legend'));
      assert(html.includes('Selected sites: 0'));
      assert(html.includes('preserving variant order. Click a feature name'));
      assert(html.includes('Meanings are shown using Broad categories; complete original codes'));
      assert.equal(tr('modules.geolinguistics.dialect_sub_subgroup'),'Dialect subdivision');
    }
    assert.notEqual(tr('modules.geolinguistics.dialect_subgroup'),tr('modules.geolinguistics.dialect_sub_subgroup'));
    assert.notEqual(tr('modules.geolinguistics.subgroup'),tr('modules.geolinguistics.sub_subgroup'));
  }
  await setLocale('zh-CN');
});
