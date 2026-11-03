import {createSSRApp,h,computed} from 'vue';
import {renderToString} from 'vue/server-renderer';
import {setLocale,tr} from '../src/i18n/index.ts';
import {defaults,generationDefaults} from '../src/modules/phonation-synthesis/state.ts';
import test from 'node:test';
import assert from 'node:assert/strict';

test('M07 one-group status and permission render complete messages without changing settings',async()=>{
 const analysis=defaults(),generation=generationDefaults(),before=JSON.stringify({analysis,generation});
 const status=computed(()=>tr('modules.phonation-synthesis.taskHistoryCount',{count:1}));
 try{
  for(const language of ['zh-CN','en','ja']as const){
   await setLocale(language);
   const attribution='Author team. 原著者';
   const html=await renderToString(createSSRApp({render:()=>h('section',[
    h('p',status.value),
    h('p',tr('modules.phonation-synthesis.resultSummary',{job:'abcdefgh',batch:'12345678',index:1,total:1})),
    h('p',tr('modules.phonation-synthesis.attributionWithPermission',{attribution,date:'2026-09-10'})),
    h('p',tr('modules.phonation-synthesis.0_value_groups_completed',{v0:1})),
    h('p',tr('modules.phonation-synthesis.value_value_groups_completed',{v0:1,v1:1})),
   ])}));
   assert(!/\{(?:count|attribution|job|batch|index|total|date)\}/.test(html));
   assert(html.includes(attribution)&&html.includes('2026-09-10'));
   assert(!/\b1 groups\b|permission\.\(/.test(html));
   if(language==='en'){
    assert.equal(status.value,'Task history · Groups: 1');assert(html.includes('Group 1 / 1'));
    assert(html.includes('原著者 Permission by author email: 2026-09-10.'));
   }
   assert.equal(JSON.stringify({analysis,generation}),before);
  }
 }finally{await setLocale('zh-CN');}
});
