import test from 'node:test';
import assert from 'node:assert/strict';
import {computed} from 'vue';
import {setLocale,tr,messageRef} from '../src/i18n/index.ts';
import {inversePanels} from '../src/modules/egg-analysis/inverse-plots.ts';
import {uiMessage} from '../src/modules/module-localization.ts';
import {defaults,exportParams,presetMessage} from '../src/modules/speech-synthesis/state.ts';

test('EGG translated plot titles retain all numerical projections and never repeat normalization',async()=>{
 const data:any={relative_times_s:[-.025,0,.025],frequencies_hz:[0,1000,2000],audio_values:[-2,0,2],egg_values:[1,3,1],inverse_values:[-1,0,1],audio_db:[1,2,3],egg_db:[2,3,4],inverse_db:[3,4,5]};
 const before=JSON.stringify(data);let builds=0;
 const panels=computed(()=>{builds++;return inversePanels(data,'combined');});
 const title=computed(()=>panels.value[0].title),trace=computed(()=>panels.value[1].traces[0].label);
 const projection=panels.value,normalized=projection[1].traces[0].values;
 try{
  await setLocale('en');assert.equal(title.value,'Audio (Solid) / EGG (Dotted) / IF (Dashed) spectrum');assert.equal(trace.value,'Audio');
  assert.equal(tr('modules.egg-analysis.batchSelectAll',{selected:1,total:1}),'Select all (1 / 1)');
  assert.equal(tr('modules.egg-analysis.saveBatchCount',{count:1}),'Save results for this batch (1)');
  assert.match(tr('modules.egg-analysis.inverseWindowWarning',{order:50,gci:226,crossings:225}),/^This result uses LPC of order 50\. Of 226 GCI start points, 225 have fixed 3 ms windows/);
  await setLocale('ja');assert.match(title.value,/スペクトル$/);assert.equal(panels.value,projection);assert.equal(panels.value[1].traces[0].values,normalized);
  assert.equal(builds,1);assert.equal(JSON.stringify(data),before);assert.deepEqual(normalized,[-1,0,1]);
 }finally{await setLocale('zh-CN');}
});

test('open phonation confirmation follows language while preset identities and synthesis parameters stay fixed',async()=>{
 const config=defaults(),before=exportParams(config),notice=messageRef();
 notice.value=uiMessage('modules.speech-synthesis.confirmApplyPhonation',{preset:presetMessage('气声')});
 try{
  await setLocale('en');assert.match(notice.value,/Apply the Breathy voice source parameters/);
  await setLocale('ja');assert.match(notice.value,/気息性発声/);assert.equal(exportParams(config),before);
 }finally{await setLocale('zh-CN');}
});
