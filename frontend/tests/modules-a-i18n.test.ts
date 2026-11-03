import {test,afterEach} from 'node:test';
import assert from 'node:assert/strict';
import {computed,watch} from 'vue';
import {setLocale,tr,messageRef} from '../src/i18n/index.ts';
import {localized,uiMessage,joinMessages,uiError,uiErrorMessage,setMessageProperty} from '../src/modules/module-localization.ts';
import {defaults as acousticDefaults,createState,draftJson,settingLabels} from '../src/modules/parameter-estimation/state.ts';
import {defaults as synthesisDefaults,exportParams,preset,presetLabel,f0Methods} from '../src/modules/speech-synthesis/state.ts';
import {defaults as phonationDefaults,generationDefaults,designs,kinds} from '../src/modules/phonation-synthesis/state.ts';
import {defaults as pitchDefaults,modes} from '../src/modules/pitch-manipulation/state.ts';
import {eggJobTitle,isEggPreviewJob,type EggJobLabel} from '../src/modules/egg-analysis/job-label.ts';
import {LipCapture} from '../src/modules/lip-extraction/capture.ts';
import {emptyInputLevel} from '../src/modules/lip-extraction/audio-input.ts';
afterEach(async()=>{await setLocale('zh-CN');});

test('module display collections track language, preserve iteration, and cache repeated reads',async()=>{
 await setLocale('zh-CN');let builds=0;
 const labels=localized(()=>{builds++;return [{id:'f0',label:tr('modules.parameter-estimation.minimum_fundamental_frequency')}];});
 const selected=computed(()=>labels.map(row=>row.label).join(','));
 assert.ok(Array.isArray(labels));assert.equal(selected.value,'最小基频');
 const first=labels[0];assert.deepEqual(Object.keys(labels),['0']);assert.equal(Object.entries(labels)[0][1],first);
 assert.equal(JSON.parse(JSON.stringify(labels))[0].id,'f0');
 for(let i=0;i<100;i++){assert.equal(labels[0],first);assert.equal(labels.length,1);assert.equal(selected.value,'最小基频');}
 assert.equal(builds,1);
 await setLocale('en');assert.equal(selected.value,'Minimum fundamental frequency');assert.equal(builds,2);
 await setLocale('en');assert.equal(labels[0].label,'Minimum fundamental frequency');assert.equal(builds,2);
 await setLocale('ja');assert.equal(selected.value,'基本周波数の下限');assert.equal(builds,3);
});

test('structured composed notices change language while captured names and numbers stay fixed',async()=>{
 const notice=messageRef();
 await setLocale('zh-CN');notice.value=joinMessages(uiMessage('modules.lpc-spectrum.sent_to_browser_for_download'),'被试甲.wav');
 assert.equal(notice.value,'已交给浏览器下载：被试甲.wav');
 await setLocale('en');assert.equal(notice.value,'Sent to browser for download: 被试甲.wav');
 notice.value=uiMessage('modules.pitch-manipulation.saved_value_audio_files_value',{v0:3,v1:uiMessage('modules.pitch-manipulation.value_not_saved',{v0:2})});
 assert.equal(notice.value,'Saved audio files: 3; 2 not saved');
 await setLocale('ja');assert.equal(notice.value,'3件の音声を保存しました、2件未保存');
});

test('known validation errors retain descriptors while original external diagnostics remain unchanged',async()=>{
 await setLocale('zh-CN');const error=uiError(uiMessage('modules.speech-synthesis.invalid_f0_range'));
 assert.ok(error instanceof Error);assert.equal(error.message,'F0 范围无效');
 const notice=messageRef();notice.value=uiErrorMessage(error);
 await setLocale('en');assert.equal(notice.value,'Invalid F0 range');
 await setLocale('ja');assert.equal(notice.value,'F0範囲が無効です');
 assert.equal(uiErrorMessage(new Error('REAPER stderr: exit 2')),'REAPER stderr: exit 2');
});

test('EGG task visibility uses stable modes and new titles change language without rewriting historical names',async()=>{
 const preview:EggJobLabel={file:'被试.wav',mode:'preview'},batch:EggJobLabel={file:'交互分析.wav',mode:'batch',retries:1};
 const snapshot=JSON.stringify([preview,batch]);
 for(const language of ['en','ja','zh-CN'] as const){
  await setLocale(language);assert.equal(isEggPreviewJob(preview,'any old title'),true);assert.equal(isEggPreviewJob(batch,'交互分析.wav · 批次'),false);
  assert.equal(eggJobTitle(undefined,'既有任务名称'),'既有任务名称');assert.equal(eggJobTitle({mode:'unknown'} as any,'旧值'),'旧值');
  assert.ok(eggJobTitle(batch,'').startsWith('交互分析.wav · '));assert.equal(JSON.stringify([preview,batch]),snapshot);
 }
 await setLocale('en');assert.equal(eggJobTitle(preview,''),'被试.wav · Interactive analysis');
 await setLocale('ja');assert.equal(eggJobTitle(preview,''),'被试.wav · 対話的解析');
});

test('visible capture errors and input-monitor warnings change language without reopening capture or altering samples',async()=>{
 let acquisitions=0,publications=0;const capture=new LipCapture({} as HTMLVideoElement,()=>publications++,async()=>{acquisitions++;throw Error('unexpected acquisition');});
 const level=emptyInputLevel(),statistics=JSON.stringify(capture.state.stats),state=capture.state;
 setMessageProperty(state,'error',uiMessage('modules.lip-extraction.device_disconnected_finalizing_captured_data'));
 setMessageProperty(level,'monitorError',uiMessage('modules.lip-extraction.input_level_monitoring_paused_stop_and_listen_to_the'));
 const visible=computed(()=>({...state,error:state.error,warning:level.monitorError}));
 await setLocale('en');assert.equal(visible.value.error,'Device disconnected; finalizing captured data.');assert.match(visible.value.warning,/Input-level monitoring paused/);
 await setLocale('ja');assert.equal(visible.value.error,'デバイスが切断されました。収録済みデータを確定しています。');assert.match(visible.value.warning,/入力レベル監視を一時停止/);
 assert.equal(capture.state,state);assert.equal(JSON.stringify(state.stats),statistics);assert.equal(acquisitions,0);assert.equal(publications,0);
 assert.equal(typeof state.error,'string');assert.equal(typeof level.monitorError,'string');state.error='Raw device diagnostic';await setLocale('zh-CN');assert.equal(state.error,'Raw device diagnostic');
});

test('switching module labels never changes scientific drafts, preset IDs, or parameter exports',async()=>{
 await setLocale('zh-CN');const acoustic=createState(),speech=synthesisDefaults();preset(speech,'假声');
 const snapshot={acoustic:draftJson(acoustic),settings:JSON.stringify(acousticDefaults()),speech:exportParams(speech),phonation:JSON.stringify([phonationDefaults(),generationDefaults(),designs()]),pitch:JSON.stringify(pitchDefaults())};
 let scientificWrites=0;const stop=watch(()=>speech,()=>scientificWrites++,{deep:true,flush:'sync'});
 for(const language of ['en','ja','zh-CN'] as const){
  await setLocale(language);
  assert.equal(draftJson(acoustic),snapshot.acoustic);assert.equal(JSON.stringify(acousticDefaults()),snapshot.settings);
  assert.equal(exportParams(speech),snapshot.speech);assert.equal(speech.f0_transform.preset,'假声');
  assert.equal(JSON.stringify([phonationDefaults(),generationDefaults(),designs()]),snapshot.phonation);assert.equal(JSON.stringify(pitchDefaults()),snapshot.pitch);
  assert.equal(modes[0].value,'full');assert.equal(Object.keys(f0Methods).join(','),'praat_cc,praat_ac,reaper');
  assert.equal(typeof settingLabels.min_f0[0],'string');assert.equal(typeof kinds[1],'string');
 }
 stop();assert.equal(scientificWrites,0);
 await setLocale('en');assert.equal(presetLabel('假声'),'Falsetto');assert.equal(presetLabel('耳语'),'Whisper');
 await setLocale('ja');assert.equal(presetLabel('气声'),'気息性発声');assert.equal(presetLabel('自定义名称'),'自定义名称');
});
