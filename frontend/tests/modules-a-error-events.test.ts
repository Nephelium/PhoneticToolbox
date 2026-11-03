import test,{afterEach} from 'node:test';
import assert from 'node:assert/strict';
import {setLocale,messageRef,messageText,tr} from '../src/i18n/index.ts';
import {uiMessage,uiError,uiErrorMessage,messageSnapshot,responseError} from '../src/modules/module-localization.ts';
import {createState,defaults,validateSettingsMessage,applySettings,associateLips,association,draftJson} from '../src/modules/parameter-estimation/state.ts';
import {eggErrorMessage,validateParameters,defaults as eggDefaults,signature} from '../src/modules/egg-analysis/state.ts';
import {jobErrorMessage,parameters,defaults as lpcDefaults} from '../src/modules/lpc-spectrum/state.ts';
import {reconstructionErrorMessage} from '../src/modules/spectrogram-to-audio/error-messages.ts';
import {LipCapture} from '../src/modules/lip-extraction/capture.ts';
import {deviceAccessErrorMessage} from '../src/modules/lip-extraction/devices.ts';
import {checkedOffsetMs} from '../src/modules/lip-extraction/alignment.ts';
import {desktopM05} from '../src/platform/m05.ts';
import {m06Port} from '../src/platform/m06.ts';
import {defaults as synthesisDefaults,exportParams} from '../src/modules/speech-synthesis/state.ts';
import {VocalPlayback} from '../src/platform/vocal.ts';

afterEach(async()=>{await setLocale('zh-CN');});
function caught(work:()=>unknown){try{work();}catch(error){return error;}throw Error('Expected the real validation to reject');}
async function follows(display:ReturnType<typeof uiErrorMessage>){
 const shown=messageRef();shown.value=display;const texts=[];
 for(const code of ['en','ja','zh-CN']as const){await setLocale(code);texts.push(shown.value);assert.equal(shown.value,messageText(display));}
 assert.equal(new Set(texts).size,3);return texts;
}
test('M01 settings and ambiguous association errors switch language without changing drafts or file identities',async()=>{
 const state=createState(),before=draftJson(state),value={...defaults(),min_f0:500,max_f0:100};
 await follows(validateSettingsMessage(value));await follows(uiErrorMessage(caught(()=>applySettings(state,value))));
 const range={...defaults(),smooth_win_size:-1};const [en,ja]=await follows(validateSettingsMessage(range));assert.match(en,/Parameter smoothing window/);assert.doesNotMatch(ja,/Parameter smoothing window/);
 state.files=[{id:'a',name:'被试甲.wav',kind:'audio'},{id:'l1',name:'被试甲.lip.json',kind:'lip'},{id:'l2',name:'被试甲.lip.json',kind:'lip'}]as any;
 const ids=state.files.map(file=>file.id);assert.equal(associateLips(state).ambiguous,1);const link=association(state,'a');
 await setLocale('en');const original=link.lipError;await setLocale('ja');assert.notEqual(link.lipError,original);assert.equal(draftJson(state),before);assert.deepEqual(state.files.map(file=>file.id),ids);
});
test('M03/M04/M09 rejected parameters and task errors retain descriptors, including nested parameter labels',async()=>{
 const egg=eggDefaults(),eggBefore=signature(egg);await follows(eggErrorMessage('egg_filter_failed'));await follows(uiErrorMessage(caught(()=>validateParameters({...egg,spec_window_ms:1000}))));
 await follows(jobErrorMessage('lpc_textgrid_range'));const invalid={...lpcDefaults(),order:''};const [en,ja]=await follows(uiErrorMessage(caught(()=>parameters(invalid))));assert.match(en,/LPC order/);assert.doesNotMatch(ja,/LPC order/);
 await follows(reconstructionErrorMessage('preview_timeout'));assert.equal(signature(egg),eggBefore);
 const literal='Vendor log: 被试甲.wav / E_RAW_17';assert.equal(messageText(reconstructionErrorMessage(new Error(literal))),literal);
});
test('M05 real acquisition rejection, device validation and offset rejection update the same cached notice',async()=>{
 const old=Object.getOwnPropertyDescriptor(globalThis,'navigator');Object.defineProperty(globalThis,'navigator',{configurable:true,value:{storage:{estimate:async()=>null}}});
 try{
  let acquires=0;const video={requestVideoFrameCallback:()=>1,pause:()=>{},srcObject:null}as any;
  const capture=new LipCapture(video,()=>{},async()=>{acquires++;throw uiError(uiMessage('modules.lip-extraction.no_microphone_audio_track_is_available_select_audio_input'));});
  let failure:unknown;try{await capture.start({camera:'',microphone:'',mode:'preview',filter:true,cutoff:15,delegate:'CPU',requestedFps:30});}catch(error){failure=error;}
  await follows(uiErrorMessage(failure));await setLocale('en');const before=capture.state.error;await setLocale('ja');assert.notEqual(capture.state.error,before);assert.equal(acquires,1);assert.equal(capture.state.phase,'failed');assert.equal(capture.frames.length,0);
  await follows(deviceAccessErrorMessage(Object.assign(new Error('raw'),{name:'NotAllowedError'})));await follows(uiErrorMessage(caught(()=>checkedOffsetMs(Infinity))));
 }finally{if(old)Object.defineProperty(globalThis,'navigator',old);else Reflect.deleteProperty(globalThis,'navigator');}
});
test('M05 transport and M06 failed task keep one failure across language changes without re-running',async()=>{
 let calls=0,failure:unknown;try{await desktopM05(async()=>{calls++;throw Error('m05_timeout');},async()=>null);}catch(error){failure=error;}
 await follows(uiErrorMessage(failure));assert.equal(calls,1);
 const config=synthesisDefaults(),before=exportParams(config);let submitted=0;
 const port=m06Port({project:'test',parameters:async()=>({asset_id:'p',sha256:'s'}),source:async()=>({asset_id:'a',sha256:'s'}),create:async()=>{submitted++;return {id:'job',state:'failed',error_code:'m06_reaper_failed'}as any;},job:async()=>{throw Error('Must not poll');},cancel:async()=>{},read:async()=>{throw Error('Must not read a failed result');}});
 try{await port.run('synthesize',config,undefined,new AbortController().signal,()=>{});}catch(error){failure=error;}
 await follows(uiErrorMessage(failure));assert.equal(submitted,1);assert.equal(exportParams(config),before);
});
test('M10 audio pump rejection keeps a live display getter and serializable descriptor without another pull',async()=>{
 let pulls=0;const player=new VocalPlayback(async()=>{pulls++;throw uiError(uiMessage('platform.audio_transfer_failed'));});
 player.context={currentTime:0,state:'running',resume:async()=>{}}as any;await player.play('test');await new Promise(resolve=>setImmediate(resolve));
 assert.equal(player.active,false);assert.equal(pulls,1);assert.ok(player.errorDescriptor);const snapshot=messageSnapshot(player.errorDescriptor!);assert.equal(typeof snapshot,'object');
 await setLocale('en');const before=player.error;await setLocale('ja');assert.notEqual(player.error,before);assert.equal(pulls,1);assert.equal(player.duration,0);
});
test('M19 unavailable-data error retains its special identity and current-language message',async()=>{
 const old=Object.getOwnPropertyDescriptor(globalThis,'document');Object.defineProperty(globalThis,'document',{configurable:true,value:{baseURI:'https://atlas.test/'}});
 try{const {GeoDataUnavailableError}=await import('../src/platform/geolinguistics.ts');const error=new GeoDataUnavailableError();assert.ok(error instanceof GeoDataUnavailableError);assert.equal(error.name,'GeoDataUnavailableError');await follows(uiErrorMessage(error));}
 finally{if(old)Object.defineProperty(globalThis,'document',old);else Reflect.deleteProperty(globalThis,'document');}
});
test('M10 video rejection envelopes retain raw diagnostics and exact limits through three display languages',async()=>{
 for(const [raw,key,values]of[
  ['单次视频导出最多 60 秒，请缩短序列。','native.errors.vocalWeb.videoLimit',{duration:'60'}],
  ['视频时长应为 0.05–60 秒','native.errors.vocalWeb.videoDurationRange',{maximum:'60'}],
 ]as const){
  const envelope={detail:raw,display:{key,values}},before=JSON.stringify(envelope);
  const texts=await follows(uiErrorMessage(responseError(envelope.detail,envelope.display)));
  for(const text of texts)assert.match(text,/60/);assert.equal(JSON.stringify(envelope),before);
 }
 const literal='ExternalCodec: 被试甲.mp4 E_RAW_17';const shown=messageRef();shown.value=uiErrorMessage(responseError(literal,{key:'native.unknown-product-key'}));
 for(const code of ['en','ja','zh-CN']as const){await setLocale(code);assert.equal(shown.value,literal);}
});
