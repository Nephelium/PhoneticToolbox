import test from 'node:test';
import assert from 'node:assert/strict';
import {setLocale,messageRef,errorMessageDescriptor,formatErrorCode,message} from '../src/i18n/index.ts';

test('recording recovery and processing diagnostics relocalize without rewriting saved raw records',async()=>{
 const raw={error:'异常中断恢复，仅包含已落盘且哈希通过的前缀，末尾可能缺失',warning:'噪声段统计变化较大，可能含语音或非定常噪声，请试听确认'};
 const snapshot=JSON.stringify(raw),notice=messageRef(''),saved=messageRef('');
 await setLocale('zh-CN');
 notice.value=errorMessageDescriptor(raw.warning)??raw.warning;
 saved.value=message('modules.recording.text.cd4337431579');
 const originalSaved=saved.value;
 assert.equal(notice.value,raw.warning);
 for(const code of ['en','ja'] as const){
  await setLocale(code);
  assert.notEqual(notice.value,raw.warning);
  assert.notEqual(saved.value,originalSaved);
  assert.notEqual(formatErrorCode(raw.error),raw.error);
  assert.equal(formatErrorCode('unknown diagnostic / private.wav'),'unknown diagnostic / private.wav');
  assert.equal(JSON.stringify(raw),snapshot);
 }
 await setLocale('zh-CN');
 assert.equal(notice.value,raw.warning);
 assert.equal(saved.value,originalSaved);
});

test('five device failure contexts keep descriptors and literal driver diagnostics across three locales',async()=>{
 const {recordingErrorMessage}=await import('../src/platform/recording.ts');
 const {recordingRequests}=await import('../src/platform/recording-requests.ts');
 const keys=['deviceUnavailable','playbackStartCleanup','playbackStreamStatus','playbackAbortFailed','playbackCloseFailed'];
 const diagnostic='driver-{reason} / device原名',notices=keys.map(key=>{
  const value=messageRef('');value.value=recordingErrorMessage('raw '+diagnostic,{key:'native.errors.recording.'+key,values:{reason:diagnostic}});return value;
 });
 let reply:(id:string,raw:string)=>void=()=>{};
 const request=recordingRequests({recordingReady:{connect(fn){reply=fn;}},recording(id){reply(id,JSON.stringify({ok:false,error:'raw driver',display:{key:'native.errors.recording.playbackStartCleanup',values:{reason:'操作已取消。'}}}));}});
 let caught:unknown;try{await request({op:'play'});}catch(error){caught=error;}
 const status=messageRef('');status.value=recordingErrorMessage(caught);
 try{
  await setLocale('zh-CN');const before=notices.map(value=>value.value);assert(status.value.includes('操作已取消'));
  for(const code of ['en','ja'] as const){await setLocale(code);for(let i=0;i<notices.length;i++){assert.notEqual(notices[i].value,before[i]);assert(notices[i].value.endsWith(diagnostic));}assert(!status.value.includes('操作已取消'));}
 }finally{await setLocale('zh-CN');}
});
