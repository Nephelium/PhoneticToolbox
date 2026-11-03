import {test} from 'node:test';
import assert from 'node:assert/strict';
import {computed} from 'vue';
import {locale,tr,setLocale,initializeLocale,message,messageRef,languageLoadCounts,receiveDisplayError,bindNativeLocale,localeError} from '../src/i18n/index.ts';
import {accountLocaleStore} from '../src/platform/locale-preferences.ts';
import {RuntimeDisplayError,runtimeError} from '../src/i18n/runtime-errors.ts';
import {uiErrorMessage,uiError,responseError,messageProperty,setMessageProperty} from '../src/modules/module-localization.ts';
import {reactive} from 'vue';
import {parseWav} from '../src/platform/wav.ts';
test('default Chinese, explicit language changes and source-independent IDs',async()=>{
 await initializeLocale({read:(_key,fallback)=>fallback,write:()=>true});assert.equal(locale.value,'zh-CN');assert.equal(tr('common.actions.save'),'保存');
 await setLocale('en');assert.equal(tr('common.actions.save'),'Save');await setLocale('ja');assert.equal(tr('common.actions.save'),'保存');
});
test('templates, computed labels and current message descriptors react without changing data',async()=>{
 const research={id:'M16',sampleRate:48000,frames:90210},before=structuredClone(research),label=computed(()=>tr('common.actions.cancel'));
 const notice=messageRef();notice.value=message('common.test.named',{name:'sample.wav',count:2});
 await setLocale('en');assert.equal(label.value,'Cancel');assert.equal(notice.value,'sample.wav: selected items: 2');
 await setLocale('ja');assert.equal(label.value,'キャンセル');assert.equal(notice.value,'sample.wav：選択項目数 2');assert.deepEqual(research,before);
});
test('language imports are coalesced and cached; latest rapid choice wins',async()=>{
 await Promise.all([setLocale('en'),setLocale('ja'),setLocale('en')]);assert.equal(locale.value,'en');
 await setLocale('ja');await setLocale('en');assert.deepEqual(languageLoadCounts(),{en:1,ja:1});
});
test('named data is literal, not markup or linked message syntax',async()=>{
 await setLocale('en');assert.equal(tr('common.test.literal'),'name@example.com | {a + b}');
 assert.equal(tr('common.test.named',{name:'<script>alert(1)</script>',count:0}),'<script>alert(1)</script>: selected items: 0');
  assert.equal(tr('common.test.named'),'{name}: selected items: {count}');
});
test('unsupported or corrupt saved languages preserve a working Chinese default',async()=>{
 await initializeLocale({read:()=>({unexpected:true}) as any,write:()=>true});assert.equal(locale.value,'zh-CN');
 assert.equal(await setLocale('invalid' as any),false);assert.equal(locale.value,'zh-CN');
});

test('account choices stay with their owner and pending choices cannot write to the next account',async()=>{
 const valuesA:Record<string,unknown>={uiLocale:'en'},valuesB:Record<string,unknown>={};
 const session=(values:Record<string,unknown>)=>({read:()=>({...values}),write:(_module:string,next:Record<string,unknown>)=>{Object.assign(values,next);return true;}}) as any;
 const accountA=accountLocaleStore(session(valuesA)),accountB=accountLocaleStore(session(valuesB));
 await initializeLocale(accountA);assert.equal(locale.value,'en');
 await setLocale('ja');assert.equal(valuesA.uiLocale,'ja');
 const stale=setLocale('en'),changed=initializeLocale(accountB);await Promise.all([stale,changed]);
 assert.equal(locale.value,'zh-CN');assert.equal(valuesA.uiLocale,'ja');assert.equal(valuesB.uiLocale,'zh-CN');
 await setLocale('en');assert.equal(valuesB.uiLocale,'en');assert.equal(valuesA.uiLocale,'ja');
 await initializeLocale(accountA);assert.equal(locale.value,'ja');
});

test('leaving an account restores the anonymous device language',async()=>{
 const original=Object.getOwnPropertyDescriptor(globalThis,'localStorage');
 const data=new Map([['ptb.v3.uiLocale',JSON.stringify('en')]]);
 Object.defineProperty(globalThis,'localStorage',{configurable:true,value:{getItem:(key:string)=>data.get(key)??null,setItem:(key:string,value:string)=>data.set(key,value)}});
 try{
  const account:Record<string,unknown>={uiLocale:'ja'};
  await initializeLocale({read:<T>(key:string,fallback:T)=>(account[key]??fallback) as T,write:(key,value)=>{account[key]=value;return true;}});
  assert.equal(locale.value,'ja');await initializeLocale();assert.equal(locale.value,'en');
  await setLocale('zh-CN');assert.equal(account.uiLocale,'ja');assert.equal(data.get('ptb.v3.uiLocale'),JSON.stringify('zh-CN'));
 }finally{if(original)Object.defineProperty(globalThis,'localStorage',original);else Reflect.deleteProperty(globalThis,'localStorage');}
});

test('real parser errors survive Worker serialization and retranslate without parsing again',async()=>{
 let error:unknown;
 try{parseWav(new ArrayBuffer(50),'original.wav');}catch(value){error=value;}
 assert(error instanceof RuntimeDisplayError);
 const payload=JSON.parse(JSON.stringify({display:error.display,error:error.message}));
 const received=receiveDisplayError(payload.display,payload.error),visible=messageRef(uiErrorMessage(received));
 await setLocale('en');assert.match(visible.value,/RIFF WAV/);assert.match(visible.value,/Choose/);
 await setLocale('ja');assert.match(visible.value,/選び直/);assert(!visible.value.includes('[object Object]'));
 const budget=runtimeError('platform.audio_exceeds_the_preview_memory_budget_value_mb_file',{v0:64});
 assert.match(budget.message,/64 MB/);
 assert.equal(receiveDisplayError({key:'unknown.user.key'},'original diagnostic').message,'original diagnostic');
 await setLocale('zh-CN');assert.match(visible.value,/有效的 RIFF WAV/);
});

test('legacy platform errors resolve at the error boundary; unrelated diagnostics remain literal',async()=>{
 await setLocale('zh-CN');const error=uiError('本机文件不存在，请重新导入。');
 const notice=messageRef(uiErrorMessage(error));await setLocale('en');assert.match(notice.value,/local file does not exist/);
 const raw='Research diagnostic: Sample A {x}';assert.equal(uiError(raw).message,raw);
 await setLocale('ja');assert.match(notice.value,/再度取り込/);assert.equal(uiError(raw).message,raw);
});

test('native response failures retain descriptors while unknown diagnostics and malformed failures stay safe',async()=>{
 await setLocale('zh-CN');const error=responseError('native.cache.qaExitCleanupUnavailable');
 const notice=messageRef(uiErrorMessage(error));
 await setLocale('en');assert.match(notice.value,/isolated cache/);
 await setLocale('ja');assert.match(notice.value,/独立したキャッシュ/);
 const diagnostic='Invalid frame in sample-井井.wav: {V}';assert.equal(responseError(diagnostic).message,diagnostic);
 assert.equal(responseError({message:'untrusted'}).message,tr('common.status.failed'));
 await setLocale('zh-CN');
});

test('short native product failures retranslate after a single failed invocation',async()=>{
 await setLocale('zh-CN');let invocations=0;
 const invoke=()=>{invocations++;return responseError('项目不匹配');};
 const visible=messageRef(uiErrorMessage(invoke()));assert.equal(visible.value,'项目不匹配');
 await setLocale('en');assert.equal(visible.value,'The project does not match.');
 await setLocale('ja');assert.equal(visible.value,'プロジェクトが一致しません。');assert.equal(invocations,1);
 assert.equal(responseError('项目不匹配-original-user-filename.wav').message,'项目不匹配-original-user-filename.wav');
 await setLocale('zh-CN');assert.equal(visible.value,'项目不匹配');
});

test('short registered platform failures are translated only at the error boundary',async()=>{
 await setLocale('zh-CN');const raw='文件后缀无效。',notice=messageRef(uiErrorMessage(responseError(raw)));
 await setLocale('en');assert.equal(notice.value,'Invalid file extension.');assert.equal(raw,'文件后缀无效。');
 await setLocale('ja');assert.notEqual(notice.value,raw);assert.equal(responseError('文件后缀无效。-研究文件.wav').message,'文件后缀无效。-研究文件.wav');
 await setLocale('zh-CN');assert.equal(notice.value,raw);
});

test('native parameterized errors retain literal filenames and can retranslate after the response',async()=>{
 const name='样本-{V}-name@example.com.wav',raw=name+' 需要唯一同名转写。有多种格式时，请选择转写来源后重新读取。';
 await setLocale('zh-CN');const notice=messageRef(uiErrorMessage(responseError(raw,{key:'native.errors.mfa.uniqueTranscript',values:{name}})));assert.equal(notice.value,raw);
 await setLocale('en');assert(notice.value.includes(name));assert.match(notice.value,/exactly one transcript/);
 await setLocale('ja');assert(notice.value.includes(name));assert.match(notice.value,/再読込/);
 assert.equal(responseError('unknown diagnostic',{key:'unknown',values:{name}}).message,'unknown diagnostic');
 assert.equal(responseError(raw,{key:'native.errors.mfa.uniqueTranscript',values:[]}).message,raw);
 await setLocale('zh-CN');
});

test('one message property is shared by the raw object and its Vue proxy',async()=>{
 const raw=messageProperty({error:''},'error'),visible=reactive(raw);
 setMessageProperty(visible,'error',message('common.language.failed'));const getter=Object.getOwnPropertyDescriptor(raw,'error')!.get;
 messageProperty(visible,'error');setMessageProperty(raw,'error',message('common.status.failed'));
 assert.equal(Object.getOwnPropertyDescriptor(raw,'error')!.get,getter);
 await setLocale('en');assert.equal(visible.error,'The operation failed. Please retry.');
 await setLocale('zh-CN');assert.equal(visible.error,'操作失败，请重试。');
});

test('native reconstruction failures and source-audio diagnostics retranslate without repeating work',async()=>{
 await setLocale('zh-CN');let calls=0;
 const request=()=>{calls++;return responseError('输入或笔迹超过预算，请缩小图片、缩短音频或减少笔迹。',{key:'native.errors.reconstruction.inputBudget',values:{}});};
 const visible=messageRef(uiErrorMessage(request()));
 const name='原始音频-{V}-test@example.com.wav';
 const source=messageRef(uiErrorMessage(responseError(name+'：最长支持 30 分钟，请先切分音频。',{key:'native.errors.audio.durationLimit',values:{name}})));
 await setLocale('en');assert.match(visible.value,/processing limit/);assert.match(source.value,/30 minutes/);assert(source.value.includes(name));
 await setLocale('ja');assert.match(visible.value,/処理上限/);assert.match(source.value,/30分/);assert(source.value.includes(name));assert.equal(calls,1);
 await setLocale('zh-CN');assert.match(visible.value,/超过预算/);
});
test('native preference recovers an unavailable browser choice and a failed native save stays visible',async()=>{
 const received:string[]=[];
 bindNativeLocale(async code=>{received.push(code);return false;},'ja');
 try{
  await initializeLocale();assert.equal(locale.value,'ja');assert.equal(localeError.value,'common.language.saveFailed');
  await setLocale('en');assert.deepEqual(received,['ja','en']);assert.equal(locale.value,'en');
 }finally{bindNativeLocale(async()=>true);await initializeLocale({read:(_key,fallback)=>fallback,write:()=>true});}
});
