import test from 'node:test';
import assert from 'node:assert/strict';
import {computed} from 'vue';
import {setLocale,messageRef} from '../src/i18n/index.ts';
import {cacheCandidateLabel,cacheCandidateReason,cacheCleanupMessage,type CacheCandidate} from '../src/platform/storage.ts';

test('existing cache rows and cleanup messages update without rescanning or changing identities',async()=>{
 const item:CacheCandidate={id:'stable',kind:'runtime',path:'D:/owned-qa/former-user-data/startup-cache',label:'旧运行缓存',reason:'仍有程序使用此处的缓存，请关闭相关窗口后重新检查。',reason_code:'in_use',bytes:42,size_complete:true,can_clean:false};
 const original=JSON.stringify(item),label=computed(()=>cacheCandidateLabel(item)),reason=computed(()=>cacheCandidateReason(item));
 const notice=messageRef(cacheCleanupMessage({complete:true,message:'该处旧缓存已清理。',message_code:'complete'}));
 const labels=new Set<string>(),reasons=new Set<string>(),notices=new Set<string>();
 for(const code of ['zh-CN','en','ja'] as const){await setLocale(code);labels.add(label.value);reasons.add(reason.value);notices.add(notice.value);assert.equal(JSON.stringify(item),original);assert.ok(!label.value.startsWith('native.'));}
 assert.equal(labels.size,3);assert.equal(reasons.size,3);assert.equal(notices.size,3);
 const unknown={...item,kind:'future',label:'Custom cache',reason_code:'future',reason:'Diagnostic'};
 assert.equal(cacheCandidateLabel(unknown),'Custom cache');assert.equal(cacheCandidateReason(unknown),'Diagnostic');
 await setLocale('zh-CN');
});
