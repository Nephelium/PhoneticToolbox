import test from 'node:test';
import assert from 'node:assert/strict';
import {installWebFeedback} from '../src/platform/feedback-web.ts';
import {feedbackCall} from '../src/platform/feedback.ts';
test('web feedback uses current account Cookie and CSRF without collecting email or storing credentials',async()=>{
 const original=globalThis.fetch,calls:{owner:string|null;body:any}[]=[];
 globalThis.fetch=async(url,init={})=>{assert.equal(String(url),'/api/v1/web-feedback');assert.equal(init.credentials,'same-origin');assert.equal(init.redirect,'error');const headers=new Headers(init.headers);assert.equal(headers.get('X-CSRF-Token'),'csrf');assert.equal(headers.get('Authorization'),null);calls.push({owner:headers.get('X-PTB-Account'),body:JSON.parse(init.body as string)});return Response.json({connected:true,email:'owned@example.test',verified:true,owner:'feedback-owner',platform:'web'});};
 let close=installWebFeedback('a',()=>'csrf');
 try{const result=await feedbackCall<any>('status');assert(result.connected&&result.verified);assert.deepEqual(calls[0],{owner:'a',body:{operation:'status',args:{}}});await assert.rejects(feedbackCall('verifyStart',{email:'other@example.test'}),/不可用/);close();close=installWebFeedback('b',()=>'csrf');await feedbackCall('status');assert.equal(calls.at(-1)?.owner,'b');}finally{close();globalThis.fetch=original;}
});
test('expired account cannot fall back to a separate browser feedback identity',async()=>{
 const original=globalThis.fetch;globalThis.fetch=async()=>Response.json({detail:'authentication_required'},{status:401});const close=installWebFeedback('a',()=>'csrf');
 try{await assert.rejects(feedbackCall('status'),/登录已过期/);}finally{close();globalThis.fetch=original;}
});
