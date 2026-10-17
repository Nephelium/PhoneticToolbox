import test from 'node:test';
import assert from 'node:assert/strict';
import {installFeedbackChannel,feedbackCall,onFeedbackNotification,feedbackClientLabel,supportsFeedbackTrash} from '../src/platform/feedback.ts';

test('recycle-bin controls require explicit server support and a 30-day policy',()=>{
  assert.equal(supportsFeedbackTrash({}),false);
  assert.equal(supportsFeedbackTrash({conversationTrash:false,trashRetentionDays:30}),false);
  assert.equal(supportsFeedbackTrash({conversationTrash:true}),false);
  assert.equal(supportsFeedbackTrash({conversationTrash:true,trashRetentionDays:7}),false);
  assert.equal(supportsFeedbackTrash({conversationTrash:true,trashRetentionDays:30}),true);
});

test('feedback provenance labels preserve unknown historical fields and distinguish packages',()=>{
  assert.equal(feedbackClientLabel(), '来源未记录');
  assert.equal(feedbackClientLabel({source:'web',version:'3.0.2'}), '网页版');
  assert.equal(feedbackClientLabel({source:'desktop',version:'3.0.1',packageKind:'installer'}), '3.0.1 · 安装版');
  assert.equal(feedbackClientLabel({source:'desktop',version:'3.0.2',packageKind:'portable'}), '3.0.2 · 免安装版');
  assert.equal(feedbackClientLabel({source:'desktop',version:'3.0.2',packageKind:'source'}), '3.0.2 · 源码运行');
  assert.equal(feedbackClientLabel({source:'desktop'}), '版本未记录 · 安装形式未记录');
});

test('late replies cannot complete new requests and disconnect rejects pending work',async()=>{
  let listener:(id:string,payload:string)=>void=()=>{};
  const sent:{id:string;payload:string}[]=[];
  installFeedbackChannel({request:(id,payload)=>sent.push({id,payload}),ready:{connect:fn=>listener=fn}});
  const first=feedbackCall('status');
  listener(sent[0]!.id,JSON.stringify({ok:true,value:{verified:false}}));
  assert.deepEqual(await first,{verified:false});
  const next=feedbackCall('messages',{after:0});
  listener(sent[0]!.id,JSON.stringify({ok:true,value:'late'}));
  installFeedbackChannel(undefined);
  await assert.rejects(next,/断开/);
});

test('notifications are separate from request completion and malformed payloads are ignored',()=>{
  let notification:(payload:string)=>void=()=>{};
  const values:unknown[]=[];
  installFeedbackChannel({request:()=>{},ready:{connect:()=>{}},notification:{connect:fn=>notification=fn}});
  const stop=onFeedbackNotification(value=>values.push(value));
  notification('{bad');
  notification(JSON.stringify({kind:'daily_feedback',count:2}));
  assert.equal(values.length,1);
  stop();
  notification(JSON.stringify({kind:'daily_feedback',count:3}));
  assert.equal(values.length,1);
  installFeedbackChannel(undefined);
});
