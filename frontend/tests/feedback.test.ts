import test from 'node:test';
import assert from 'node:assert/strict';
import {installFeedbackChannel,feedbackCall,onFeedbackNotification} from '../src/platform/feedback.ts';

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
