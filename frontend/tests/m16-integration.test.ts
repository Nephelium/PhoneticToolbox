import {test} from 'node:test';
import assert from 'node:assert/strict';
import {claimCapture,releaseCapture,captureOwner,assertPlaybackAllowed} from '../src/platform/capture-lease.ts';
import {recordingRequests,type RecordingChannel} from '../src/platform/recording-requests.ts';

test('capture conflict cannot release another module and failed acquisition preserves owner',()=>{
  claimCapture('M05');assert.throws(()=>claimCapture('M16'),/唇形采集/);assert.doesNotThrow(assertPlaybackAllowed);
  releaseCapture('M16');assert.equal(captureOwner(),'M05');releaseCapture('M05');
  claimCapture('M16');claimCapture('M16');assert.throws(()=>claimCapture('M05'),/录音模块/);assert.throws(assertPlaybackAllowed,/录音或录前检测/);
  releaseCapture('M16');assert.equal(captureOwner(),undefined);assert.doesNotThrow(assertPlaybackAllowed);
});
test('native recording channel serializes own requests and recovers after error without task queue',async()=>{
  let reply!:(id:string,raw:string)=>void;
  const sent:{id:string;body:any}[]=[];
  const channel:RecordingChannel={recording:(id,body)=>sent.push({id,body:JSON.parse(body)}),recordingReady:{connect:fn=>{reply=fn;}}};
  const send=recordingRequests(channel,1000);
  const first=send({op:'status'}),second=send<{stopped:boolean}>({op:'stop'});
  await Promise.resolve();assert.equal(sent.length,1);
  reply(sent[0].id,JSON.stringify({ok:false,error:'device lost'}));
  await assert.rejects(first,/device lost/);await Promise.resolve();
  assert.equal(sent.length,2);assert.equal(sent[1].body.op,'stop');
  reply(sent[1].id,JSON.stringify({ok:true,value:{stopped:true}}));
  assert.deepEqual(await second,{stopped:true});
});
test('late reply after recording request timeout cannot resolve a subsequent stop',async()=>{
  let reply!:(id:string,raw:string)=>void;const ids:string[]=[];
  const send=recordingRequests({recording:id=>{ids.push(id);},recordingReady:{connect:fn=>{reply=fn;}}},20);
  await assert.rejects(send({op:'status'}),/超时/);
  const stop=send({op:'stop'});await Promise.resolve();
  reply(ids[0],JSON.stringify({ok:true,value:'old'}));
  reply(ids[1],JSON.stringify({ok:true,value:'stopped'}));
  assert.equal(await stop,'stopped');
});
