import test from 'node:test';
import assert from 'node:assert/strict';
import {installUpdatesChannel,updater,updatesAvailable,checkMessage,type UpdatesChannel,type DownloadProgress} from '../src/platform/updates.ts';
class Signal{
  callbacks=new Set<(id:string,payload:string)=>void>();
  connect(callback:(id:string,payload:string)=>void){this.callbacks.add(callback);}
  disconnect(callback:(id:string,payload:string)=>void){this.callbacks.delete(callback);}
  emit(id:string,value:unknown){for(const callback of this.callbacks)callback(id,typeof value==='string'?value:JSON.stringify(value));}
}
class Channel implements UpdatesChannel{
  ready=new Signal();progress=new Signal();requests:{id:string;value:{operation:string;args:Record<string,unknown>}}[]=[];cancelled:string[]=[];
  request(id:string,payload:string){this.requests.push({id,value:JSON.parse(payload)});}
  cancel(id:string){this.cancelled.push(id);}
}
test('browser has no fetch fallback or updater capability',async()=>{
  installUpdatesChannel(undefined);assert.equal(updatesAvailable(),false);
  await assert.rejects(updater.check({manual:true}),/桌面版/);
});
test('native requests preserve two-source unknown status and exact requested preferences',async()=>{
  const channel=new Channel();installUpdatesChannel(channel);
  try{const request=updater.check({manual:true,source:'server',channel:'preview'});const sent=channel.requests.at(-1)!;
    assert.deepEqual(sent.value,{operation:'check',args:{manual:true,source:'server',channel:'preview'}});
    channel.ready.emit(sent.id,{ok:true,value:{status:'incomplete',currentVersion:'3.0.0-preview.1',candidate:null,shouldPrompt:false,sources:{server:{status:'error'},github:{status:'no-releases'}}}});
    const result=await request;assert.equal(result.status,'incomplete');assert.match(checkMessage(result),/未完整/);
  }finally{installUpdatesChannel(undefined);}
});
test('download sends opaque release only, receives verified identifiers and bounded progress',async()=>{
  const channel=new Channel();installUpdatesChannel(channel);const progress:DownloadProgress[]=[];
  try{const request=updater.download('release_1','portable',true,{progress:value=>progress.push(value)});const sent=channel.requests.at(-1)!;
    assert.deepEqual(sent.value.args,{releaseId:'release_1',packageKind:'portable',confirmed:true});
    channel.progress.emit(sent.id,{phase:'downloading',source:'server',received:5,total:10});
    channel.progress.emit(sent.id,{phase:'downloading',received:20,total:10});
    channel.progress.emit(sent.id,'broken');
    assert.equal(progress.length,1);
    channel.ready.emit(sent.id,{ok:true,value:{downloadId:'download_1',verified:true,name:'portable.zip',size:10,sha256:'hash',kind:'portable',source:'server',applyAvailable:false}});
    assert.equal((await request).verified,true);
  }finally{installUpdatesChannel(undefined);}
});
test('abort cancels only owned request, late completion never replaces cancellation',async()=>{
  const channel=new Channel();installUpdatesChannel(channel);const abort=new AbortController();
  try{const request=updater.check({},abort.signal);const sent=channel.requests.at(-1)!;abort.abort();
    channel.ready.emit(sent.id,{ok:true,value:{status:'up-to-date'}});
    await assert.rejects(request,error=>(error as {code:string}).code==='CANCELLED');assert.deepEqual(channel.cancelled,[sent.id]);
  }finally{installUpdatesChannel(undefined);}
});
test('disconnect rejects pending work and disconnects native handlers',async()=>{
  const channel=new Channel();installUpdatesChannel(channel);
  const request=updater.preferences();installUpdatesChannel(undefined);
  await assert.rejects(request,error=>(error as {code:string}).code==='DISCONNECTED');
  assert.equal(channel.ready.callbacks.size,0);assert.equal(channel.progress.callbacks.size,0);
});
test('native errors and invalid response are clear and do not resolve success',async()=>{
  const channel=new Channel();installUpdatesChannel(channel);
  try{const a=updater.download('selected','portable',false);channel.ready.emit(channel.requests.at(-1)!.id,{ok:false,error:{code:'CONFIRM_REQUIRED',message:'请确认后下载。'}});await assert.rejects(a,/确认/);
    const b=updater.preferences();channel.ready.emit(channel.requests.at(-1)!.id,'invalid');await assert.rejects(b,/无法解析/);
  }finally{installUpdatesChannel(undefined);}
});
