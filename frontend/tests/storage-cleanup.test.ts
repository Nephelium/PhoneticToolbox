import test, {type TestContext} from 'node:test';
import assert from 'node:assert/strict';
import {setImmediate} from 'node:timers/promises';
import {initializePlatform} from '../src/platform/desktop.ts';
import {localStoragePort} from '../src/platform/storage.ts';

async function connect(t:TestContext){
  let ready:(id:string,raw:string)=>void=()=>{};
  const requests:{id:string;op:string}[]=[];
  const bridge={
    invoke(_raw:string,callback:(raw:string)=>void){callback(JSON.stringify({ok:true,value:{kind:'desktop',session:'test',api_version:'1.1.0',tasks:true}}));},
    previewReady:{connect(){}},
    taskReady:{connect(callback:typeof ready){ready=callback;}},
    task(id:string,raw:string){const {op}=JSON.parse(raw);if(op==='m05_catalog')ready(id,JSON.stringify({ok:true,value:{available:false}}));else requests.push({id,op});},
  };
  const previous=Object.getOwnPropertyDescriptor(globalThis,'window');
  Object.defineProperty(globalThis,'window',{configurable:true,value:{qt:{webChannelTransport:{}},QWebChannel:class{constructor(_transport:unknown,callback:(value:unknown)=>void){callback({objects:{files:bridge}});}}}});
  t.after(()=>{if(previous)Object.defineProperty(globalThis,'window',previous);else Reflect.deleteProperty(globalThis,'window');});
  await initializePlatform();
  return {requests,reply:(id:string,value:unknown)=>ready(id,JSON.stringify({ok:true,value})),fail:(id:string)=>ready(id,JSON.stringify({ok:false,error:'清理服务已断开。'}))};
}

test('cleanup waits beyond five minutes and serializes refresh until native completion',async t=>{
  t.mock.timers.enable({apis:['setTimeout']});
  const channel=await connect(t),port=localStoragePort();
  const cleanup=port.clean(),refresh=port.status();
  await setImmediate();
  assert.deepEqual(channel.requests.map(r=>r.op),['local_storage_cleanup']);
  t.mock.timers.tick(24*60*60*1000);
  await setImmediate();
  assert.equal(channel.requests.length,1);
  channel.reply(channel.requests[0].id,{skipped:false,count:12,bytes:1024,complete:true});
  assert.equal((await cleanup).count,12);
  await setImmediate();
  assert.equal(channel.requests[1].op,'local_storage_status');
  channel.reply(channel.requests[1].id,{result_count:72});
  assert.equal((await refresh).result_count,72);
});

test('cleanup still reports native errors after a long wait and releases the queue',async t=>{
  t.mock.timers.enable({apis:['setTimeout']});
  const channel=await connect(t),port=localStoragePort();
  const cleanup=port.clean(),rejected=assert.rejects(cleanup,/清理服务已断开/),refresh=port.status();
  await setImmediate();
  t.mock.timers.tick(24*60*60*1000);
  await setImmediate();
  channel.fail(channel.requests[0].id);
  await rejected;
  await setImmediate();
  assert.equal(channel.requests[1].op,'local_storage_status');
  channel.reply(channel.requests[1].id,{result_count:72});
  assert.equal((await refresh).result_count,72);
});

test('storage status keeps its existing five minute renderer timeout',async t=>{
  t.mock.timers.enable({apis:['setTimeout']});
  await connect(t);
  const rejected=assert.rejects(localStoragePort().status(),/任务操作超时/);
  await setImmediate();
  t.mock.timers.tick(300001);
  await rejected;
});
