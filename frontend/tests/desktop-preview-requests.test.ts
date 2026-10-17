import test from 'node:test';
import assert from 'node:assert/strict';
import {setImmediate} from 'node:timers/promises';
import {initializePlatform,desktopFiles} from '../src/platform/desktop.ts';
import type {ResearchFile} from '../src/platform/research.ts';

test('real desktop adapter sends one native preview/read for concurrent identical callers',async t=>{
  let previewReady!:(id:string,raw:string)=>void,taskReady!:(id:string,raw:string)=>void;
  const previews:Array<{id:string;body:unknown}>=[],tasks:Array<{id:string;body:{op:string}}>=[];
  const bridge={
    invoke(_raw:string,done:(value:string)=>void){done(JSON.stringify({ok:true,value:{kind:'desktop',session:'preview-test',api_version:'1.1.0',tasks:true}}));},
    previewReady:{connect(callback:typeof previewReady){previewReady=callback;}},
    taskReady:{connect(callback:typeof taskReady){taskReady=callback;}},
    preview(id:string,raw:string){previews.push({id,body:JSON.parse(raw)});},
    task(id:string,raw:string){const body=JSON.parse(raw);if(body.op==='m05_catalog')taskReady(id,JSON.stringify({ok:true,value:{available:false}}));else tasks.push({id,body});},
  };
  const previous=Object.getOwnPropertyDescriptor(globalThis,'window');
  Object.defineProperty(globalThis,'window',{configurable:true,value:{qt:{webChannelTransport:{}},QWebChannel:class{constructor(_transport:unknown,done:(value:unknown)=>void){done({objects:{files:bridge}});}}}});
  t.after(()=>{if(previous)Object.defineProperty(globalThis,'window',previous);else Reflect.deleteProperty(globalThis,'window');});
  await initializePlatform();
  const file:ResearchFile={id:'test',name:'signal.wav',kind:'audio',size:64};
  const view={channel:0,start:0,end:1,width:300};
  const replies=Array.from({length:10},()=>desktopFiles!.spectrogram!(file,view));
  assert.equal(previews.length,1);
  previewReady(previews[0].id,JSON.stringify({ok:true,value:{pixels_base64:'same'}}));
  await Promise.all(replies);
  const reads=Array.from({length:10},()=>desktopFiles!.parameters!(file));
  await setImmediate();assert.equal(tasks.length,1);assert.equal(tasks[0].body.op,'parameters');
  taskReady(tasks[0].id,JSON.stringify({ok:true,value:{sha256:'same',rows:[]}}));await Promise.all(reads);
  const fresh=desktopFiles!.parameters!(file);await setImmediate();assert.equal(tasks.length,2);
  taskReady(tasks[1].id,JSON.stringify({ok:true,value:{sha256:'fresh',rows:[]}}));assert.equal((await fresh).sha256,'fresh');
});
