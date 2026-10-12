import {test} from 'node:test';
import assert from 'node:assert/strict';
import {openImageDatabase} from '../src/platform/image-storage.ts';
test('blocked image storage rejects promptly and closes any late success handle',async()=>{
 let closed=0;const request:any={result:{close:()=>closed++}};
 const original=globalThis.indexedDB;
 Object.defineProperty(globalThis,'indexedDB',{value:{open:()=>request},configurable:true});
 try{const pending=openImageDatabase('test','image',50);request.onblocked();await assert.rejects(pending,/另一个软件窗口占用/);request.onsuccess();assert.equal(closed,1);}
 finally{Object.defineProperty(globalThis,'indexedDB',{value:original,configurable:true});}
});
test('non-responsive image storage times out without retaining a later connection',async()=>{
 let closed=0;const request:any={result:{close:()=>closed++}},original=globalThis.indexedDB;
 Object.defineProperty(globalThis,'indexedDB',{value:{open:()=>request},configurable:true});
 try{await assert.rejects(openImageDatabase('test','image',10),/暂未响应/);request.onsuccess();assert.equal(closed,1);}
 finally{Object.defineProperty(globalThis,'indexedDB',{value:original,configurable:true});}
});
