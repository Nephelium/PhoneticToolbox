import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {accountLink,accountError} from '../src/account/public-account.ts';
import {LocalStore,perceptionDatabaseName} from '../src/modules/perception/storage.ts';

test('login and registration patterns compile with the HTML Unicode sets flag',()=>{
 for(const name of ['ServerPage.vue','PublicAccountForm.vue']){
  const source=readFileSync(new URL('../src/account/'+name,import.meta.url),'utf8');
  const pattern=/pattern="([^"]+)"/.exec(source)![1];
  const valid=new RegExp('^(?:'+pattern+')$','v');
  for(const username of ['abc','user-name','name.test','name_123'])assert(valid.test(username));
  for(const username of ['ab','_name','name@domain','name space'])assert(!valid.test(username));
 }
});

test('email links accept only exact private fragments, never request URLs',()=>{
 const token='a'.repeat(43);
 assert.deepEqual(accountLink('#verify='+token),{flow:'verify',token});
 assert.deepEqual(accountLink('#reset='+token),{flow:'reset',token});
 for(const fragment of ['?verify='+token,'#verify=short','#reset='+token+'&next=https://evil.test','#verify='+token+'%0a'])assert.equal(accountLink(fragment),null);
 assert.match(accountError('email_link_invalid'),/失效/);
});

test('perception keeps the exact desktop database and isolates account/workspace scopes',()=>{
 assert.equal(perceptionDatabaseName(),'ptb-m15-local-v1');
 const keys=[undefined,'server:a:one:M15','server:b:one:M15','server:a:two:M15'].map(perceptionDatabaseName);
 assert.equal(new Set(keys).size,4);
 assert.equal(new LocalStore().name,'ptb-m15-local-v1');
});

test('closing a store prevents an async open from reviving the previous account',async()=>{
 const old=globalThis.indexedDB;let request:any;let closed=0;
 Object.defineProperty(globalThis,'indexedDB',{configurable:true,value:{open(){request={result:{close(){closed++;}}};return request;}}});
 try{
  const store=new LocalStore('server:a:one:M15'),pending=store.open();
  store.close();request.onsuccess();
  await assert.rejects(pending,/关闭/);assert.equal(closed,1);
  await assert.rejects(()=>store.open(),/关闭/);
 }finally{Object.defineProperty(globalThis,'indexedDB',{configurable:true,value:old});}
});
