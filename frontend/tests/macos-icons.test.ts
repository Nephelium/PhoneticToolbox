import {test} from 'node:test';
import assert from 'node:assert/strict';
import {emptyIcons,validateIcons,iconRequest,installIconPort} from '../src/platform/icons.ts';

test('browser rejects Dock resizing before reading or writing local preferences',async()=>{
 await assert.rejects(iconRequest({action:'resize',id:'default',size:82}),/macOS/);
});

test('Dock size is optional and validated at the native boundary',()=>{
 assert.deepEqual(validateIcons(emptyIcons()),emptyIcons());
 const dockSize={value:82,default:82,min:60 as const,max:100 as const};
 assert.equal(validateIcons({...emptyIcons(),dockSize}).dockSize?.value,82);
 for(const value of [59,101,82.5,NaN,Infinity,true,'82']){
  assert.throws(()=>validateIcons({...emptyIcons(),dockSize:{...dockSize,value}}));
 }
 assert.throws(()=>validateIcons({...emptyIcons(),dockSize:{...dockSize,min:50}}));
 assert.throws(()=>validateIcons({...emptyIcons(),dockSize:{...dockSize,default:101}}));
});

test('native resize forwards selected identity and validates returned capability',async()=>{
 installIconPort(async request=>{
  assert.deepEqual(request,{action:'resize',id:'default',size:93});
  return {...emptyIcons(),dockSize:{value:93,default:82,min:60,max:100}};
 });
 assert.equal((await iconRequest({action:'resize',id:'default',size:93})).dockSize?.value,93);
});
