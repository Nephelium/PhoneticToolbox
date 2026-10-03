import test from 'node:test';
import assert from 'node:assert/strict';
import {copyText,installClipboardWriter} from '../src/platform/clipboard.ts';

test('copy preserves Unicode through native writer and propagates native failures',async()=>{
  const text='ḁ 𝼆 V𐞀 e\u0301\n中文';let written='';
  try{
    installClipboardWriter(async value=>{written=value;});
    await copyText(text);assert.equal(written,text);
    installClipboardWriter(async()=>{throw Error('native write failed');});
    await assert.rejects(copyText(text),/native write failed/);
  }finally{installClipboardWriter(undefined);}
});

test('browser path writes through Clipboard API without requiring a desktop',async()=>{
  const descriptor=Object.getOwnPropertyDescriptor(globalThis,'navigator');let written='';
  try{
    Object.defineProperty(globalThis,'navigator',{configurable:true,value:{clipboard:{writeText:async(text:string)=>{written=text;}}}});
    await copyText('t͡s');assert.equal(written,'t͡s');
    Object.defineProperty(globalThis,'navigator',{configurable:true,value:{}});
    await assert.rejects(copyText('a'),/剪贴板不可用/);
  }finally{
    if(descriptor)Object.defineProperty(globalThis,'navigator',descriptor);
    else Reflect.deleteProperty(globalThis,'navigator');
  }
});
