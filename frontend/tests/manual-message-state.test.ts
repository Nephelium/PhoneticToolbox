import test from 'node:test';
import assert from 'node:assert/strict';
import {setLocale,tr,messageRef} from '../src/i18n/index.ts';
import {manualPackError} from '../src/manual/errors.ts';
import {m07Message} from '../src/platform/m07.ts';
import {errorMessage as m08Message} from '../src/modules/pitch-manipulation/state.ts';
import {uiError,uiMessage} from '../src/modules/module-localization.ts';

test('retained manual and synthesis errors follow language without repeating their operation',async()=>{
 let operations=0;
 const run=()=>{operations++;return new Error('manual.pack.hashMismatch');};
 const pack=messageRef(manualPackError(run())),resource=messageRef({key:'manual.reader.resourceError'});
 const m07=messageRef(m07Message(Error('m07_no_voiced_region'))),m08=messageRef(m08Message(Error('m08_audio_decode_failed')));
 const descriptor=uiMessage('modules.pitch-manipulation.task_cancelled');
 assert.equal(m08Message(uiError(descriptor)),descriptor);
 assert.equal(m07Message(uiError(descriptor)),descriptor);
 try{
  const values:string[]=[];
  for(const language of ['en','ja','zh-CN']as const){
   await setLocale(language);values.push(pack.value);
   assert.equal(pack.value,tr('manual.reader.pack.hashMismatch'));
   assert.equal(resource.value,tr('manual.reader.resourceError'));
   assert.equal(m07.value,tr('platform.no_valid_voiced_region_check_the_recording_and_f0'));
   assert.equal(m08.value,tr('modules.pitch-manipulation.audio_cannot_be_decoded_check_file_format_and_contents'));
   assert.equal(operations,1);
  }
  assert.equal(new Set(values).size,3);
  assert.equal(m08Message(Error('unregistered diagnostic 123')),'unregistered diagnostic 123');
  assert.equal(manualPackError(Error('manual.pack.unknown')).key,'manual.reader.pack.failed');
 }finally{await setLocale('zh-CN');}
});
