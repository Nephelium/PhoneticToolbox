import test from 'node:test';
import assert from 'node:assert/strict';
import {messageRef,setLocale,tr} from '../src/i18n/index.ts';
import {MemoryAssetDatabase,LocalStorageError} from '../src/platform/local-assets.ts';
import {storageErrorMessage} from '../src/account/storage-error.ts';

test('one real local recording revision conflict re-translates in project storage without another write',async()=>{
 const db=new MemoryAssetDatabase();let writes=0,error:unknown;
 await db.compare('project',null,'first');
 try{writes++;await db.compare('project',null,'second');}catch(value){error=value;}
 assert(error instanceof LocalStorageError);
 const notice=messageRef(storageErrorMessage(error));
 try{
  for(const language of ['en','ja','zh-CN']as const){
   await setLocale(language);
   assert.equal(notice.value,tr('platform.the_recording_project_was_updated_in_another_window_reload'));
   assert.equal((await db.get<{key:string;value:string}>('values','project'))?.value,'first');
  }
  assert.equal(writes,1);
  notice.value=storageErrorMessage(new Error('quota_exceeded'));
  await setLocale('en');assert.equal(notice.value,tr('account.there_is_not_enough_storage_download_files_you_need'));
  await setLocale('ja');assert.equal(notice.value,tr('account.there_is_not_enough_storage_download_files_you_need'));
 }finally{await setLocale('zh-CN');}
});
