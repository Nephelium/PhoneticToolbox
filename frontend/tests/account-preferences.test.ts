import test from 'node:test';
import assert from 'node:assert/strict';
import {AccountPreferences,mergePreferences,accountSettingsError} from '../src/platform/account-preferences.ts';

function server(){
 const rows=new Map<string,any>(),calls:any[]=[];let offline=false;
 const fetcher:typeof fetch=async(url,init)=>{
  if(offline)throw Error('offline');const owner=(init!.headers as Record<string,string>)['X-PTB-Account'];
  calls.push({owner,url,init});
  if(!init?.body)return Response.json({allowed_fields:{M04:['draft','panelWidths'],M18:['readerZoom','paperInterval'],APP:['theme','fonts']},modules:[...rows.entries()].filter(([key])=>key.startsWith(owner+':')).map(([,value])=>value)});
  assert.equal((init.headers as Record<string,string>)['X-CSRF-Token'],'csrf');
  const module=String(url).split('/').at(-1)!,key=owner+':'+module,body=JSON.parse(String(init.body)),before=rows.get(key);
  if((before?.revision??0)!==body.expected_revision)return Response.json({detail:'settings_revision_conflict'},{status:409});
  const row={module,revision:(before?.revision??0)+1,values:mergePreferences(before?.values??{},body.changes),updated_at:'2026-10-17T00:00:00Z'};rows.set(key,row);return Response.json(row);
 };
 return {rows,calls,fetcher,setOffline:(value:boolean)=>offline=value};
}
test('new browser/device obtains the account settings entirely from the server',async()=>{
 const s=server(),a=await new AccountPreferences('alice',()=> 'csrf',s.fetcher).open();
 a.write('M04',{draft:{order:'72'}});await a.flush();a.close();
 const b=await new AccountPreferences('alice',()=> 'csrf',s.fetcher).open(),other=await new AccountPreferences('bob',()=> 'csrf',s.fetcher).open();
 assert.equal(b.read('M04').draft.order,'72');assert.deepEqual(other.read('M04'),{});b.close();other.close();
});
test('two devices editing different nested fields merge after a revision conflict',async()=>{
 const s=server(),a=await new AccountPreferences('alice',()=> 'csrf',s.fetcher).open(),b=await new AccountPreferences('alice',()=> 'csrf',s.fetcher).open();
 a.write('M04',{draft:{order:'72'}});b.write('M04',{draft:{amp_max_db:'45'}});await a.flush();await b.flush();
 assert.deepEqual(s.rows.get('alice:M04').values,{draft:{order:'72',amp_max_db:'45'}});assert.ok(s.calls.filter(c=>c.init.method==='GET').length>=3);a.close();b.close();
});
test('numeric draft hydration does not turn an unchanged field into a stale-device edit',async()=>{
 const s=server();s.rows.set('alice:M04',{module:'M04',revision:1,values:{draft:{order:72}},updated_at:'2026-10-17T00:00:00Z'});
 const a=await new AccountPreferences('alice',()=> 'csrf',s.fetcher).open(),b=await new AccountPreferences('alice',()=> 'csrf',s.fetcher).open();
 b.write('M04',{draft:{order:75}});await b.flush();a.write('M04',{draft:{order:'72',amp_max_db:'45'}});await a.flush();
 assert.deepEqual(s.rows.get('alice:M04').values,{draft:{order:75,amp_max_db:'45'}});a.close();b.close();
});
test('offline failure retains pending edits and only acknowledgement clears the error',async()=>{
 const s=server(),a=await new AccountPreferences('alice',()=> 'csrf',s.fetcher).open();
 s.setOffline(true);a.write('M04',{draft:{order:'80'}});await assert.rejects(a.flush());assert.equal(a.pending.size,1);assert.match(accountSettingsError.value,/尚未保存到服务器/);
 s.setOffline(false);await a.flush();assert.equal(a.pending.size,0);assert.equal(accountSettingsError.value,'');assert.equal(s.rows.get('alice:M04').values.draft.order,'80');a.close();
});
test('an empty flush cannot leave a completed flight blocking all later saves',async()=>{
 const s=server(),a=await new AccountPreferences('alice',()=> 'csrf',s.fetcher).open();
 await a.flush();a.write('M04',{draft:{order:'90'}});await a.flush();await a.flush();
 a.write('M04',{draft:{order:'91'}});await a.flush();assert.equal(s.rows.get('alice:M04').values.draft.order,'91');a.close();
});
test('an edit made while saving is not lost by the earlier acknowledgement',async()=>{
 const s=server();let release!:()=>void,held=false;
 const transport:typeof fetch=async(url,init)=>{if(init?.body&&!held){held=true;await new Promise<void>(resolve=>release=resolve);}return s.fetcher(url,init);};
 const a=await new AccountPreferences('alice',()=> 'csrf',transport).open();a.write('M04',{draft:{order:'70'}});const pending=a.flush();await Promise.resolve();a.write('M04',{draft:{order:'75'}});release();await pending;
 assert.equal(s.rows.get('alice:M04').values.draft.order,'75');assert.equal(a.pending.size,0);a.close();
});
test('server settings take precedence over stale account-scoped browser records',async()=>{
 const old=globalThis.localStorage;Object.defineProperty(globalThis,'localStorage',{configurable:true,value:{getItem:()=>JSON.stringify({draft:{order:'199'}})}});
 const s=server();s.rows.set('alice:M04',{module:'M04',revision:3,values:{draft:{order:'72'}},updated_at:'2026-10-17T00:00:00Z'});
 try{const a=await new AccountPreferences('alice',()=> 'csrf',s.fetcher).open();assert.equal(a.read('M04').draft.order,'72');assert.equal(s.rows.get('alice:M04').revision,3);a.close();}
 finally{Object.defineProperty(globalThis,'localStorage',{configurable:true,value:old});}
});
