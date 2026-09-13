import test from 'node:test';import assert from 'node:assert/strict';
import {serverFiles,type ResearchFile} from '../src/platform/research.ts';
test('M04 server adapter binds account, CSRF and immutable audio/TextGrid references',async()=>{
 const original=globalThis.fetch,seen:{url:string;init:RequestInit}[]=[];
 globalThis.fetch=async(url,init={})=>{seen.push({url:String(url),init});return new Response(JSON.stringify(String(url).includes('fonts')?{available:true,fonts:[]}:String(url).includes('?')?{jobs:[{id:'lpc',operation:'lpc_analysis'},{id:'egg',operation:'egg_analysis'}]}:{id:'new'}),{status:200});};
 const files=serverFiles('account-a','project-a',()=>assert.fail('unexpected identity failure'),()=>'csrf-a');
 try{const audio:ResearchFile={id:'audio-a',name:'a.wav',kind:'audio',size:44,sha256:'a'.repeat(64)},grid:ResearchFile={id:'grid-a',name:'a.TextGrid',kind:'textgrid',size:50,sha256:'b'.repeat(64)};
  await files.tasks!.lpc!(audio,grid,{roi_start:.1,roi_end:.2,order:50},'once');const request=seen[0];assert.equal(request.url,'/api/v1/jobs/lpc/create');assert.equal((request.init.headers as Record<string,string>)['X-PTB-Account'],'account-a');assert.equal((request.init.headers as Record<string,string>)['X-CSRF-Token'],'csrf-a');
  assert.deepEqual(JSON.parse(String(request.init.body)),{project_id:'project-a',idempotency_key:'once',audio:{asset_id:'audio-a',sha256:audio.sha256},textgrid:{asset_id:'grid-a',sha256:grid.sha256},config:{roi_start:.1,roi_end:.2,order:50}});
  assert.deepEqual(await files.tasks!.lpcJobs!(),[{id:'lpc',operation:'lpc_analysis'}]);
  const before=seen.length;await assert.rejects(files.tasks!.lpc!({...audio,sha256:undefined},null,{roi_end:.2},'bad'),/校验值/);assert.equal(seen.length,before);
 }finally{files.dispose();globalThis.fetch=original;}
});
