/** Existing private PostgreSQL service; creates and logs out only owned QA sessions. */
const fs=require('node:fs/promises'),path=require('node:path'),os=require('node:os'),assert=require('node:assert/strict');
const {request}=require(path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
const [origin,out]=process.argv.slice(2);
(async()=>{
 const credentials=JSON.parse(await fs.readFile(process.env.PTB_VOCAL_QA_ACCOUNT,'utf8')),clients=[],checks=[],timings={};
 const check=s=>{checks.push(s);console.log(s);};
 async function login(){
  const client=await request.newContext({baseURL:origin,timeout:180000});clients.push(client);
  const challenge=await(await client.get('/api/v1/auth/challenge')).json();
  const response=await client.post('/api/v1/auth/login',{data:credentials,headers:{Origin:origin,'X-CSRF-Token':challenge.csrf_token}});assert.equal(response.status(),200);
  const auth=await response.json(),headers={Origin:origin,'X-CSRF-Token':auth.csrf_token,'X-PTB-Account':auth.user.id};
  const post=(url,data={})=>client.post(url,{data,headers});
  assert.equal((await post('/api/v1/workspaces/open',{fresh:true})).status(),200);
  const project=await(await post('/api/v1/workspaces/default-project')).json();
  return {client,headers,post,project:project.id};
 }
 const root=p=>'/api/v1/vocal/sessions?project_id='+p;
 const endpoint=(id,p)=>'/api/v1/vocal/sessions/'+id+'?project_id='+p;
 const call=(c,id,op,body={})=>c.post(endpoint(id,c.project),{op,body});
 try{
  const a=await login(),b=await login();assert.notEqual(a.project,b.project);
  const anon=await request.newContext({baseURL:origin});try{assert.equal((await anon.post(root(a.project),{headers:{Origin:origin},data:{}})).status(),401);}finally{await anon.dispose();}
  assert.equal((await a.client.post(root(a.project),{data:{},headers:{Origin:origin}})).status(),403);
  assert.equal((await a.client.post(root(a.project),{data:{},headers:{...a.headers,Origin:'https://other.invalid'}})).status(),403);
  assert.equal((await b.post(root(a.project))).status(),404);check('real PostgreSQL authentication / CSRF / Origin / workspace-project isolation');
  const openedA=await a.post(root(a.project));assert.equal(openedA.status(),200);assert.equal(openedA.headers()['cache-control'],'no-store');const aid=(await openedA.json()).id;
  const openedB=await b.post(root(b.project));assert.equal(openedB.status(),200);const bid=(await openedB.json()).id;
  assert.equal((await call(b,aid,'meta')).status(),404);
  assert.equal((await a.post(root(a.project))).status(),429);
  const meta=await(await call(a,aid,'meta')).json();assert.equal(meta.section_count,33);
  for(const section_count of [129,257])assert.equal((await call(a,aid,'pose',{params:meta.presets.a,revision:1,section_count})).status(),422);
  assert.equal((await call(a,aid,'unapproved-operation')).status(),422);
  assert.equal((await call(a,aid,'presets/save',{presets:[{id:'qa-a',name:'测试元音',params:meta.presets.a}]})).status(),200);
  assert.deepEqual(await(await call(b,bid,'presets/load')).json(),[]);check('two native sessions; third rejected; sampling/operation rejection; session profile isolation');
  assert.equal((await a.post(endpoint(aid,a.project),{op:'status',body:{padding:'x'.repeat(9_100_000)}})).status(),413);check('oversized request rejected before native dispatch');
  assert.equal((await call(a,aid,'live',{active:true})).status(),422);
  assert.equal((await call(a,aid,'preview',{duration:2})).status(),422);
  assert.equal((await call(a,aid,'animation/prepare',{frames:[{params:meta.presets.a,duration:3},{params:meta.presets.i,duration:2.01}]})).status(),422);
  assert.equal((await call(a,aid,'video/begin',{duration:5.01,width:1280,height:720,fps:30})).status(),422);check('server rejects continuous phonation, long single audition and sequences/video over five seconds');
  const pending=call(a,aid,'animation/prepare',{frames:['a','i','u','a'].map(name=>({params:meta.presets[name],duration:1.25,name}))});
  await new Promise(resolve=>setTimeout(resolve,300));
  for(const op of ['heartbeat','status']){const t=performance.now();assert.equal((await call(a,aid,op)).status(),200);timings[op+'_ms']=performance.now()-t;}
  const tBusy=performance.now();assert.equal((await call(b,bid,'pose',{params:meta.presets.i,revision:1,section_count:33})).status(),429);timings.busy_ms=performance.now()-tBusy;
  const tStop=performance.now();assert.equal((await call(a,aid,'animation/stop')).status(),200);timings.stop_ms=performance.now()-tStop;
  const cancelled=await pending;assert([409,422].includes(cancelled.status()));check('single compute slot; bounded busy reply; heartbeat and cancel remain available during native render');
  assert.equal((await a.client.delete(endpoint(aid,a.project),{headers:a.headers})).status(),200);
  assert.equal((await call(a,aid,'status')).status(),404);
  const reopened=await a.post(root(a.project));assert.equal(reopened.status(),200);const rid=(await reopened.json()).id;
  assert.equal((await a.post('/api/v1/auth/logout')).status(),204);
  assert.equal((await call(a,rid,'status')).status(),401);
  const c=await login();assert.equal((await c.post(root(a.project))).status(),404);check('explicit close frees capacity; logout fences old native/session/project access; fresh login has new workspace');
  await new Promise(resolve=>setTimeout(resolve,5500));
  await fs.writeFile(path.join(out,'boundaries.json'),JSON.stringify({success:true,scope:'actual private Linux HTTP and PostgreSQL; owned QA rows only',checks,timings},null,2));
 }finally{
  for(const client of clients){try{const me=await client.get('/api/v1/auth/me');if(me.ok()){const session=await me.json();assert.equal((await client.post('/api/v1/auth/logout',{headers:{Origin:origin,'X-CSRF-Token':session.csrf_token}})).status(),204);}}finally{await client.dispose();}}
 }
})().catch(e=>{console.error(e);process.exitCode=1;});
