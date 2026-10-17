const fs=require('node:fs/promises'),path=require('node:path'),os=require('node:os'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const [origin,accountFile,out,scratch]=process.argv.slice(2);
const {chromium}=require(path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
function wav(){const n=8000,b=Buffer.alloc(44+n*2);b.write('RIFF');b.writeUInt32LE(b.length-8,4);b.write('WAVEfmt ',8);b.writeUInt32LE(16,16);b.writeUInt16LE(1,20);b.writeUInt16LE(1,22);b.writeUInt32LE(16000,24);b.writeUInt32LE(32000,28);b.writeUInt16LE(2,32);b.writeUInt16LE(16,34);b.write('data',36);b.writeUInt32LE(n*2,40);for(let i=0;i<n;i++)b.writeInt16LE(Math.round(6000*Math.sin(2*Math.PI*220*i/16000)+2000*Math.sin(2*Math.PI*830*i/16000)),44+i*2);return b;}
(async()=>{
 const accounts=JSON.parse(await fs.readFile(accountFile,'utf8')),account=accounts.general;
 assert(account.username.startsWith('publicqa_'));
 const ctx=await chromium.launchPersistentContext(path.join(scratch,'chrome'),{headless:process.env.PTB_PUBLIC_HEADED!=='1',executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',viewport:{width:1440,height:1050},acceptDownloads:true,downloadsPath:path.join(scratch,'downloads')});
 const page=await ctx.newPage(),errors=[],consoleErrors=[],checks=[];let session,project,asset;
 page.setDefaultTimeout(90000);page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error'&&!/status of 401/.test(m.text()))consoleErrors.push(m.text());});
 const check=s=>{checks.push(s);console.log('PASS',s);};
 async function login(){await page.getByLabel('登录名').fill(account.username);await page.getByLabel('密码',{exact:true}).fill(account.password);await page.getByRole('button',{name:'登录',exact:true}).click();await page.locator('#tool-sidebar').waitFor();session=await(await ctx.request.get(origin+'/api/v1/auth/me')).json();}
 const headers=()=>({Origin:origin,'X-CSRF-Token':session.csrf_token,'X-PTB-Account':session.user.id});
 async function api(method,route,data){const r=await ctx.request.fetch(origin+'/api/v1/'+route,{method,headers:headers(),...(data===undefined?{}:{data}),timeout:180000});assert(r.ok(),method+' '+route+' '+r.status()+' '+(await r.text()).slice(0,500));return r.status()===204?null:r.json();}
 async function upload(name,raw){const a=await api('POST','uploads',{project_id:project,name,expected_bytes:raw.length,idempotency_key:crypto.randomUUID()});for(let at=0;at<raw.length;at+=256*1024){const r=await ctx.request.put(origin+'/api/v1/uploads/'+a.id+'/blocks?offset='+at,{headers:{...headers(),'Content-Type':'application/octet-stream'},data:raw.subarray(at,at+256*1024)});assert(r.ok());}return api('POST','uploads/'+a.id+'/finalize',{sha256:hash(raw)});}
 async function paper(operation,args){const r=await ctx.request.post(origin+'/api/v1/papers/'+project+'/request',{headers:headers(),data:{operation,args},timeout:180000});assert(r.ok(),await r.text());let result;for(const line of(await r.text()).trim().split('\n')){const item=JSON.parse(line);if('ok'in item){assert(item.ok,item.error);result=item.value;}}assert(result!==undefined);return result;}
 try{
  await page.goto(origin+'/app/#verify='+account.verify_token,{waitUntil:'domcontentloaded'});
  await page.getByRole('button',{name:'确认验证',exact:true}).click();await page.getByLabel('登录名').waitFor();
  check('real HTTPS email-token confirmation; no external email sent by QA');
  await login();project=(await api('GET','projects')).projects[0].id;
  const cookie=(await ctx.cookies()).find(c=>c.name==='__Host-ptb-session');assert(cookie&&cookie.secure&&cookie.httpOnly&&cookie.path==='/');
  check('public login, secure session cookie and default workbench');
  await page.locator('#tool-sidebar').getByRole('button',{name:'LPC 谱图',exact:true}).click();
  await page.getByLabel('LPC 阶数',{exact:true}).fill('62');await page.getByLabel('LPC 阶数',{exact:true}).press('Tab');
  await page.waitForFunction(async()=>{const r=await fetch('/api/v1/preferences');const v=await r.json();return String(v.modules?.find(x=>x.module==='M04')?.values?.draft?.order)==='62';});
  await page.reload();await page.locator('#tool-sidebar').waitFor();await page.locator('#tool-sidebar').getByRole('button',{name:'LPC 谱图',exact:true}).click();assert.equal(await page.getByLabel('LPC 阶数',{exact:true}).inputValue(),'62');
  check('account settings committed to PostgreSQL and restored after refresh');
  asset=await upload('public-synthetic.wav',wav());
  let job=await api('POST','jobs/lpc/create',{project_id:project,idempotency_key:crypto.randomUUID(),audio:{asset_id:asset.id,sha256:asset.sha256},config:{roi_end:.05,font:{schema_version:'font/1',zh:'Microsoft YaHei',latin:'Segoe UI',ipa:'Doulos SIL',size_px:12,fallback_policy:'portable'}}});
  const deadline=Date.now()+150000;while(['queued','running','cancel_requested'].includes(job.state)){assert(Date.now()<deadline,'LPC deadline');await page.waitForTimeout(350);job=await api('GET','jobs/'+job.id);}
  assert.equal(job.state,'succeeded',JSON.stringify(job));assert(job.result_manifest.files.length);
  for(const f of job.result_manifest.files){const r=await ctx.request.get(origin+'/api/v1/assets/'+f.id+'/content',{headers:headers()});assert(r.ok());const raw=await r.body();assert.equal(hash(raw),f.sha256);}
  check('real HTTPS upload, Linux LPC task and downloaded result hashes');
  const pdf=await upload('公开测试论文.pdf',await fs.readFile(path.join(scratch,'fixture.pdf')));
  await paper('status',{});await paper('autoReceive',{enabled:false});
  const library=await paper('importLocal',{files:[{asset_id:pdf.id,relative:'公开测试论文.pdf'}],folder:''});const doc=library.documents.find(d=>d.title?.includes('公开测试')||d.name?.includes('公开测试'))||library.documents[0];assert(doc);
  const rendered=await paper('render',{id:doc.id,language:'original',page:1,width:800});assert(rendered.text.includes('Public phonetics smoke fixture'));assert.equal(rendered.imageWidth,800);
  check('real public paper import and bounded Linux Qt PDF rendering');
  await page.locator('#tool-sidebar').getByRole('button',{name:'MFA 自动标注',exact:true}).click();assert((await page.locator('body').innerText()).includes('服务器'));
  check('MFA web limitation retained');
  await page.locator('#tool-sidebar').getByRole('button',{name:'LPC 谱图',exact:true}).click();
  for(const theme of ['light','dark']){await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);await page.waitForTimeout(150);await page.screenshot({path:path.join(out,'public-'+theme+'.png')});}
  await page.setViewportSize({width:390,height:844});await page.screenshot({path:path.join(out,'public-mobile.png')});
  check('light/dark and mobile browser screenshots');
  await api('POST','auth/logout');assert.equal((await ctx.request.get(origin+'/api/v1/auth/me')).status(),401);
  assert.equal((await ctx.request.get(origin+'/api/v1/assets/'+asset.id+'/content')).status(),401);
  await page.setViewportSize({width:1440,height:1050});await page.reload();await login();await page.locator('#tool-sidebar').getByRole('button',{name:'LPC 谱图',exact:true}).click();assert.equal(await page.getByLabel('LPC 阶数',{exact:true}).inputValue(),'62');
  check('logout revokes files and next login restores account settings');
  assert.deepEqual(errors,[]);assert.deepEqual(consoleErrors,[]);
  await fs.writeFile(path.join(out,'browser.json'),JSON.stringify({success:true,origin,checks,errors,consoleErrors},null,2));
 }catch(e){await page.screenshot({path:path.join(out,'failure.png')});await fs.writeFile(path.join(out,'failure.json'),JSON.stringify({error:String(e),checks,errors,consoleErrors,body:await page.locator('body').innerText()},null,2));throw e;}
 finally{try{if(session)await ctx.request.post(origin+'/api/v1/auth/logout',{headers:headers()});}finally{await ctx.close();}}
})().catch(e=>{console.error(e);process.exitCode=1;});
