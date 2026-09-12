// Real authenticated server UI, own accounts and standalone Chrome.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
(async()=>{
 let raw='';for await(const chunk of process.stdin)raw+=chunk;const c=JSON.parse(raw),{chromium}=require(c.playwright);
 const browser=await chromium.launch({executablePath:c.browser,headless:true}),context=await browser.newContext({viewport:{width:1440,height:1000},acceptDownloads:true}),page=await context.newPage(),checks=[],errors=[],downloads=[],ids=[];page.on('pageerror',e=>errors.push(e.message));
 const click=name=>page.getByRole('button',{name,exact:true}).first().click();
 const login=async person=>{await page.getByLabel('登录名',{exact:true}).fill(person.username);await page.getByLabel('密码',{exact:true}).fill(person.password);await click('登录');await page.getByRole('heading',{name:'你的研究项目'}).waitFor();await page.getByRole('button',{name:/^EGG 网页验证/}).click();};
 const workbench=async()=>{await click('进入研究工作台');await page.locator('.nav-item[title="EGG 信号分析"]').click();};
 const complete=async id=>{for(let i=0;i<180;i++){const j=await(await context.request.get(c.origin+'/api/v1/jobs/'+id)).json();if(!['queued','running','cancel_requested'].includes(j.state)){assert.equal(j.state,'succeeded',JSON.stringify(j));return j;}await page.waitForTimeout(500);}throw Error('Job timeout');};
 const submit=async name=>{const next=page.waitForResponse(r=>r.url().endsWith('/jobs/egg/create'));await click(name);const response=await next;assert.equal(response.status(),201,await response.text());const job=await response.json();return complete(job.id);};
 try{
  await page.goto(c.origin+'/server/');await login(c.people[0]);
  if(c.stage==='limits'){
   assert([404,410].includes((await context.request.get(c.origin+'/api/v1/assets/'+c.expired+'/content')).status()));checks.push('expired owned output denied by actual authenticated HTTP');
   await workbench();await page.getByLabel('EGG 音频文件').selectOption({label:'EGG ɑ̃˥.wav'});await page.waitForFunction(()=>!document.querySelector('.egg-source select').disabled);
   const response=page.waitForResponse(r=>r.url().endsWith('/jobs/egg/create'));await click('更新分析');const created=await(await response).json();let job;
   for(let i=0;i<100;i++){job=await(await context.request.get(c.origin+'/api/v1/jobs/'+created.id)).json();if(job.state==='failed')break;await page.waitForTimeout(500);}
   assert.equal(job.state,'failed');assert.equal(job.error_code,'quota_exceeded');assert.equal(job.result_manifest,null);await page.getByRole('alert').filter({hasText:'文件空间不足'}).first().waitFor();checks.push('fully reserved quota fails real EGG worker without published output');
   await page.getByLabel('EGG 音频文件').selectOption({label:'silent.wav'});await page.waitForFunction(()=>!document.querySelector('.egg-source select').disabled);const rejected=page.waitForResponse(r=>r.url().endsWith('/jobs/egg/create'));await click('更新分析');const denied=await rejected;assert.equal(denied.status(),409);assert.equal((await denied.json()).detail,'input_unavailable');await page.getByRole('alert').filter({hasText:'源文件已失效或临近到期'}).first().waitFor();checks.push('near-expiry input rejected before new task creation');
   await page.screenshot({path:path.join(c.output,'limits-web.png')});await fs.writeFile(path.join(c.output,'limits-report.json'),JSON.stringify({checks,errors},null,2));return;
  }
  for(const file of c.uploads){await page.getByLabel('选择上传文件',{exact:true}).setInputFiles(file);const done=page.waitForResponse(r=>r.url().includes('/uploads/')&&r.url().endsWith('/finalize'));await click('上传文件');assert.equal((await done).status(),200);await page.locator('.storage-files').getByText(path.basename(file),{exact:true}).waitFor();}
  await workbench();await page.getByLabel('EGG 音频文件').selectOption({label:'EGG ɑ̃˥.wav'});await page.waitForFunction(()=>!document.querySelector('.egg-source select').disabled);await submit('更新分析');await page.locator('.cq-pane svg').waitFor();checks.push('authenticated upload and real preview worker');
  await page.getByLabel('EGG 选区时长').fill('.12');await page.getByLabel('EGG 选区时长').blur();await submit('更新分析');await page.locator('.cq-pane svg').waitFor();
  const single=await submit('保存 CSV / 三图');ids.push(single.id);
  const inverse=await submit('逆滤波 IF');ids.push(inverse.id);
  await click('批量分析');const batch=page.waitForResponse(r=>r.url().endsWith('/jobs/egg/create'));await click('提交所选文件');const first=await(await batch).json();await complete(first.id);await page.waitForFunction(()=>!document.querySelector('.batch-files'));await page.waitForFunction(()=>![...document.querySelectorAll('.task-row [role=status]')].some(e=>['排队中','运行中','正在取消'].includes(e.textContent)),{},{timeout:90000});
  const list=await(await context.request.get(c.origin+'/api/v1/jobs?project_id='+c.people[0].project)).json();assert(list.jobs.length>=6);ids.push(first.id);checks.push('single, inverse and batch tasks published on PostgreSQL');
  await fs.mkdir(path.join(c.output,'downloads'),{recursive:true});
  for(const [job,label] of [[single,'CSV + 三张图'],[inverse,'逆滤波'],[await complete(first.id),'批次']]){
   const metaFile=job.result_manifest.files.find(f=>f.name==='egg.ptb.json');const meta=await(await context.request.get(c.origin+'/api/v1/assets/'+metaFile.id+'/content')).json();await page.locator('.egg-result-links button').filter({hasText:meta.input_name+' · '+label}).first().click();await page.getByRole('dialog',{name:'EGG 任务结果'}).waitFor();
   if(job===inverse){await page.locator('.inverse-grid svg').nth(3).waitFor();await page.locator('dialog .audio-transport').nth(1).waitFor();}
   for(const f of job.result_manifest.files){const expected=page.getByRole('dialog').getByRole('button',{name:/^下载 /}).filter({hasText:f.name==='egg.ptb.json'?'.ptb.json':f.name.slice(3)});const waiting=page.waitForEvent('download');await expected.click();const got=await waiting;const saved=job.id.slice(0,8)+'-'+got.suggestedFilename();await got.saveAs(path.join(c.output,'downloads',saved));downloads.push({...f,saved});}
   await page.screenshot({path:path.join(c.output,label==='逆滤波'?'inverse-web.png':label==='批次'?'batch-web.png':'single-web.png')});await click('返回分析');
  }
  checks.push('named authenticated CSV, PNG and dual WAV downloads from actual result dialog');
  await page.reload();await page.getByRole('heading',{name:'你的研究项目'}).waitFor();await page.getByRole('button',{name:/^EGG 网页验证/}).click();await workbench();await page.locator('.task-row').first().waitFor();assert((await page.locator('.task-row').count())>=6);checks.push('reload retains PostgreSQL jobs');
  const other=await browser.newContext(),foreign=await other.newPage();await foreign.goto(c.origin+'/server/');await foreign.getByLabel('登录名',{exact:true}).fill(c.people[1].username);await foreign.getByLabel('密码',{exact:true}).fill(c.people[1].password);await foreign.getByRole('button',{name:'登录',exact:true}).click();await foreign.getByRole('heading',{name:'你的研究项目'}).waitFor();
  assert.equal((await other.request.get(c.origin+'/api/v1/jobs/'+single.id)).status(),404);assert.equal((await other.request.get(c.origin+'/api/v1/assets/'+downloads[0].id+'/content')).status(),404);await other.close();checks.push('independent account cannot read foreign job or result');
  await page.getByRole('button',{name:/返回项目与文件管理/}).click();await click('退出登录');await login(c.people[1]);await workbench();assert.equal(await page.locator('.task-row').count(),0);assert.equal(await page.locator('.egg-result-links button').count(),0);assert.equal(await page.locator('.egg-four-plots svg').count(),0);assert.equal(await page.getByLabel('EGG 音频文件').locator('option').count(),1);checks.push('same-browser account switch clears files, plots and history');
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(c.output,'downloads.json'),JSON.stringify(downloads,null,2));await fs.writeFile(path.join(c.output,'web-report.json'),JSON.stringify({checks,errors,job_ids:ids},null,2));
 }catch(e){await page.screenshot({path:path.join(c.output,'web-failed.png')});throw e;}finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
