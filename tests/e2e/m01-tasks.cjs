// Actual PG/worker acceptance through the shared UI. Independent owned Chrome.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
(async()=>{
 let raw='';for await(const block of process.stdin)raw+=block;const config=JSON.parse(raw);
 const {chromium}=require(config.playwright),browser=await chromium.launch({executablePath:config.browser,headless:true});
 const context=await browser.newContext({viewport:{width:1440,height:900},acceptDownloads:true}),page=await context.newPage(),errors=[],checks=[],downloads=[];
 page.on('pageerror',e=>errors.push(e.message));
 try{
  const login=async p=>{await page.getByLabel('登录名',{exact:true}).fill(p.username);await page.getByLabel('密码',{exact:true}).fill(p.password);await page.getByRole('button',{name:'登录',exact:true}).click();await page.getByRole('heading',{name:'你的研究项目'}).waitFor();};
  const project=async(name='批次验证项目')=>page.getByRole('button',{name:new RegExp('^'+name)}).click();
  const enter=async()=>{await page.getByRole('button',{name:'进入研究工作台',exact:true}).click();await page.locator('.nav-item[title="参数估计"]').click();await page.locator('.m01-page').waitFor();};
  const upload=async files=>{for(const file of files){await page.getByLabel('选择上传文件',{exact:true}).setInputFiles(file);const done=page.waitForResponse(r=>r.url().includes('/uploads/')&&r.url().endsWith('/finalize'));await page.getByRole('button',{name:'上传文件',exact:true}).click();assert.equal((await done).status(),200);await page.locator('.storage-files').getByText(path.basename(file),{exact:true}).waitFor();}};
  const submit=async(label,complete='1 / 1 已完成')=>{const waiting=page.waitForResponse(r=>r.url().endsWith('/jobs/batches/create')&&r.request().method()==='POST');await page.getByRole('button',{name:label,exact:true}).click();const response=await waiting;assert.equal(response.status(),201);const batch=await response.json();await page.locator('.m01-results').getByText(complete,{exact:true}).waitFor({timeout:90000});assert.equal(await page.locator('.m01-results select').inputValue(),batch.id);return batch;};
  const saveAll=async(batch,folder)=>{const details=await(await context.request.get(config.origin+'/api/v1/jobs/batches/'+batch.id)).json();await fs.mkdir(path.join(config.output,folder));
   let index=0;for(const item of details.summary.items){if(item.state!=='succeeded')continue;const job=await(await context.request.get(config.origin+'/api/v1/jobs/'+item.job_id)).json();
    for(const file of job.result_manifest.files){const button=page.locator('.m01-result-downloads button').nth(index++);await button.waitFor();const event=page.waitForEvent('download');await button.click();const received=await event;assert.equal(await received.failure(),null);const destination=path.join(folder,received.suggestedFilename());await received.saveAs(path.join(config.output,destination));assert.equal((await fs.stat(path.join(config.output,destination))).size,file.size_bytes);downloads.push({batch:batch.id,operation:job.operation,...file,path:destination,suggested_name:received.suggestedFilename()});}}
   await fs.writeFile(path.join(config.output,'web-downloads.json'),JSON.stringify(downloads,null,2));};
  await page.goto(config.origin+'/server/');await login(config.people[0]);await project();await upload(config.uploads);checks.push('actual UI WAV and TextGrid upload through stream and finalize');await enter();
  await page.locator('.m01-file-list .file-row').first().click();await page.locator('.m01-intervals button').first().waitFor();
  const analysis=await submit('开始全列表分析');checks.push('actual browser full-list scientific batch');
  await saveAll(analysis,'analysis-downloads');assert(downloads.every(f=>f.suggested_name.startsWith(config.audio_name.slice(0,-4))));checks.push('authenticated XLSX SQLite JSON downloads retain original audio stem');
  await page.reload();await page.getByRole('heading',{name:'你的研究项目'}).waitFor();await project();await enter();await page.locator('.m01-results').getByText('1 / 1 已完成',{exact:true}).waitFor();assert.equal(await page.locator('.m01-results select').inputValue(),analysis.id);checks.push('browser reload retains persistent completed batch');
  await page.locator('.m01-file-list .file-row').first().click();await page.getByLabel('同时切分最近一次完整参数结果',{exact:true}).check();
  const cut=await submit('保存当前层切分音频');assert.notEqual(cut.id,analysis.id);
  await page.locator('.m01-result-downloads button').nth(6).waitFor();assert.equal(await page.locator('.m01-result-downloads button').count(),7);checks.push('actual parent-linked TextGrid audio and parameter cuts');
  await saveAll(cut,'cut-downloads');
  await page.locator('.parameter-summary').evaluate(el=>el.scrollTop=el.scrollHeight);
  await page.screenshot({path:path.join(config.output,'web-light.png')});await page.evaluate(()=>document.documentElement.dataset.theme='dark');await page.screenshot({path:path.join(config.output,'web-dark.png')});
  const sizes=[];for(const [width,height] of [[1920,1080],[1280,720],[1000,700],[390,844]]){await page.setViewportSize({width,height});await page.locator('.m01-results').scrollIntoViewIfNeeded();const size=await page.evaluate(()=>({width:innerWidth,document:document.documentElement.scrollWidth,panels:[...document.querySelectorAll('.workbench-pane')].map(el=>({client:el.clientWidth,scroll:el.scrollWidth}))}));assert(size.document<=width+1,JSON.stringify(size));assert(size.panels.every(p=>p.scroll<=p.client+1),JSON.stringify(size));sizes.push(size);await page.screenshot({path:path.join(config.output,`web-${width}.png`)});}
  checks.push('real completed result layouts at 1920 1280 1000 and 390 CSS px without horizontal overflow');await fs.writeFile(path.join(config.output,'web-layout.json'),JSON.stringify(sizes,null,2));
  await page.setViewportSize({width:1440,height:900});await page.getByRole('button',{name:'使用说明',exact:true}).click();await page.getByRole('heading',{name:'参数估计：分析与切分'}).waitFor();await page.keyboard.press('Escape');await page.getByRole('dialog').waitFor({state:'hidden'});checks.push('updated offline-help content and keyboard dismissal');
  await page.getByRole('button',{name:/返回项目与文件管理/}).click();await project('失败验证项目');await upload(config.failure_uploads);await enter();
  await submit('开始全列表分析','1 / 3 已完成');await page.getByText('WAV 内容或编码不受支持，请检查文件。',{exact:true}).waitFor();await page.getByText(/音频超过本次计算的 200 万采样值上限/).waitFor();assert.equal(await page.getByRole('button',{name:'重试此文件',exact:true}).count(),2);checks.push('invalid WAV and actual sample budget failures remain actionable; later valid file succeeds');
  await page.locator('.parameter-summary').evaluate(el=>el.scrollTop=el.scrollHeight);await page.screenshot({path:path.join(config.output,'web-errors.png')});
  await page.getByRole('button',{name:/返回项目与文件管理/}).click();await page.getByRole('button',{name:'退出登录',exact:true}).click();await page.getByLabel('登录名',{exact:true}).waitFor();await login(config.people[1]);await project();await enter();
  await page.locator('.m01-results').getByText('尚无处理记录。',{exact:true}).waitFor();assert.equal(await page.locator('.m01-result-downloads button').count(),0);
  const foreign=await context.request.get(config.origin+'/api/v1/jobs/batches/'+analysis.id);assert.equal(foreign.status(),404);checks.push('account switch clears UI and refuses foreign batch');
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(config.output,'web-report.json'),JSON.stringify({checks,errors},null,2));
 }finally{await context.close();await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
