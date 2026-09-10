// Actual PG/worker acceptance through the shared UI. Independent owned Chrome.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
(async()=>{
 let raw='';for await(const block of process.stdin)raw+=block;const config=JSON.parse(raw);
 const {chromium}=require(config.playwright),browser=await chromium.launch({executablePath:config.browser,headless:true});
 const context=await browser.newContext({viewport:{width:1440,height:900},acceptDownloads:true}),page=await context.newPage(),errors=[],checks=[];
 page.on('pageerror',e=>errors.push(e.message));
 try{
  const login=async p=>{await page.getByLabel('登录名',{exact:true}).fill(p.username);await page.getByLabel('密码',{exact:true}).fill(p.password);await page.getByRole('button',{name:'登录',exact:true}).click();await page.getByRole('heading',{name:'你的研究项目'}).waitFor();};
  const enter=async()=>{await page.getByRole('button',{name:/^批次验证项目/}).click();await page.getByRole('button',{name:'进入研究工作台',exact:true}).click();await page.locator('.nav-item[title="参数估计"]').click();await page.locator('.m01-page').waitFor();};
  const submit=async label=>{const waiting=page.waitForResponse(r=>r.url().endsWith('/jobs/batches/create')&&r.request().method()==='POST');await page.getByRole('button',{name:label,exact:true}).click();const response=await waiting;assert.equal(response.status(),201);const batch=await response.json();await page.locator('.m01-results').getByText('1 / 1 已完成',{exact:true}).waitFor({timeout:70000});assert.equal(await page.locator('.m01-results select').inputValue(),batch.id);return batch;};
  await page.goto(config.origin+'/server/');await login(config.people[0]);await enter();
  await page.locator('.m01-file-list .file-row').first().click();await page.locator('.m01-intervals button').first().waitFor();
  const analysis=await submit('开始全列表分析');checks.push('actual browser full-list scientific batch');
  const download=page.waitForEvent('download');await page.locator('.m01-result-downloads').getByRole('button',{name:'result.xlsx',exact:true}).click();
  const file=await download;await file.saveAs(path.join(config.output,'web-result.xlsx'));assert((await fs.stat(path.join(config.output,'web-result.xlsx'))).size>1000);checks.push('authenticated real result download');
  await page.reload();await page.getByRole('heading',{name:'你的研究项目'}).waitFor();await enter();await page.locator('.m01-results').getByText('1 / 1 已完成',{exact:true}).waitFor();assert.equal(await page.locator('.m01-results select').inputValue(),analysis.id);checks.push('browser reload retains persistent completed batch');
  await page.locator('.m01-file-list .file-row').first().click();await page.getByLabel('同时切分最近一次完整参数结果',{exact:true}).check();
  const cut=await submit('保存当前层切分音频');assert.notEqual(cut.id,analysis.id);
  await page.locator('.m01-result-downloads button').nth(6).waitFor();assert.equal(await page.locator('.m01-result-downloads button').count(),7);checks.push('actual parent-linked TextGrid audio and parameter cuts');
  await page.locator('.parameter-summary').evaluate(el=>el.scrollTop=el.scrollHeight);
  await page.screenshot({path:path.join(config.output,'web-light.png')});await page.evaluate(()=>document.documentElement.dataset.theme='dark');await page.screenshot({path:path.join(config.output,'web-dark.png')});
  await page.getByRole('button',{name:/返回项目与文件管理/}).click();await page.getByRole('button',{name:'退出登录',exact:true}).click();await page.getByLabel('登录名',{exact:true}).waitFor();await login(config.people[1]);await enter();
  await page.locator('.m01-results').getByText('尚无处理记录。',{exact:true}).waitFor();assert.equal(await page.locator('.m01-result-downloads button').count(),0);
  const foreign=await context.request.get(config.origin+'/api/v1/jobs/batches/'+analysis.id);assert.equal(foreign.status(),404);checks.push('account switch clears UI and refuses foreign batch');
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(config.output,'web-report.json'),JSON.stringify({checks,errors},null,2));
 }finally{await context.close();await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
