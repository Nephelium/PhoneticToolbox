// M04-E authenticated local server, real PostgreSQL/worker and standalone Chrome.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
(async()=>{
 let raw='';for await(const chunk of process.stdin)raw+=chunk;
 const c=JSON.parse(raw),{chromium}=require(c.playwright);
 const browser=await chromium.launch({executablePath:c.browser,headless:true});
 const context=await browser.newContext({viewport:{width:1440,height:1000},acceptDownloads:true});
 const page=await context.newPage(),checks=[],errors=[],downloads=[],jobIds=[],apiFailures=[];
 page.on('pageerror',e=>errors.push(e.message));
 page.on('response',r=>{if(r.url().includes('/api/v1/')&&r.status()>=400)void r.json().then(v=>apiFailures.push({path:new URL(r.url()).pathname,status:r.status(),detail:v.detail??null})).catch(()=>apiFailures.push({path:new URL(r.url()).pathname,status:r.status()}));});
 const click=name=>page.getByRole('button',{name,exact:true}).first().click();
 const login=async person=>{await page.getByLabel('登录名',{exact:true}).fill(person.username);await page.getByLabel('密码',{exact:true}).fill(person.password);await click('登录');await page.getByRole('heading',{name:'你的研究项目'}).waitFor();await page.getByRole('button',{name:/^LPC 网页验证/}).click();};
 const workbench=async()=>{await click('进入研究工作台');await page.locator('.nav-item[title="LPC 谱图"]').click();await page.locator('.lpc-page').waitFor();};
 const complete=async id=>{for(let i=0;i<160;i++){const r=await context.request.get(c.origin+'/api/v1/jobs/'+id),j=await r.json();if(!['queued','running','cancel_requested'].includes(j.state))return j;await page.waitForTimeout(500);}throw Error('LPC job timeout');};
 try{
  await page.goto(c.origin+'/server/');await login(c.people[0]);
  for(const file of c.uploads){await page.getByLabel('选择上传文件',{exact:true}).setInputFiles(file);const [done]=await Promise.all([page.waitForResponse(r=>r.url().includes('/uploads/')&&r.url().endsWith('/finalize'),{timeout:10000}),click('上传文件')]);assert.equal(done.status(),200);await page.locator('.storage-files').getByText(path.basename(file),{exact:true}).waitFor();}
  checks.push('authenticated project upload publishes WAV, TextGrid and failure fixture');
  await workbench();await page.getByLabel('LPC 音频文件').selectOption({label:'LPC ɑ̃˥.wav'});
  await page.locator('.lpc-page .wave-track svg').first().waitFor();
  await page.waitForFunction(()=>!document.querySelector('.lpc-files').innerText.includes('正在读取音频'));
  assert.equal(await page.getByLabel('LPC 标注层').inputValue(),'phones');
  await page.getByLabel('LPC 选区起点').fill('.1');await page.getByLabel('LPC 选区终点').fill('.2');
  const created=page.waitForResponse(r=>r.url().endsWith('/jobs/lpc/create'));await click('开始分析');
  const accepted=await created;assert.equal(accepted.status(),201,await accepted.text());
  const job=await complete((await accepted.json()).id);assert.equal(job.state,'succeeded',JSON.stringify(job));jobIds.push(job.id);
  await page.locator('.lpc-spectrum:visible svg').waitFor({timeout:60000});
  assert.equal(await page.locator('.lpc-spectrum .lpc-label').textContent(),'ɑ̃˥');
  await page.screenshot({path:path.join(c.output,'web-spectrum.png')});
  checks.push('real authenticated preview, TextGrid label, font preflight, LPC child and visible spectrum');
  await fs.mkdir(path.join(c.output,'downloads'),{recursive:true});
  for(const file of job.result_manifest.files){
   const label=file.name.endsWith('.png')?'下载 PNG':file.name.endsWith('.wav')?'下载 选区 WAV':'下载 参数与谱值 JSON';
   const waiting=page.waitForEvent('download');await click(label);const got=await waiting;
   const saved=file.name;await got.saveAs(path.join(c.output,'downloads',saved));
   const bytes=await fs.readFile(path.join(c.output,'downloads',saved));assert.equal(require('node:crypto').createHash('sha256').update(bytes).digest('hex'),file.sha256);
   downloads.push({...file,saved});
  }
  const metadata=JSON.parse(await fs.readFile(path.join(c.output,'downloads','lpc.ptb.json'),'utf8'));
  assert.equal(metadata.selection.start_sample,4800);assert.equal(metadata.selection.end_sample,9600);assert.equal(metadata.spectrum.frequencies_hz.length,1024);
  checks.push('three browser downloads match server SHA-256 and preserve 1024 bins/half-open ROI');
  await page.getByLabel('LPC 阶数').fill('44');await click('保存参数草稿');await page.getByRole('button',{name:'关闭 LPC 谱图',exact:true}).click();
  await page.locator('.lpc-page').waitFor({state:'detached'});await page.locator('.nav-item[title="LPC 谱图"]').click();
  assert.equal(await page.getByLabel('LPC 阶数').inputValue(),'44');await page.locator('.history-links button').first().click();await page.locator('.lpc-spectrum:visible svg').waitFor();
  checks.push('module close and reopen restores persisted result and saved draft');
  await page.reload();await page.getByRole('heading',{name:'你的研究项目'}).waitFor();await page.getByRole('button',{name:/^LPC 网页验证/}).click();await workbench();
  assert(await page.locator('.history-links button').count()>=1);checks.push('browser reload retains PostgreSQL history');
  const other=await browser.newContext(),foreign=await other.newPage();await foreign.goto(c.origin+'/server/');
  await foreign.getByLabel('登录名',{exact:true}).fill(c.people[1].username);await foreign.getByLabel('密码',{exact:true}).fill(c.people[1].password);await foreign.getByRole('button',{name:'登录',exact:true}).click();await foreign.getByRole('heading',{name:'你的研究项目'}).waitFor();
  assert.equal((await other.request.get(c.origin+'/api/v1/jobs/'+job.id)).status(),404);
  assert.equal((await other.request.get(c.origin+'/api/v1/assets/'+job.result_manifest.files[0].id+'/content')).status(),404);await other.close();
  checks.push('second account cannot read task or output assets');
  await page.getByRole('button',{name:/返回项目与文件管理/}).click();await click('退出登录');await login(c.people[1]);await workbench();
  assert.equal(await page.locator('.history-links button').count(),0);assert.equal(await page.getByLabel('LPC 音频文件').locator('option').count(),1);
  checks.push('same browser account switch clears files and results');
  assert.deepEqual(errors,[]);
  await fs.writeFile(path.join(c.output,'web-report.json'),JSON.stringify({checks,errors,job_ids:jobIds,downloads},null,2));
 }catch(e){await page.screenshot({path:path.join(c.output,'web-failed.png')});await fs.writeFile(path.join(c.output,'web-failed.json'),JSON.stringify({checks,errors,apiFailures,error:String(e)},null,2));throw e;}
 finally{await context.close();await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
