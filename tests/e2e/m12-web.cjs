// Actual authenticated web upload/edit/version/save/download + separate owner.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto');
(async()=>{
 let raw='';for await(const chunk of process.stdin)raw+=chunk;const c=JSON.parse(raw),{chromium}=require(c.playwright);
 const browser=await chromium.launch({executablePath:c.browser,headless:true}),context=await browser.newContext({viewport:{width:1600,height:1100},acceptDownloads:true}),page=await context.newPage(),checks=[],errors=[];page.on('pageerror',e=>errors.push(e.message));
 const click=name=>page.getByRole('button',{name,exact:true}).first().click();
 const login=async(p,person)=>{await p.goto(c.origin+'/server/');await p.getByLabel('登录名',{exact:true}).fill(person.username);await p.getByLabel('密码',{exact:true}).fill(person.password);await p.getByRole('button',{name:'登录',exact:true}).click();await p.getByRole('heading',{name:'你的研究项目'}).waitFor();await p.getByRole('button',{name:/^M12 网页验证/}).click();};
 const assets=async()=>{const r=await context.request.get(c.origin+'/api/v1/assets?project_id='+c.people[0].project);assert.equal(r.status(),200);return (await r.json()).assets;};
 const loaded=()=>page.waitForFunction(()=>document.querySelector('.annotation-page')?.getAttribute('aria-busy')==='false'&&document.querySelector('.annotation-grid'));
 const open=async()=>{await click('进入研究工作台');await page.locator('.nav-item[title="TextGrid标注"]').click();await page.locator('.annotation-file-list button').filter({hasText:'audio_recording.wav'}).click();await loaded();};
 try{
  await login(page,c.people[0]);
  if(c.stage==='limits'){
   assert.equal((await context.request.get(c.origin+'/api/v1/assets/'+c.expired+'/content')).status(),410);await open();
   const box=await page.locator('.annotation-grid').boundingBox();await page.mouse.click(box.x+box.width*.15,box.y+box.height*.2);await page.getByLabel('编辑选中标注文本').fill('quota');await page.getByLabel('编辑选中标注文本').press('Enter');
   await click('保存 TextGrid *');await page.getByRole('alert').filter({hasText:'文件空间不足'}).waitFor();assert(await page.getByRole('button',{name:'保存 TextGrid *',exact:true}).count());checks.push('actual quota rejection keeps edited document; expired asset HTTP 410');
   await fs.writeFile(path.join(c.output,'limits-report.json'),JSON.stringify({checks,errors},null,2));return;
  }
  for(const file of c.uploads){await page.getByLabel('选择上传文件',{exact:true}).setInputFiles(file);const done=page.waitForResponse(r=>r.url().includes('/uploads/')&&r.url().endsWith('/finalize'));await click('上传文件');assert.equal((await done).status(),200);}
  await open();assert.equal(await page.getByLabel('唇形共同偏移毫秒').inputValue(),'7');checks.push('real authenticated WAV/TextGrid/lab/safe lip uploads and automatic pair');
  let box=await page.locator('.annotation-grid').boundingBox();await page.mouse.click(box.x+box.width*.15,box.y+box.height*.2);await page.getByLabel('编辑选中标注文本').fill('网页 æ');await page.getByLabel('编辑选中标注文本').press('Enter');await click('保存 TextGrid *');await loaded();
  let values=await assets(),saved=values.find(a=>a.name==='audio_recording_自动保存.TextGrid');assert(saved);const original=values.find(a=>a.name==='audio_recording.TextGrid');assert(original);const r=await context.request.get(c.origin+'/api/v1/assets/'+saved.id+'/content');const bytes=await r.body();assert(bytes.toString('utf8').includes('网页 æ'));assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'),saved.sha256);checks.push('new server version is actual saved bytes with Unicode and independent original');
  const wait=page.waitForEvent('download');await click('下载当前 TextGrid');const download=await wait;await download.saveAs(path.join(c.output,'download.TextGrid'));assert.equal(await fs.readFile(path.join(c.output,'download.TextGrid'),'utf8'),bytes.toString('utf8'));checks.push('downloaded editor TextGrid exactly matches saved content');
  await page.getByLabel('唇形共同偏移毫秒').fill('31');await page.getByLabel('唇形共同偏移毫秒').press('Tab');await click('保存唇偏 *');await loaded();values=await assets();const lips=values.filter(a=>a.name==='audio_recording.lip.json');assert.equal(lips.length,2);let offsets=[];for(const asset of lips)offsets.push((await(await context.request.get(c.origin+'/api/v1/assets/'+asset.id+'/content')).json()).data.metadata.lip_manual_offset);assert(offsets.includes(.031)&&offsets.includes(.007));checks.push('lip save creates safe JSON version, retaining previous metadata version');
  // Before leaving, make another edit. The project-return action must save it.
  box=await page.locator('.annotation-grid').boundingBox();await page.mouse.click(box.x+box.width*.15,box.y+box.height*.2);await page.getByLabel('编辑选中标注文本').fill('自动保留');await page.getByLabel('编辑选中标注文本').press('Enter');await page.getByRole('button',{name:/返回项目与文件管理/}).click();await page.getByRole('button',{name:'进入研究工作台',exact:true}).waitFor();await open();
  assert.equal(await page.getByLabel('唇形共同偏移毫秒').inputValue(),'31');box=await page.locator('.annotation-grid').boundingBox();await page.mouse.click(box.x+box.width*.15,box.y+box.height*.2);assert.equal(await page.getByLabel('编辑选中标注文本').inputValue(),'自动保留');checks.push('leaving/re-entering project saves edits and selects latest versions');
  const other=await browser.newContext(),foreign=await other.newPage();await login(foreign,c.people[1]);assert.equal((await other.request.get(c.origin+'/api/v1/assets/'+saved.id+'/content')).status(),404);assert.deepEqual((await(await other.request.get(c.origin+'/api/v1/assets?project_id='+c.people[0].project)).json()).assets,[]);await foreign.getByRole('button',{name:'进入研究工作台',exact:true}).click();await foreign.locator('.nav-item[title="TextGrid标注"]').click();assert.equal(await foreign.locator('.annotation-file-list button').count(),0);checks.push('second account cannot read first account files or carry its annotation state');await other.close();
  await page.getByLabel('配色主题').selectOption('dark');await page.screenshot({path:path.join(c.output,'web-dark.png'),fullPage:true});assert.deepEqual(errors,[]);
  await fs.writeFile(path.join(c.output,'web-report.json'),JSON.stringify({checks,errors,saved_id:saved.id,original_id:original.id},null,2));
 }catch(error){await page.screenshot({path:path.join(c.output,'web-failure.png'),fullPage:true});await fs.writeFile(path.join(c.output,'web-failure.txt'),String(error)+'\n'+await page.locator('body').innerText());throw error;}
 finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
