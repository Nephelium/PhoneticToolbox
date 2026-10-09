// M15-R1 production AppShell, fresh Chrome profile and synthetic local files.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{pathToFileURL}=require('node:url');
async function main(){
 const root=path.resolve(__dirname,'../..'),out=path.join(root,'output/validation/m15-r1','chrome-'+Date.now());await fs.mkdir(out,{recursive:true});
 const {preview}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await preview({root:path.join(root,'frontend'),preview:{host:'127.0.0.1',port:0}}),base=server.resolvedUrls.local[0];
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
 const context=await browser.newContext({viewport:{width:1440,height:900},acceptDownloads:true}),page=await context.newPage(),checks=[],errors=[],downloads=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('dialog',d=>d.accept());page.on('download',d=>downloads.push(d));
 const button=name=>page.getByRole('button',{name,exact:true}),idle=()=>page.waitForFunction(()=>document.querySelector('.perception-page')?.getAttribute('aria-busy')==='false');
 const phase=p=>page.locator(`.m15-run[data-phase="${p}"]`).waitFor({timeout:15000});
 const initial=()=>page.getByLabel('实验范式',{exact:true}).waitFor();
 async function prepare(){await button('预检与试音').click();await idle();await page.getByLabel('已听见试音，设备正确').check();await page.getByLabel('已暂停其他录制、播放及重型任务').check();}
 async function run(name='被试甲'){await button('填写被试信息').click();assert.equal(await page.getByLabel('姓名/编号',{exact:true}).inputValue(),'');assert.equal(await page.getByLabel('女',{exact:true}).isChecked(),false);await page.getByLabel('姓名/编号',{exact:true}).fill(name);await page.getByLabel('女',{exact:true}).check();await button('建立独立会话').click();await phase('intro');assert.equal(await button('导出结果').count(),1);await button('开始实验').click();await phase('responding');}
 async function exported(format,name){await idle();await page.getByLabel('结果格式',{exact:true}).selectOption(format);const waiting=page.waitForEvent('download');await button('导出结果').click();const d=await waiting,file=path.join(out,name);await d.saveAs(file);await idle();return format==='json'?JSON.parse(await fs.readFile(file,'utf8')):file;}
 async function layouts(stage){for(const theme of ['light','dark'])for(const [width,height,size] of [[1440,900,14],[900,700,24]]){await page.setViewportSize({width,height});await page.evaluate(({theme,size})=>{document.documentElement.dataset.theme=theme;document.documentElement.style.setProperty('--body-size',size+'px');document.documentElement.style.fontSize=size+'px';},{theme,size});await page.screenshot({path:path.join(out,`${stage}-${theme}-${width}-${size}.png`),fullPage:true});assert.equal(await button('导出结果').count(),1);assert(await button('导出结果').isVisible());const r=await button('导出结果').boundingBox();assert(r.x>=0&&r.x+r.width<=width+1);}}
 try{
  await page.goto(base);await page.locator('.nav-item').filter({hasText:'感知实验'}).click();await idle();
  await page.getByLabel('导入刺激 X',{exact:true}).setInputFiles([1,2,3].map(i=>({name:`刺激${i}.txt`,mimeType:'text/plain',buffer:Buffer.from(`刺激 ${i} a̠ ŋ`)})));await idle();
  await page.getByRole('tab',{name:'参数',exact:true}).click();await page.getByLabel('试次间隔毫秒',{exact:true}).fill('0');await page.getByLabel('提示音',{exact:true}).uncheck();await prepare();await run();
  for(let i=0;i<3;i++){assert.equal(await button('导出结果').count(),1);await page.keyboard.press('j');await phase(i===2?'completed':'responding');}
  await idle();assert.equal(downloads.length,1);const complete=await exported('json','natural.json');assert.equal(complete.status,'completed');assert.equal(complete.attempts.length,3);
  const xlsx=await exported('xlsx','natural.xlsx'),csv=await exported('csv','natural.csv');
  const {read,utils}=await import(pathToFileURL(path.join(root,'frontend/src/modules/perception/vendor/xlsx.mjs')));assert.equal(utils.sheet_to_json(read(await fs.readFile(xlsx)).Sheets.Results).length,3);assert((await fs.readFile(csv,'utf8')).includes('Response_Key'));
  await button('确认导出文件已保存').click();await idle();await layouts('completed');
  await button('结束实验').click();await initial();await idle();assert.equal(await page.locator('.m15-asset').count(),3);assert.equal(await page.getByRole('tab',{name:'素材',exact:true}).getAttribute('aria-selected'),'true');assert(await page.getByLabel('已听见试音，设备正确').isDisabled());assert(await button('填写被试信息').isDisabled());
  assert.equal((await exported('json','after-return.json')).status,'completed');await button('确认导出文件已保存').click();await idle();
  await layouts('initial');checks.push('single export control, actual JSON/XLSX/CSV readback, compact results, completed end returns initial with settings/materials retained');
  const before=downloads.length;await button('查看结果').click();await phase('completed');await idle();assert.equal(downloads.length,before);assert(await button('确认导出文件已保存').isDisabled());await button('结束实验').click();await initial();await idle();checks.push('completed recovery preserves confirmation and does not redownload or downgrade status');
  await page.setViewportSize({width:1440,height:900});await page.evaluate(()=>{document.documentElement.style.fontSize='14px';document.documentElement.style.setProperty('--body-size','14px');});
  await prepare();await run('被试乙');await page.keyboard.press('f');await phase('responding');await button('结束实验').focus();await page.keyboard.press('Enter');await initial();await idle();
  const ended=await exported('json','early-end.json');assert.equal(ended.status,'ended');assert.equal(ended.nextIndex,1);assert.deepEqual(ended.attempts.map(a=>a.status),['completed','interrupted']);assert.equal(ended.attempts[1].reason,'explicit-end');assert.equal(ended.answers.q1,'被试乙');assert.notEqual(ended.participantId,complete.participantId);
  checks.push('native Enter on end button works during focused experiment; partial answer and interrupted attempt persist; new participant starts with empty questionnaire');
  await prepare();await run('被试丙');await page.evaluate(()=>{window.__m15Put=IDBObjectStore.prototype.put;IDBObjectStore.prototype.put=function(...args){if(this.name==='sessions')throw new DOMException('M15-R1 isolated quota injection','QuotaExceededError');return window.__m15Put.apply(this,args);};});
  await button('结束实验').click();await phase('saving-error');await idle();assert(await button('结束实验').isDisabled());assert(await button('返回设计器').isDisabled());
  const memory=await exported('json','failed-save-memory.json');assert.equal(memory.status,'ended');assert.equal(memory.attempts[0].status,'interrupted');await page.screenshot({path:path.join(out,'save-error.png'),fullPage:true});
  await page.evaluate(()=>{IDBObjectStore.prototype.put=window.__m15Put;});await button('重试本地保存').click();await phase('completed');await idle();await button('结束实验').click();await initial();await idle();assert.equal((await exported('json','retry-durable.json')).attempts[0].status,'interrupted');
  checks.push('real IndexedDB write failure blocks leaving, permits memory JSON, and retry persists ended status before returning');
  await prepare();await run('被试丁');await exported('json','partial.json');await phase('paused');await button('确认导出文件已保存').click();await idle();await button('明确继续 / 重新呈现').click();await phase('responding');await page.keyboard.press('j');await phase('responding');await button('暂停').click();await phase('paused');await idle();assert(await button('确认导出文件已保存').isDisabled());await button('结束实验').click();await initial();await idle();
  checks.push('new response and interruption invalidate an old export confirmation');
  // Failure injection only affects this owned browser context; no test data is deleted.
  await prepare();await page.evaluate(()=>{window.__m15Get=IDBObjectStore.prototype.get;IDBObjectStore.prototype.get=function(...args){const req=window.__m15Get.apply(this,args);if(this.name==='assets')req.addEventListener('success',()=>Object.defineProperty(req,'result',{value:undefined}));return req;};});
  await button('预检与试音').click();await idle();assert(await button('填写被试信息').isDisabled());assert.equal(await page.getByLabel('已听见试音，设备正确').isChecked(),false);await page.evaluate(()=>{IDBObjectStore.prototype.get=window.__m15Get;});checks.push('failed re-preflight clears previous ready and heard flags');
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(out,'report.json'),JSON.stringify({success:true,checks,errors,downloads:downloads.length,layouts:8,physicalTimingMeasured:false,browser:await browser.version()},null,2));console.log(out);
 }catch(e){await page.screenshot({path:path.join(out,'failure.png'),fullPage:true});await fs.writeFile(path.join(out,'failure.txt'),String(e)+'\n'+await page.locator('body').innerText());throw e;}finally{await browser.close();await new Promise(r=>server.httpServer.close(r));}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
