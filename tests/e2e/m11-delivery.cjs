// Owned browser and deterministic transport; actual archive/native checks are separate.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');const root=path.resolve(__dirname,'../..');
async function main(){
 const out=process.argv[2],scratch=process.argv[3];let server,browser;const errors=[],checks=[];
 try{
  const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
  server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0,watch:null},optimizeDeps:{noDiscovery:true,include:['vue']}});await server.listen();
  const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
  browser=await chromium.launchPersistentContext(path.join(scratch,'chrome-profile'),{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,viewport:{width:1440,height:900}});
  const page=await browser.newPage();page.on('pageerror',e=>errors.push(e.message));
  await page.goto(server.resolvedUrls.local[0]+'tests/m11-delivery.html');await page.getByRole('button',{name:'组件安装与环境检查',exact:true}).click();
  assert.equal(await page.getByPlaceholder('从组件发布者的可信渠道核对').count(),0);
  for(const [label,kind] of [['下载声学模型（官方）','acoustic'],['下载发音词典（官方）','dictionary']]){
   const url=`https://mfa-models.readthedocs.io/en/latest/${kind}/index.html`;
   await browser.route(url,route=>route.fulfill({status:200,body:'Official link destination fixture'}));
   const opening=page.waitForEvent('popup');await page.getByRole('link',{name:label,exact:true}).click();const popup=await opening;
   await popup.waitForLoadState();assert.equal(popup.url(),url);await popup.close();
  }
  checks.push('two official resource links open the verified acoustic/dictionary catalog URLs in separate pages');
  await page.evaluate(()=>window.cancelPick=true);await page.getByRole('button',{name:'选择组件压缩包',exact:true}).click();
  assert.equal(await page.evaluate(()=>window.calls.length),0);await page.evaluate(()=>window.cancelPick=false);
  await page.getByRole('button',{name:'下载并安装组件',exact:true}).click();
  await page.waitForFunction(()=>document.querySelector('.download-metrics')?.textContent.includes('2.5 MB/s'));
  assert.equal(await page.getByRole('progressbar',{name:'组件安装进度'}).getAttribute('value'),'250000000');
  assert((await page.locator('.download-metrics').innerText()).includes('250.0 MB / 1.00 GB'));
  assert(await page.getByRole('button',{name:'选择组件压缩包',exact:true}).isDisabled());
  for(const theme of ['light','dark']){await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);await page.screenshot({path:path.join(out,`progress-${theme}.png`)});}
  await page.evaluate(()=>window.cat.component_status={stage:'extracting',completed:500,total:1000});
  await page.waitForFunction(()=>document.querySelector('.component-progress')?.textContent.includes('正在解压组件'));
  await page.evaluate(()=>window.finishInstall());await page.waitForFunction(()=>document.querySelector('.mfa-page')?.getAttribute('aria-busy')==='false');
  assert((await page.locator('.mfa-page').innerText()).includes('已检查并应用：mandarin_mfa.zip'));
  checks.push('download percent/bytes/speed updated while busy; extraction stage; success applies bundled model; controls serialized');
  await page.getByRole('button',{name:'选择组件压缩包',exact:true}).click();await page.waitForFunction(()=>window.calls.length===2);
  assert.deepEqual(await page.evaluate(()=>window.calls[1]),{action:'install_zip',archive:'grant'});await page.evaluate(()=>window.finishInstall());
  await page.waitForFunction(()=>document.querySelector('.mfa-page')?.getAttribute('aria-busy')==='false');
  await page.getByRole('button',{name:'选择已解压文件夹',exact:true}).click();await page.waitForFunction(()=>window.calls.length===3);
  assert.deepEqual(await page.evaluate(()=>window.calls[2]),{action:'install_folder',runtime:'grant'});await page.evaluate(()=>window.finishInstall());
  await page.waitForFunction(()=>document.querySelector('.mfa-page')?.getAttribute('aria-busy')==='false');
  checks.push('cancel picker has no side effect; ZIP and directory require no model selection, manifest or hash input');
  for(const theme of ['light','dark'])for(const width of [1440,900]){await page.setViewportSize({width,height:800});await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));await page.screenshot({path:path.join(out,`installed-${width}-${theme}.png`)});}
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(out,'report.json'),JSON.stringify({success:true,evidence:'simulated transport / actual Vue in isolated Chrome',checks,errors},null,2));console.log(JSON.stringify({out,checks}));
 }finally{if(browser)await browser.close();if(server)await server.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
