// Real Vue page + scientific Worker, owned profile and synthetic inputs only.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),http=require('node:http');
const {pathToFileURL}=require('node:url'),{spawnSync}=require('node:child_process'),{createHash}=require('node:crypto');
async function main(){
 const root=path.resolve(__dirname,'../..'),out=path.join(root,'output/validation/m04-reference','chrome-'+Date.now()),scratch=path.join(out,'scratch');await fs.mkdir(scratch,{recursive:true});
 const write=(name,value)=>fs.writeFile(path.join(out,name),JSON.stringify(value,null,2)+'\n');
 const marker={schema:'ptb-test-run/1',owner_pid:process.pid,state:'active',recipe:'In-memory 16000 Hz, 3200-frame FLOAT64 stereo; same shared page and real Pyodide LPC/FFT; no account or server upload'};
 await write('run.json',marker);let browser,server;const report={success:false,checks:[],errors:[],apiRequests:[]};
 try{
  const {build}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js'))),{default:vue}=await import(pathToFileURL(path.join(root,'frontend/node_modules/@vitejs/plugin-vue/dist/index.mjs')));
  await build({configFile:false,plugins:[vue()],base:'./',root:path.join(root,'frontend'),publicDir:false,logLevel:'warn',build:{emptyOutDir:false,target:'es2022',outDir:path.join(scratch,'dist'),rolldownOptions:{input:path.join(root,'frontend/tests/lpc-reference.html')}}});
  const runtime=path.join(root,'output/validation/m04-reference/browser-runtime');
  server=http.createServer(async(req,res)=>{try{const name=decodeURIComponent(new URL(req.url,'http://localhost').pathname),isRuntime=name.startsWith('/browser-science/'),base=isRuntime?runtime:path.join(scratch,'dist'),relative=isRuntime?name.slice(17):name.slice(1),file=path.resolve(base,relative);assert(file.startsWith(base+path.sep));const data=await fs.readFile(file);res.setHeader('Content-Type',({'.js':'application/javascript','.wasm':'application/wasm','.html':'text/html','.json':'application/json','.css':'text/css','.zip':'application/zip','.whl':'application/zip'})[path.extname(file)]||'application/octet-stream');res.end(data);}catch{res.statusCode=404;res.end('Not found');}});
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
  browser=await chromium.launchPersistentContext(path.join(scratch,'profile'),{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--no-proxy-server'],viewport:{width:1440,height:1000}});
  report.browser=browser.browser().version();
  await browser.route('**/browser-science/**',async route=>{const response=await browser.request.get(route.request().url());assert.equal(response.status(),200);await route.fulfill({response});});
  const page=await browser.newPage();page.setDefaultTimeout(120000);page.on('pageerror',e=>report.errors.push(e.message));page.on('request',r=>{if(r.url().includes('/api/'))report.apiRequests.push(r.url());});
  await page.goto('http://127.0.0.1:'+server.address().port+'/tests/lpc-reference.html');
  const click=name=>page.getByRole('button',{name,exact:true}).click();
  await page.getByLabel('LPC 音频文件').selectOption('stereo');await page.locator('.wave-track svg').first().waitFor();
  await click('开始分析');await page.waitForFunction(()=>document.querySelector('.lpc-spectrum svg')||qa.failure||document.querySelector('.lpc-page [role=alert]'));
  assert.equal(await page.evaluate(()=>qa.failure),undefined);assert(await page.locator('.lpc-spectrum svg').count(),await page.locator('.lpc-page').innerText());
  const mean=await page.evaluate(()=>qa.metadata),meanPath=await page.locator('.lpc-spectrum .scientific-trace path').getAttribute('d');assert.equal(mean.config.analysis_channel,null);
  await click('显示频谱图');assert.equal(await page.locator('.lpc-spectrum .scientific-trace').count(),2);assert.equal(await page.locator('.lpc-spectrum .scientific-trace path').nth(1).getAttribute('d'),meanPath);
  assert.equal(await page.evaluate(()=>qa.requests.length),1);await click('显示频谱图');assert.equal(await page.locator('.lpc-spectrum .scientific-trace').count(),1);report.checks.push('real browser LPC/FFT; overlay toggles without new task or changed LPC trace');
  await page.getByLabel('LPC 分析声道').selectOption('0');await page.getByText(/当前文件、选区或参数已改变/).waitFor();await click('开始分析');await page.waitForFunction(()=>qa.metadata?.config.analysis_channel===0&&document.querySelector('.lpc-spectrum+p')?.textContent.includes('分析声道 1'));
  const selected=await page.evaluate(()=>qa.metadata);assert.notDeepEqual(mean.spectrum.magnitude_db,selected.spectrum.magnitude_db);assert.equal(selected.reference_spectrum.frequencies_hz.length,2049);assert.equal(selected.reference_spectrum.frequencies_hz.at(-1),8000);report.checks.push('selected channel changes immutable task, FFT and exact half-open ROI');
  assert.equal(await page.getByText(/当前文件、选区或参数已改变/).count(),0);
  await click('显示频谱图');await click('波形');await click('显示频谱图');assert.equal(await page.locator('.lpc-spectrum .scientific-trace').count(),2);await click('动态纵轴');
  for(const [width,height] of [[1440,1000],[1100,800],[390,844]])for(const theme of ['light','dark']){await page.setViewportSize({width,height});await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);await page.screenshot({path:path.join(out,`chrome-${width}-${theme}.png`),fullPage:true});assert.equal(await page.locator('.lpc-spectrum .scientific-trace').count(),2);report.checks.push(`layout ${width} ${theme}`);}
  const [download]=await Promise.all([page.waitForEvent('download'),click('保存 PNG 图片')]);const png=path.join(scratch,'overlay.png');await download.saveAs(png);const bytes=await fs.readFile(png);assert.equal(bytes.readUInt32BE(16),2400);assert.equal(bytes.readUInt32BE(20),1350);await fs.copyFile(png,path.join(out,'overlay.png'));report.checks.push('browser current-axis overlay PNG is 2400x1350');
  const raw=await page.evaluate(()=>btoa(String.fromCharCode(...new Uint8Array(qa.raw))));const comparison=path.join(scratch,'comparison.json');await fs.writeFile(comparison,JSON.stringify({raw,mean,selected}));
  const prefix=path.join(root,'.venv/m03-compatible'),env={...process.env,PATH:[prefix,path.join(prefix,'Library/bin'),process.env.PATH].join(path.delimiter)};
  const native=spawnSync(path.join(prefix,'python.exe'),['-B','-X','utf8',path.join(root,'scripts/verify_m04_reference.py'),'--compare-browser',comparison],{windowsHide:true,env,encoding:'utf8'});report.nativeComparison=JSON.parse(native.stdout);report.ui_success=true;await write('browser-spectrum.json',{mean,selected});assert.equal(native.status,0,native.stdout+native.stderr);report.checks.push('same WAV bytes: Windows MKL versus real Wasm LPC/FFT within existing rtol=1e-7, atol=1e-10');
  await write('browser-spectrum.json',{mean,selected});assert.deepEqual(report.errors,[]);assert.deepEqual(report.apiRequests,[]);report.browser=browser.browser().version();report.success=true;
 }catch(error){report.failure=String(error.stack||error);if(browser){const p=browser.pages().at(-1);if(p){report.dom=await p.locator('body').innerText().catch(()=>null);report.workerFailure=await p.evaluate(()=>window.qa?.failure).catch(()=>null);await p.screenshot({path:path.join(out,'failure.png'),fullPage:true}).catch(()=>{});}}throw error;}
 finally{if(browser)await browser.close();if(server)await new Promise(resolve=>server.close(resolve));await write('report.json',report);
  async function inventory(folder,prefix=''){let rows=[];for(const item of await fs.readdir(folder,{withFileTypes:true})){const relative=path.join(prefix,item.name),file=path.join(folder,item.name);assert(!item.isSymbolicLink());if(item.isDirectory())rows.push(...await inventory(file,relative));else{const raw=await fs.readFile(file);rows.push({path:relative.replaceAll('\\','/'),bytes:raw.length,sha256:createHash('sha256').update(raw).digest('hex')});}}return rows;}
  await write('scratch-manifest.json',{schema:'ptb-test-scratch-files/1',files:await inventory(scratch)});await write('run.json',{...marker,state:'ready',test_outcome:report.success?'completed':'failed'});
  const cleanup=spawnSync('powershell.exe',['-NoProfile','-ExecutionPolicy','Bypass','-File',path.join(root,'scripts/cleanup_test_scratch.ps1'),'-RunDirectory',out],{windowsHide:true,encoding:'utf8'});assert.equal(cleanup.status,0,cleanup.stdout+cleanup.stderr);console.log(JSON.stringify({out,success:report.success,checks:report.checks.length}));
 }
}
main().catch(e=>{console.error(e);process.exitCode=1;});
