const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const config=JSON.parse(await fs.readFile(process.argv[2],'utf8')),out=config.out,dist=config.dist||path.join(root,'frontend/dist');
 if(config.frontendInputs){const hashes=JSON.parse(await fs.readFile(config.frontendInputs,'utf8'));for(const [name,record]of Object.entries(hashes))if(name.startsWith('frontend/src/'))assert.equal(require('node:crypto').createHash('sha256').update(await fs.readFile(path.join(root,name))).digest('hex'),record.sha256,'Source differs from frozen input: '+name);}
 const {importParams}=await import(pathToFileURL(path.join(root,'frontend/src/modules/speech-synthesis/state.ts')));
 const runtimeLock=JSON.parse(await fs.readFile(config.runtimeLock||path.join(root,'resources/manifests/browser-science-runtime.json'),'utf8'));
 const workers=(await fs.readdir(path.join(dist,'assets'))).filter(n=>/^science\.worker-.*\.js$/.test(n));
 let worker;
 for(const name of workers){const bytes=await fs.readFile(path.join(dist,'assets',name));if(bytes.includes(runtimeLock.archive_sha256))worker=bytes;}
 assert(worker,'Build current science worker before browser acceptance');
 // This Windows host intercepts localhost *.zip responses (browser sees zero
 // bytes). Alias only the owned test transport; original package hashes and
 // all production scientific code remain unchanged and are checked below.
 const aliasFetch=Buffer.from(`const originalFetch=globalThis.fetch.bind(globalThis);globalThis.fetch=(input,init)=>{if(typeof input==='string'||input instanceof URL){const u=new URL(input,globalThis.location.href);if(u.pathname.endsWith('.zip')){u.pathname=u.pathname.slice(0,-4)+'.ptbdata';input=u.href;}}return originalFetch(input,init);};\n`);
 worker=Buffer.concat([aliasFetch,worker]);
 const served=[];
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const mime=name=>name.endsWith('.js')?'text/javascript':name.endsWith('.json')?'application/json':name.endsWith('.wasm')?'application/wasm':'application/octet-stream';
 const server=await createServer({root:path.join(root,'frontend'),cacheDir:path.join(config.scratch,'vite'),server:{host:'127.0.0.1',port:0,hmr:false,watch:null},plugins:[{name:'owned-m06-runtime',configureServer(s){s.middlewares.use(async(req,res,next)=>{
  const pathname=new URL(req.url,'http://localhost').pathname;
  if(pathname==='/__m06_built_science.js'){res.setHeader('Content-Type','text/javascript');res.setHeader('Content-Length',worker.length);res.end(worker);return;}
  const match=/^\/(?:src\/)?browser-science\/(.+)$/.exec(pathname);if(!match)return next();
  const relative=decodeURIComponent(match[1]).replace(/\.ptbdata$/,'.zip');if(relative.split('/').some(p=>p==='..'||p==='')){res.statusCode=400;res.end();return;}
  for(const folder of [config.overlay,config.science]){const file=path.resolve(folder,relative);if(!file.startsWith(path.resolve(folder)+path.sep))continue;try{const raw=await fs.readFile(file);served.push({path:pathname,bytes:raw.length});res.setHeader('Content-Type',mime(file));res.setHeader('Content-Length',raw.length);res.setHeader('Content-Disposition','inline');res.end(raw);return;}catch(error){if(error.code!=='ENOENT')throw error;}}
  served.push({path:pathname,status:404});res.statusCode=404;res.end();
 });}}]});
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 let browser,page,success=false;const checks=[],errors=[],consoleMessages=[];
 try{
  await server.listen();browser=await chromium.launchPersistentContext(path.join(config.scratch,'chrome-profile'),{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,viewport:{width:1440,height:1000},acceptDownloads:true,downloadsPath:path.join(config.scratch,'downloads')});
  page=await browser.newPage();page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>consoleMessages.push(m.type()+': '+m.text()));page.setDefaultTimeout(120000);
  await page.goto(server.resolvedUrls.local[0]+'tests/m06-r6.html');await page.waitForFunction(()=>window.qaReady);
  const stdlib=await page.evaluate(async()=>{const r=await fetch('/browser-science/runtime/python_stdlib.ptbdata'),raw=await r.arrayBuffer();return {bytes:raw.byteLength,status:r.status,sha:[...new Uint8Array(await crypto.subtle.digest('SHA-256',raw))].map(n=>n.toString(16).padStart(2,'0')).join('')};});
  await fs.writeFile(path.join(out,'stdlib-readback.json'),JSON.stringify(stdlib));assert.equal(stdlib.sha,runtimeLock.files['python_stdlib.zip'].sha256);
  const scope=page.locator('.m06-page'),click=text=>scope.getByRole('button',{name:text,exact:true}).click(),idle=()=>page.waitForFunction(()=>document.querySelector('.m06-page')?.getAttribute('aria-busy')==='false');
  const snapshot=async()=>{const pending=page.waitForEvent('download');await click('导出参数');const download=await pending;return importParams(await fs.readFile(await download.path(),'utf8'));};
  for(const preferences of [{duration:.5},{f0_transform:{preset:'假声',offset_hz:180}},{f0_transform:{preset:'嘎裂',offset_hz:-50}},{duration:.5,f0_range:[200,400]}]){
   await page.evaluate(p=>window.qa.mount(p),preferences);const actual=await snapshot();assert.equal(actual.duration,preferences.duration??2);assert(actual.curves.F0.points.every(([t])=>t<=actual.duration));
   if(preferences.f0_transform)assert.equal(actual.f0_transform.preset,preferences.f0_transform.preset);
   await scope.getByLabel('IPA 元音序列',{exact:true}).fill('a');await click('生成元音');await idle();await scope.getByText('元音曲线已生成，请点击合成音频',{exact:true}).waitFor({timeout:5000});
  }
  checks.push('actual mounted web page restores short duration, falsetto, creaky and F0 range, then generates through original Pyodide core');
  await page.evaluate(()=>window.qa.mount({duration:.6}));
  const original=(await snapshot()).curves.AV;
  for(const sequence of ['a a','a']){await scope.getByLabel('IPA 元音序列',{exact:true}).fill(sequence);await click('生成元音');await idle();assert.deepEqual((await snapshot()).curves.AV,original);}
  assert.deepEqual((await snapshot()).silence,[]);checks.push('browser regeneration removes the old pause and retains the exact AV control');
  for(const name of ['A1','A2','A3','A4','A5'])assert(await scope.locator('.parameter-list button').filter({hasText:new RegExp('^'+name+' ')}).isDisabled());
  await scope.locator('.parameter-list button').filter({hasText:/^HNR /}).click();await scope.getByLabel('曲线覆盖',{exact:true}).fill('100');await click('应用覆盖');await scope.getByText('HNR 的有效编辑范围为 0–60 dB',{exact:true}).waitFor();
  await scope.getByLabel('曲线覆盖',{exact:true}).fill('60');await click('应用覆盖');
  const legacy=await snapshot();legacy.curves.HNR.override=100;await scope.locator('input[accept=".csv,.json"]').setInputFiles({name:'legacy.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(legacy))});await page.getByRole('button',{name:'应用并覆盖',exact:true}).click();
  await click('保存参数草稿');await scope.getByText('完整参数草稿已保存',{exact:true}).waitFor();await scope.locator('.parameter-list button').filter({hasText:/^F0 /}).click();assert.equal((await snapshot()).curves.HNR.override,100);
  checks.push('five inactive controls disabled; HNR new edits bounded, legacy 100 retained and navigable/saveable');
  await page.evaluate(()=>window.qa.mount({duration:.3}));await scope.getByLabel('IPA 元音序列',{exact:true}).fill('a //////i+');await click('生成元音');await idle();const short=await snapshot();assert.equal(short.silence.length,1);
  await page.evaluate(()=>window.qa.block(true));const before=await page.evaluate(()=>window.qa.stats());await click('合成音频');await idle();await scope.getByRole('button',{name:'重试保存结果',exact:true}).waitFor();assert(await scope.locator('.task-row [role=status]').innerText()==='失败');
  const failed=await page.evaluate(()=>window.qa.stats());assert.equal(failed.computes,before.computes+1);
  await click('生成元音');await scope.getByRole('alert').filter({hasText:'还有待保存的合成结果'}).waitFor();assert.deepEqual(await page.evaluate(()=>window.qa.stats()),failed);
  await page.evaluate(()=>window.qa.block(false));await click('重试保存结果');await idle();await scope.getByText(/^合成完成，/).waitFor();
  assert.deepEqual(await page.evaluate(()=>window.qa.stats()),failed);assert.equal(failed.remoteCalls,0);assert.equal(await scope.getByRole('button',{name:'重试保存结果',exact:true}).count(),0);
  const metadata=await page.evaluate(()=>window.qa.metadata());assert.equal(metadata.config.silence.length,1);assert.equal(metadata.diagnostics.silence_render_revision,'m06-silence/2');
  checks.push('real browser compute plus IndexedDB: storage failure becomes failed; retry publishes same computation without remote calls');
  for(const theme of ['light','dark']){await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);await scope.locator('.parameter-list button').filter({hasText:/^HNR /}).click();await page.screenshot({path:path.join(out,'browser-'+theme+'.png'),fullPage:true});}
  const oracles=JSON.parse(await fs.readFile(config.oracles,'utf8')),comparisons=[];
  for(const oracle of oracles){
   const output=await page.evaluate(o=>window.qa.compute(o.action,o.parameters,o.seed,new AbortController().signal),oracle);
   const metadata=JSON.parse(Buffer.from(output.files.find(f=>f.name==='m06.ptb.json').data,'base64').toString('utf8'));
   assert.deepEqual(metadata.config.curves.AV,oracle.config.curves.AV);
   const actual=metadata.config.silence,expected=oracle.config.silence;assert.equal(actual.length,expected.length);actual.forEach((p,i)=>p.forEach((v,j)=>assert(Math.abs(v-expected[i][j])<1e-15)));
   let maxPcmDifference=0;
   if(oracle.wav){const a=Buffer.from(output.files.find(f=>f.name==='synthesis.wav').data,'base64'),b=Buffer.from(oracle.wav,'base64');assert.equal(a.length,b.length);for(let i=44;i<a.length;i+=2)maxPcmDifference=Math.max(maxPcmDifference,Math.abs(a.readInt16LE(i)-b.readInt16LE(i)));assert(maxPcmDifference<=1);}
   comparisons.push({case:oracle.case,maxPcmDifference});
  }
  checks.push('native Windows vs real browser Python: exact AV controls, pause boundaries and PCM within existing 1 LSB contract');
  await fs.writeFile(path.join(out,'browser-comparison.json'),JSON.stringify(comparisons,null,2));assert.deepEqual(errors,[]);await page.evaluate(()=>window.qa.dispose());success=true;
 }catch(error){if(page){await fs.writeFile(path.join(out,'browser-failure.txt'),await page.locator('body').innerText());await fs.writeFile(path.join(out,'browser-diagnostics.json'),JSON.stringify({consoleMessages,served,qa:await page.evaluate(()=>({stats:window.qa?.stats(),errors:window.qa?.diagnostics}))},null,2));await page.screenshot({path:path.join(out,'browser-failure.png'),fullPage:true});}throw error;}
 finally{await fs.writeFile(path.join(out,'browser-report.json'),JSON.stringify({success,checks,errors,consoleWarnings:consoleMessages.filter(m=>m.startsWith('warning:')||m.startsWith('error:'))},null,2));if(browser)await browser.close();await server.close();}
 console.log(JSON.stringify({success,checks}));
}
main().catch(error=>{console.error(error);process.exitCode=1;});
