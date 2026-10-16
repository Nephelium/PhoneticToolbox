// Real ResizeObserver/v-show regression, synthetic data stays in browser memory.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url'),{spawnSync,spawn}=require('node:child_process');
const {createHash}=require('node:crypto');
const root=path.resolve(__dirname,'../..'),baseline=process.argv.includes('--baseline');
async function main(){
 const out=path.join(root,'output/validation/tab-performance',(baseline?'before-':'after-')+Date.now()),scratch=path.join(out,'scratch');await fs.mkdir(scratch,{recursive:true});
 const write=(name,value)=>fs.writeFile(path.join(out,name),JSON.stringify(value,null,2)+'\n');
 const marker={schema:'ptb-test-run/1',owner_pid:process.pid,state:'active',recipe:'Six v-show tabs, in-memory 60s/48kHz PCM, 12000-point curves, deterministic spectrogram port'};await write('run.json',marker);
 const report={success:false,baseline,errors:[]};let browser,server,page;
 try{
  const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0,watch:null},optimizeDeps:{noDiscovery:true,include:['vue']}});await server.listen();
  const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
  browser=await chromium.launchPersistentContext(path.join(scratch,'profile'),{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:false,viewport:{width:1440,height:1000},args:['--mute-audio']});page=await browser.newPage();page.on('pageerror',e=>report.errors.push(e.message));
  await page.goto(server.resolvedUrls.local[0]+'tests/tab-performance.html');await page.waitForFunction(()=>document.querySelectorAll('.scientific-trace path').length===6);await page.waitForTimeout(300);
  // Visit each tab once to establish its real layout width, then measure warm switches.
  for(let id=0;id<6;id++){await page.getByRole('button',{name:'Tab '+(id+1),exact:true}).click();await page.waitForTimeout(160);}
  const snap=()=>page.evaluate(()=>({requests:[...qa.counts.requests],paths:qa.counts.paths,selection:qa.panes.map(p=>[p.wave.start,p.wave.end]),shapes:[...document.querySelectorAll('.scientific-trace path')].map(p=>p.getAttribute('d')),waveShapes:[...document.querySelectorAll('.wave-line')].map(p=>p.getAttribute('d'))}));
  const before=await snap();report.before={requests:before.requests.length,paths:before.paths};report.switchMs=[];
  for(const theme of ['light','dark']){
   await page.evaluate(theme=>document.documentElement.dataset.theme=theme,theme);
   for(let id=0;id<6;id++){const start=performance.now();await page.getByRole('button',{name:'Tab '+(id+1),exact:true}).click();await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));report.switchMs.push(performance.now()-start);await page.waitForTimeout(110);}
   await page.screenshot({path:path.join(out,theme+'.png')});
  }
  const after=await snap();report.switchRequests=after.requests.slice(before.requests.length);report.switchPathBuilds=after.paths-before.paths;
  for(let i=0;i<10;i++){await page.evaluate(()=>qa.cursor.value+=.1);await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(resolve)));}
  const cursor=await snap();report.cursorPathBuilds=cursor.paths-after.paths;
  assert.deepEqual(after.selection,before.selection);assert.deepEqual(cursor.shapes,after.shapes);assert.deepEqual(report.errors,[]);
  if(!baseline){assert.equal(report.switchRequests.length,0,'Warm tab switches must not request spectrograms');assert.equal(report.cursorPathBuilds,0,'Cursor movement must reuse scientific paths');assert.deepEqual(after.waveShapes,before.waveShapes,'Tab switches preserve waveform geometry');
   const old=after.shapes[5];
   await page.evaluate(()=>{qa.panes[5].wave.zoom=2;qa.panes[5].wave.offset=10;});await page.waitForTimeout(180);assert((await snap()).requests.length>after.requests.length,'Real viewport change still loads the spectrogram');assert.equal((await snap()).shapes[5],old);
   await page.setViewportSize({width:1300,height:1000});await page.evaluate(()=>document.querySelectorAll('section').forEach(s=>s.style.width='950px'));await page.waitForTimeout(180);assert((await snap()).requests.length>cursor.requests.length,'Real resize still refreshes resolution');
   const resized=(await snap()).shapes[5];await page.evaluate(()=>{const p=qa.panes[5],t=p.traces.value[0];p.traces.value=[{...t,values:t.values.map(v=>v===null?null:v*.5)}];});await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(resolve)));assert.notEqual((await snap()).shapes[5],resized,'New scientific values invalidate cached paths');
   await page.goto(server.resolvedUrls.local[0]+'tests/tab-performance.html?shell#M01');await page.locator('.m01-file-list .file-row').click();await page.waitForFunction(()=>document.querySelector('.wave-line'));
   await page.evaluate(()=>{window.qa.originalPane=document.querySelector('.m01-page');window.qa.originalWave=document.querySelector('.wave-line');});
   for(const id of ['M02','M03','M04']){await page.locator('.nav-item').filter({hasText:({'M02':'参数显示','M03':'EGG 信号分析','M04':'LPC 谱图'})[id]}).click();await page.waitForTimeout(180);const lists=await page.evaluate(()=>qa.counts.lists);await page.locator('#tab-M01').click();await page.waitForTimeout(180);assert.equal(await page.evaluate(()=>qa.counts.lists),lists,'Returning to M01 must not re-read directory');assert(await page.evaluate(()=>qa.originalWave===document.querySelector('.wave-line')),'Returning to M01 preserves its loaded DOM');}
   report.shellSwitches=3;
   await page.getByRole('button',{name:'关闭 参数估计',exact:true}).click();await page.waitForFunction(()=>!document.querySelector('#tab-M01'));await page.locator('.nav-item').filter({hasText:'参数估计'}).click();await page.waitForTimeout(180);assert.equal(await page.locator('.wave-line:visible').count(),0,'Closing the tab releases its loaded audio');report.closeReopen=true;
  }
  if(process.argv.includes('--qt')){await browser.close();browser=null;await new Promise((resolve,reject)=>{const child=spawn(path.join(root,'.venv/m14/Scripts/python.exe'),['-E','-s','-B','-X','utf8',path.join(root,'scripts/verify_tab_performance_qt.py'),'--url',server.resolvedUrls.local[0]+'tests/tab-performance.html'],{windowsHide:true,stdio:'inherit'});child.on('error',reject);child.on('exit',code=>code===0?resolve():reject(Error('Qt regression exit '+code)));});report.qt=true;}
  report.success=true;
 }catch(e){report.failure=String(e);if(page)await page.screenshot({path:path.join(out,'failure.png')}).catch(()=>{});throw e;}
 finally{await browser?.close();await server?.close();await write('report.json',report);const files=[];async function inventory(dir){for(const entry of await fs.readdir(dir,{withFileTypes:true})){assert(!entry.isSymbolicLink());const file=path.join(dir,entry.name);if(entry.isDirectory())await inventory(file);else try{const bytes=await fs.readFile(file);files.push({path:path.relative(scratch,file),bytes:bytes.length,sha256:createHash('sha256').update(bytes).digest('hex')});}catch(e){if(e.code!=='ENOENT')throw e;}}}await inventory(scratch);await write('scratch-manifest.json',{schema:'ptb-test-scratch-files/1',files});await write('run.json',{...marker,state:'ready',test_outcome:report.success?'completed':'failed'});const cleanup=spawnSync('powershell.exe',['-NoProfile','-ExecutionPolicy','Bypass','-File',path.join(root,'scripts/cleanup_test_scratch.ps1'),'-RunDirectory',out],{windowsHide:true,encoding:'utf8'});if(cleanup.status!==0)await write('cleanup.json',{status:'retained',reason:cleanup.stderr||cleanup.stdout});console.log(JSON.stringify({out,...report}));}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
