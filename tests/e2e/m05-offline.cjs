// Actual production desktop port, local service, MediaPipe worker and codecs.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {spawn,spawnSync}=require('node:child_process'),{createInterface}=require('node:readline'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const worker=spawn(path.join(root,'.venv/m14/Scripts/python.exe'),['-X','utf8','tests/support/m05_offline_host.py'],{cwd:root,windowsHide:true});
 let readyResolve,readyReject,counter=0,cancelNext=false,choices=0;const pending=new Map(),operations=[];
 const ready=new Promise((r,j)=>{readyResolve=r;readyReject=j});
 createInterface({input:worker.stdout}).on('line',line=>{try{const d=JSON.parse(line);if(d.ready)readyResolve(d);else if(d.id){pending.get(d.id)?.(d);pending.delete(d.id);}}catch{process.stdout.write(line+'\n');}});
 worker.stderr.on('data',d=>process.stderr.write(d));const exited=new Promise(resolve=>worker.once('exit',code=>{readyReject(Error('Host exited '+code));resolve(code);}));
 const config=await ready;console.log(config.out);let server,browser,page;
 const report={success:false,checks:[],layouts:[],errors:[],scope:'Windows isolated Chrome; real local API, legacy FaceMesh and codec export; no devices, Qt or EXE'};
 try{
  const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
  server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0,watch:null},optimizeDeps:{entries:['tests/m05-host.html']},plugins:[{name:'m05-owned-transport',configureServer(s){s.middlewares.use('/__m05_host',(req,res)=>{let raw='';req.on('data',d=>raw+=d);req.on('end',()=>{const request=JSON.parse(raw),id=++counter;operations.push(request.body);if(request.body.op==='choose'){choices++;if(cancelNext){cancelNext=false;res.end(JSON.stringify({ok:true,value:null}));return;}}pending.set(id,data=>{res.setHeader('Content-Type','application/json');res.end(JSON.stringify(data));});worker.stdin.write(JSON.stringify({...request,id})+'\n');});});}}]});await server.listen();
  const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
  browser=await chromium.launchPersistentContext(path.join(config.scratch,'profile'),{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,viewport:{width:1440,height:900},args:['--mute-audio']});page=await browser.newPage();page.setDefaultTimeout(60000);page.on('pageerror',e=>report.errors.push(e.message));
  const click=name=>page.getByRole('button',{name,exact:true}).click();const idle=()=>page.waitForFunction(()=>document.querySelector('.lip-page')?.getAttribute('aria-busy')==='false');
  await page.goto(server.resolvedUrls.local[0]+'tests/m05-host.html');await page.locator('nav').getByRole('button',{name:'唇形提取',exact:true}).click();
  await page.getByLabel('采集模式',{exact:true}).selectOption('offline');await page.getByRole('button',{name:'选择文件',exact:true}).waitFor();assert.equal(await page.getByRole('button',{name:'打开预览',exact:true}).count(),0);
  const chooser=page.waitForEvent('filechooser');await click('选择文件');await (await chooser).setFiles(['first.mkv','second.mkv'].map(n=>path.join(config.inputs,n)));
  assert.equal(await page.locator('.lip-file-list li').count(),2);await click('分析所选文件');await idle();
  assert.equal(await page.getByLabel('选择唇形结果',{exact:true}).locator('option').count(),2);
  assert.deepEqual((await page.getByLabel('历史唇形任务',{exact:true}).locator('option').allTextContents()).sort(),['first.mkv','second.mkv']);
  assert.equal(await page.getByLabel('动画导出质量',{exact:true}).inputValue(),'high');assert(await page.getByLabel('保存本批全部 2 个结果').isChecked());
  assert(await page.getByLabel('视频',{exact:true}).isChecked());assert(await page.getByLabel('动画',{exact:true}).isChecked());
  assert.equal(await page.locator('.workbench-right-body .module-section h2').first().innerText(),'结果保存');report.checks.push('toolbar offline mode, native multiple picker, real two-video analysis, automatic named history, result save first, 1080p and both media default');
  for(const [selection,offset] of [['0','125'],['1','-75']]){await page.getByLabel('选择唇形结果',{exact:true}).selectOption(selection);await click('检查偏移量');await page.getByLabel('唇形偏移量 ms',{exact:true}).fill(offset);await click('应用偏移量');}
  const before=choices;cancelNext=true;await click('保存（应用偏移）');await idle();assert.equal((await fs.readdir(config.saved)).length,0);
  await click('保存（应用偏移）');await idle();assert.equal(choices,before+2);report.saveStatus=await page.locator('.module-status').allTextContents();assert.equal((await fs.readdir(config.saved)).length,2,JSON.stringify(report.saveStatus));
  await click('保存（不应用偏移）');await idle();assert.equal(choices,before+3);assert.equal((await fs.readdir(config.saved)).length,4);report.checks.push('cancel writes nothing, one directory choice per batch, per-file signed offsets and zero-offset batch');
  const jobsBefore=operations.filter(x=>x.op==='m05_create').length;
  for(const name of ['导出动画 MP4','导出 GIF']){await click(name);await idle();}
  const saved=await fs.readdir(config.saved);assert.equal(saved.length,6);assert.equal(saved.filter(n=>n.endsWith('.gif')).length,1);assert.equal(saved.filter(n=>n.endsWith('.mp4')).length,1);
  assert.equal(operations.filter(x=>x.op==='m05_create').length,jobsBefore);assert.equal(operations.filter(x=>x.op==='m05_repeat').length,0);report.checks.push('standalone exports create only one corresponding animation file each and do not rerun analysis');
  for(const theme of ['light','dark'])for(const width of [1440,900,800])for(const zoom of [1,1.5]){
   await page.setViewportSize({width,height:900});await page.evaluate(({theme,zoom})=>{document.documentElement.dataset.theme=theme;document.documentElement.style.zoom=String(zoom);},{theme,zoom});
   const layout=await page.evaluate(()=>({viewport:innerWidth,scroll:document.documentElement.scrollWidth,toolbar:document.querySelector('.module-toolbar').scrollWidth}));assert(layout.scroll<=width+1,JSON.stringify({theme,width,zoom,...layout}));report.layouts.push({theme,width,zoom,...layout});
   if(width===1440&&zoom===1)await page.screenshot({path:path.join(config.out,theme+'.png'),fullPage:true});
  }
  report.checks.push('12 light/dark width/zoom layouts without horizontal overflow');
  await page.evaluate(()=>document.documentElement.style.zoom='1');await page.setViewportSize({width:1440,height:900});await page.reload();await page.locator('nav').getByRole('button',{name:'唇形提取',exact:true}).click();await page.getByLabel('历史唇形任务',{exact:true}).locator('option').first().waitFor({state:'attached'});
  assert.deepEqual((await page.getByLabel('历史唇形任务',{exact:true}).locator('option').allTextContents()).sort(),['first.mkv','second.mkv']);await click('读取历史结果');await idle();await page.waitForFunction(()=>!!document.querySelector('.lip-recording-preview')?.getAttribute('src'));await page.waitForFunction(()=>document.querySelector('.lip-recording-preview')?.readyState>=1);report.checks.push('reopen restores both named histories and source video playback');
  assert.deepEqual(report.errors,[]);report.success=true;
 }catch(e){report.failure=String(e);if(page)await page.screenshot({path:path.join(config.out,'failed.png'),fullPage:true});throw e;}
 finally{await browser?.close();await server?.close();await fs.writeFile(path.join(config.out,'report.json'),JSON.stringify(report,null,2)+'\n');worker.stdin.end(JSON.stringify({body:{op:'shutdown'}})+'\n');const code=await exited;const receipt=JSON.parse(await fs.readFile(path.join(config.out,'cleanup.json'),'utf8'));if(receipt.status==='retained'){const cleanup=spawnSync('powershell.exe',['-NoProfile','-ExecutionPolicy','Bypass','-File',path.join(root,'scripts/cleanup_test_scratch.ps1'),'-RunDirectory',config.out],{windowsHide:true,encoding:'utf8'});if(cleanup.status!==0)console.error(cleanup.stdout||cleanup.stderr);}console.log(JSON.stringify({out:config.out,success:report.success,host_exit:code,checks:report.checks.length,layouts:report.layouts.length}));if(code!==0)throw Error('Host/readback failed '+code);}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
