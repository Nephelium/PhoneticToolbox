// Real owned backend/child + common AppShell, standalone headless Chrome.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{spawn}=require('node:child_process'),{createInterface}=require('node:readline'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const worker=spawn(path.join(root,'.venv/m09-ui/Scripts/python.exe'),['-X','utf8','scripts/m03_ui_bridge.py','--real-only'],{cwd:root,windowsHide:true,env:{...process.env,PYTHONPATH:path.join(root,'backend/src')+';'+path.join(root,'desktop/src')+';'+path.join(root,'packages/phonetic_core/src')}});
 const pending=new Map();let counter=0;let resolveReady,rejectReady;const ready=new Promise((resolve,reject)=>{resolveReady=resolve;rejectReady=reject;});
 createInterface({input:worker.stdout}).on('line',line=>{const v=JSON.parse(line);if(v.ready)resolveReady(v);else{const done=pending.get(v.id);pending.delete(v.id);done?.(v);}});worker.stderr.on('data',d=>process.stderr.write(d));worker.on('error',rejectReady);worker.on('exit',code=>rejectReady(Error('Bridge exited before ready: '+code)));
 const {out}=await ready;const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0},optimizeDeps:{noDiscovery:true,include:['vue']},plugins:[{name:'owned-m03-rpc',configureServer(s){s.middlewares.use('/__m03',(req,res)=>{let body='';req.on('data',d=>body+=d);req.on('end',()=>{const data=JSON.parse(body),id=++counter;pending.set(id,v=>{res.setHeader('Content-Type','application/json');res.end(JSON.stringify(v));});worker.stdin.write(JSON.stringify({...data,rpc_id:id})+'\n');});});}}]});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--mute-audio']});const context=await browser.newContext({viewport:{width:1440,height:1000}}),page=await context.newPage();const checks=[],errors=[],timings=[];const resultReads=[];page.on('request',req=>{if(req.url().endsWith('/__m03')){const b=req.postDataJSON();if(b?.op==='result')resultReads.push(b);}});page.on('pageerror',e=>errors.push(e.message));
 const click=name=>page.getByRole('button',{name,exact:true}).first().click();const plotted=()=>page.waitForFunction(()=>document.querySelectorAll('.egg-four-plots .scientific-plot>svg').length===4,{},{timeout:60000});



 try{
  const idle=()=>page.waitForFunction(()=>document.querySelector('.egg-live-status')?.textContent==='实时预览'&&[...document.querySelectorAll('button')].some(b=>b.textContent==='保存 CSV / 三图'&&!b.disabled),{},{timeout:60000});
  const bottom=page.getByRole('generic',{name:'EGG 时间选区与播放'});
  await page.goto(server.resolvedUrls.local[0]+'tests/m03-r2.html');await click('EGG 信号分析');
  assert(await page.locator('.egg-bottom-bar').getByRole('button',{name:'播放选区'}).isDisabled());
  await click('打开 WAV 目录');await page.getByLabel('EGG 音频文件').selectOption({label:'3.wav'});await idle();await plotted();
  const toolbar=await page.locator('.module-toolbar button').allTextContents();assert.equal(toolbar[toolbar.indexOf('刷新文件')+1],'批量分析');
  assert.equal(await page.locator('.workbench-left').getByRole('button',{name:'播放选区'}).count(),0);assert.equal(await page.locator('.workbench-left').getByRole('button',{name:'批量分析'}).count(),0);
  assert.equal(await page.locator('.workbench-left').getByLabel('EGG 选区时长').count(),1);assert.equal(await page.locator('.egg-bottom-bar .selection-controls input').count(),2);
  checks.push('top batch button immediately after refresh; shared full transport at bottom; duration/micro stay in sidebar');
  const range=async(a,b)=>page.locator('.egg-bottom-bar .selection-controls').evaluate((el,{a,b})=>{const inputs=el.querySelectorAll('input');for(const [i,value] of [[1,b],[0,a]]){inputs[i].value=String(value);inputs[i].dispatchEvent(new Event('input',{bubbles:true}));inputs[i].dispatchEvent(new Event('change',{bubbles:true}));}},{a,b});
  await range(40,40.5);await idle();assert.equal(+await page.getByLabel('EGG 选区时长').inputValue(),.5);
  const record=await page.evaluate(async()=>{const s=Object.values((await import('/src/state/workspace.ts')).states).find(s=>s.asset?.name==='3.wav');return {start:s.start,end:s.end};});assert.equal(record.start,40);assert.equal(record.end,40.5);
  await page.getByLabel('EGG 选区时长').fill('.25');await page.getByLabel('EGG 选区时长').press('Tab');await idle();assert.equal(+await page.locator('.egg-bottom-bar .selection-controls input').nth(1).inputValue(),40.25);
  await page.locator('.egg-bottom-bar').getByRole('button',{name:'全部',exact:true}).click();assert(+await page.locator('.egg-bottom-bar .selection-controls input').nth(1).inputValue()>77);await range(40,40.5);await idle();
  await page.locator('.egg-bottom-bar').getByRole('button',{name:'播放选区',exact:true}).click();await page.waitForFunction(async()=> (await import('/src/state/audio.ts')).playback.playing);await page.locator('.egg-bottom-bar').getByRole('button',{name:'停止',exact:true}).click();assert.equal(await page.evaluate(async()=> (await import('/src/state/audio.ts')).playback.playing),false);
  checks.push('endpoints, duration, select-all and source workspace stay synchronized; normalized playback starts/stops (browser muted)');
  const geometry=[];
  for(const theme of ['light','dark'])for(const [width,height] of [[1440,1000],[960,720],[800,600]]){
   await page.evaluate(theme=>document.documentElement.dataset.theme=theme,theme);await page.setViewportSize({width,height});await page.waitForTimeout(120);
   const g=await page.locator('.egg-bottom-bar').evaluate(el=>{const work=document.querySelector('.egg-page .module-workbench'),left=work.querySelector('.workbench-left').getBoundingClientRect(),center=work.querySelector('.workbench-center').getBoundingClientRect();return {top:el.getBoundingClientRect().top,bottom:el.getBoundingClientRect().bottom,width:el.clientWidth,scroll:el.scrollWidth,viewport:innerHeight,panelsOverlap:Math.min(left.right,center.right)-Math.max(left.left,center.left)>1&&Math.min(left.bottom,center.bottom)-Math.max(left.top,center.top)>1};});
   assert(!g.panelsOverlap,'Sidebar and plots must not overlap');
   assert(g.bottom<=height+1&&g.top>=0&&g.scroll<=g.width+1,JSON.stringify(g));geometry.push({theme,width,height,...g});await page.screenshot({path:path.join(out,`r4-bottom-${theme}-${width}.png`)});
  }
  checks.push('6 light/dark viewport cases keep the bottom bar visible without horizontal overflow');
  await page.setViewportSize({width:1440,height:1000});await click('批量分析');const batch=page.getByRole('dialog',{name:'EGG 批量分析'});
  await batch.getByRole('checkbox',{name:/全选/}).uncheck();await batch.getByRole('checkbox',{name:'3.wav',exact:true}).check();
  let submissions=[];await page.route('**/__m03',async route=>{const body=route.request().postDataJSON();if(body.op==='egg')submissions.push(body);await route.continue();});
  await batch.getByLabel('低通 Hz',{exact:true}).fill('20');await click('提交所选文件');await batch.getByRole('alert').filter({hasText:'高通'}).waitFor();assert.equal(submissions.length,0);
  await batch.getByLabel('低通 Hz',{exact:true}).fill('1500');
  for(const [width,height] of [[1440,1000],[800,600],[520,720]]){
   await page.setViewportSize({width,height});await page.waitForTimeout(100);const box=await batch.boundingBox();assert(box.x>=0&&box.x+box.width<=width+1);assert(await batch.locator('.dialog-body').evaluate(e=>e.scrollWidth<=e.clientWidth+1));const button=await batch.getByRole('button',{name:'提交所选文件'}).boundingBox();assert(button.y+button.height<=height);await batch.screenshot({path:path.join(out,`r4-batch-${width}.png`)});
  }
  checks.push('batch filter/output/file groups fit 3 sizes; invalid high/low relationship blocks submission');
  await page.setViewportSize({width:1440,height:1000});await click('提交所选文件');await batch.waitFor({state:'detached'});assert.equal(submissions.length,1);assert.equal(submissions[0].config.lowpass_cutoff,1500);assert.equal(submissions[0].config.highpass_cutoff,25);
  await page.getByRole('button',{name:'保存本次批量结果（1）',exact:true}).waitFor({timeout:120000});await click('保存本次批量结果（1）');await page.getByRole('status').filter({hasText:'已保存 1 个批次任务'}).waitFor();
  const names=await fs.readdir(path.join(out,'saved')),meta=JSON.parse(await fs.readFile(path.join(out,'saved',names.find(n=>n.endsWith('.ptb.json'))),'utf8'));assert.equal(meta.config.lowpass_cutoff,1500);assert.equal(meta.config.roi_start,0);assert.equal(meta.config.roi_end,null);assert.equal(+await page.getByLabel('EGG 低通频率').inputValue(),2000);
  await click('批量分析');assert.equal(await batch.getByLabel('低通 Hz',{exact:true}).inputValue(),'1500');await click('返回工作台');checks.push('real full-file batch preserves explicit 1500 Hz cutoff in submitted/saved metadata and next batch, independent of single-file 2000 Hz');
  await page.goto(server.resolvedUrls.local[0]+'tests/m03-r4-transport.html');await page.getByLabel('起点').fill('0.25');await page.getByLabel('终点').fill('1.25');await page.getByLabel('终点').press('Tab');assert.equal(await page.getByLabel('Workspace range').textContent(),'0.25,1.25');await click('全部');assert.equal(await page.getByLabel('Workspace range').textContent(),'0,2');await click('播放选区');await page.waitForFunction(async()=>(await import('/src/state/audio.ts')).playback.playing);await click('停止');assert.equal(await page.evaluate(async()=>(await import('/src/state/audio.ts')).playback.playing),false);checks.push('shared transport default single-state selection/all/playback behavior preserved');
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(out,'r4-report.json'),JSON.stringify({success:true,checks,geometry,errors},null,2));console.log(JSON.stringify({out,checks},null,2));
 }catch(e){await page.screenshot({path:path.join(out,'r4-failed.png'),fullPage:true});console.error(out);throw e;}
 finally{await browser.close();await server.close();worker.stdin.write(JSON.stringify({op:'shutdown',rpc_id:++counter})+'\n');await new Promise(resolve=>worker.once('exit',resolve));}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
