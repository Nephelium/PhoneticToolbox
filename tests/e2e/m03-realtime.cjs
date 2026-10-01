// Real owned backend/child + common AppShell, standalone headless Chrome.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{spawn}=require('node:child_process'),{createInterface}=require('node:readline'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const worker=spawn(path.join(root,'.venv/m09-ui/Scripts/python.exe'),['-X','utf8','scripts/m03_ui_bridge.py',...(process.env.PTB_M03_TEST_INPUT?['--realtime-input',process.env.PTB_M03_TEST_INPUT]:[])],{cwd:root,windowsHide:true,env:{...process.env,PYTHONPATH:path.join(root,'backend/src')+';'+path.join(root,'desktop/src')+';'+path.join(root,'packages/phonetic_core/src')}});
 const pending=new Map();let counter=0;let resolveReady,rejectReady;const ready=new Promise((resolve,reject)=>{resolveReady=resolve;rejectReady=reject;});
 createInterface({input:worker.stdout}).on('line',line=>{const v=JSON.parse(line);if(v.ready)resolveReady(v);else{const done=pending.get(v.id);pending.delete(v.id);done?.(v);}});worker.stderr.on('data',d=>process.stderr.write(d));worker.on('error',rejectReady);worker.on('exit',code=>rejectReady(Error('Bridge exited before ready: '+code)));
 const {out}=await ready;const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0},optimizeDeps:{noDiscovery:true,include:['vue']},plugins:[{name:'owned-m03-rpc',configureServer(s){s.middlewares.use('/__m03',(req,res)=>{let body='';req.on('data',d=>body+=d);req.on('end',()=>{const data=JSON.parse(body),id=++counter;pending.set(id,v=>{res.setHeader('Content-Type','application/json');res.end(JSON.stringify(v));});worker.stdin.write(JSON.stringify({...data,rpc_id:id})+'\n');});});}}]});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});const context=await browser.newContext({viewport:{width:1440,height:1000}}),page=await context.newPage();const checks=[],errors=[],timings=[];const resultReads=[];page.on('request',req=>{if(req.url().endsWith('/__m03')){const b=req.postDataJSON();if(b?.op==='result')resultReads.push(b);}});page.on('pageerror',e=>errors.push(e.message));
 const click=name=>page.getByRole('button',{name,exact:true}).first().click();const plotted=()=>page.waitForFunction(()=>document.querySelectorAll('.egg-four-plots .scientific-plot>svg').length===4,{},{timeout:60000});

 let release;
 try{
  const ready=()=>page.waitForFunction(()=>{const p=document.querySelector('.egg-page');return p&&[...p.querySelectorAll('button')].some(b=>b.textContent==='保存 CSV / 三图'&&!b.disabled);},{},{timeout:20000});
  const fill=async(label,value)=>{await page.getByLabel(label).fill(String(value));await page.getByLabel(label).press('Tab');};
  const countJobs=()=>page.locator('.egg-history .task-row').count();
  await page.goto(server.resolvedUrls.local[0]+'tests/m03-live.html');await click('EGG 信号分析');await click('打开 WAV 目录');
  await page.getByLabel('EGG 音频文件').selectOption({label:'EGG ɑ̃˥.wav'});
  await page.waitForFunction(()=>document.querySelectorAll('.egg-four-plots .scientific-plot>svg').length===4,{},{timeout:12000});
  checks.push('file selection automatically computes and displays four real plots');
  const initialReads=resultReads.length;await page.getByLabel('EGG 微观窗口').fill('100');
  await page.waitForFunction(()=>document.querySelector('.audio-pane svg')?.textContent.includes('50'),{},{timeout:12000});
  await page.waitForFunction(()=>!document.querySelector('.egg-page').innerText.includes('正在更新'),{},{timeout:12000});
  assert.equal(resultReads.length-initialReads,2,'same-source update reads only metadata and PSD, reuses verified audio');checks.push('numeric parameter automatically updates and reuses verified same-source audio');
  let started=false,armed=true;
  await page.route('**/__m03',async route=>{const body=route.request().postDataJSON();if(body.op==='result'&&armed){armed=false;started=true;await new Promise(resolve=>release=resolve);}await route.continue();});
  await page.getByLabel('EGG 选区起点').fill('0.1');await page.getByLabel('EGG 选区起点').press('Tab');
  const deadline=Date.now()+15000;while(!started&&Date.now()<deadline)await page.waitForTimeout(20);assert(started);
  assert.equal(await page.locator('.egg-four-plots .scientific-plot>svg').count(),4,'last valid plot stays visible while updating');
  await page.getByLabel('EGG 选区起点').fill('0.2');await page.getByLabel('EGG 选区起点').press('Tab');
  await page.getByLabel('EGG 微观窗口').fill('150');release();
  await page.waitForFunction(()=>document.querySelector('.audio-pane header')?.textContent.includes('0.4500'),{},{timeout:15000});
  await page.waitForFunction(()=>!document.querySelector('.egg-page').innerText.includes('正在更新'),{},{timeout:15000});
  assert(await page.getByRole('button',{name:'保存 CSV / 三图',exact:true}).isEnabled());
  checks.push('changes during result read are coalesced and newest selection is displayed automatically');
  await page.screenshot({path:path.join(out,'realtime-light.png')});
  await page.getByLabel('EGG 音频文件').selectOption({label:'silence.wav'});
  await page.waitForFunction(()=>!document.querySelector('.egg-source select').disabled);
  await plotted();await page.waitForFunction(()=>!document.querySelector('.egg-page').innerText.includes('正在更新'),{},{timeout:15000});
  assert.equal(await page.locator('.egg-page [role=alert]').count(),0);checks.push('silent file loads automatically without stale signals or error');

  await page.unroute('**/__m03');
  await page.getByLabel('EGG 音频文件').selectOption({label:'EGG ɑ̃˥.wav'});await ready();
  await fill('EGG 选区起点',.123456);await ready();
  await page.locator('.egg-result-links button').first().click();
  await page.waitForFunction(()=>!document.querySelector('dialog')&&document.querySelector('.audio-pane header')?.textContent.includes('0.3735'));
  assert.equal(await page.locator('.egg-page p.notice').count(),0,'fractional history ROI must not invalidate itself');
  assert(await page.getByRole('button',{name:'播放选区',exact:true}).isEnabled());
  checks.push('fractional history restore retains original ROI and micro center and remains playable');

  await page.getByLabel('EGG 音频文件').selectOption({label:'EGG ɑ̃˥.wav'});await ready();
  let gate;
  const arm=()=>{gate={started:false,release:undefined,fault:false};return gate;};
  await page.route('**/__m03',async route=>{const b=route.request().postDataJSON(),g=gate;
    if(b.op==='result'&&g&&!g.started){g.started=true;await new Promise(resolve=>{g.release=resolve;release=resolve;});
      if(g.fault){await route.fulfill({json:{error:'asset_expired'}});return;}}
    await route.continue();
  });
  const waitGate=async g=>{const end=Date.now()+15000;while(!g.started&&Date.now()<end)await page.waitForTimeout(20);assert(g.started);};
  let g=arm();await fill('EGG 选区起点',.1);await waitGate(g);
  await fill('EGG 选区起点',.2);await click('取消分析');g.release();
  const cancelledCount=await countJobs();await page.waitForTimeout(900);
  assert.equal(await countJobs(),cancelledCount,'cancel must clear queued auto refresh');
  assert((await page.locator('.audio-pane header').innerText()).includes('0.2500'),'cancelled read must not replace prior snapshot');
  assert(await page.getByRole('button',{name:'保存 CSV / 三图',exact:true}).isDisabled());
  checks.push('cancel clears coalesced updates and rejects a late successful result');
  gate=undefined;await click('更新分析');await ready();
  assert((await page.locator('.audio-pane header').innerText()).includes('0.4500'));
  checks.push('explicit update recovers from cancellation');

  g=arm();g.fault=true;await fill('EGG 选区起点',.1);await waitGate(g);
  await page.getByLabel('EGG 音频文件').selectOption({label:'silence.wav'});
  await page.waitForFunction(()=>!document.querySelector('.egg-source select').disabled);g.release();await ready();
  assert.equal(await page.locator('.egg-page [role=alert]').count(),0);
  checks.push('file switch rejects old read error and automatically displays new silent source');
  g=arm();g.fault=true;await fill('EGG 微观窗口',200);await waitGate(g);g.release();
  await page.locator('.egg-page [role=alert]').filter({hasText:'asset_expired'}).waitFor();
  assert.equal(await page.locator('.egg-four-plots .scientific-plot>svg').count(),4);
  assert(await page.getByRole('button',{name:'保存 CSV / 三图',exact:true}).isDisabled());
  gate=undefined;await click('更新分析');await ready();
  checks.push('current read failure stays visible, preserves last plot and can recover');
  await page.unroute('**/__m03');
  await page.getByRole('checkbox',{name:'Praat F0',exact:true}).check();await ready();
  assert((await page.locator('.spec-pane .plot-legend').innerText()).includes('Praat'));
  await page.getByRole('checkbox',{name:'GCI F0',exact:true}).check();await ready();
  await click('交换声道');await ready();
  await page.getByLabel('EGG 高通频率').fill('30');await ready();
  checks.push('F0 switches, channel swap and filter changes all trigger automatic analysis');
  await page.evaluate(()=>document.documentElement.dataset.theme='dark');await page.screenshot({path:path.join(out,'realtime-dark.png')});
  await page.getByLabel('EGG 音频文件').selectOption({label:'mono.wav'});
  await page.locator('.egg-page [role=alert]').filter({hasText:'双声道'}).waitFor();
  assert.equal(await page.locator('.egg-four-plots .scientific-plot>svg').count(),0);
  checks.push('invalid mono source rejects and removes previous source analysis');
  if(process.env.PTB_M03_TEST_INPUT){
    await page.getByRole('checkbox',{name:'Praat F0',exact:true}).uncheck();
    await page.getByRole('checkbox',{name:'GCI F0',exact:true}).uncheck();
    const start=Date.now();await page.getByLabel('EGG 音频文件').selectOption({label:'realtime.wav'});await ready();
    timings.push({action:'actual 77s file selection through four plots',seconds:(Date.now()-start)/1000});
    const t=Date.now();await fill('EGG 选区起点',29.004);await fill('EGG 选区时长',1.5551);await ready();
    timings.push({action:'actual 77s file ROI update',seconds:(Date.now()-t)/1000});
    await page.screenshot({path:path.join(out,'actual-77s.png')});
    checks.push('actual user input loads and updates through real task/file and browser path');
  }
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(out,'report.json'),JSON.stringify({success:true,checks,errors,timings,schema_applied:[]},null,2));console.log(out);
 }catch(e){await page.screenshot({path:path.join(out,'failed.png')});await fs.writeFile(path.join(out,'report.json'),JSON.stringify({success:false,error:String(e),checks,errors},null,2));console.log(out);throw e;}
 finally{release?.();await browser.close();await server.close();worker.stdin.end(JSON.stringify({op:'shutdown'})+'\n');await new Promise(resolve=>worker.once('exit',resolve));}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
