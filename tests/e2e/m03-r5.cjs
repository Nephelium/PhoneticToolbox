// Real owned backend/child + common AppShell, standalone headless Chrome.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{spawn}=require('node:child_process'),{createInterface}=require('node:readline'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const worker=spawn(path.join(root,'.venv/m09-ui/Scripts/python.exe'),['-X','utf8','scripts/m03_ui_bridge.py','--real-only','--reaper'],{cwd:root,windowsHide:true,env:{...process.env,PYTHONPATH:path.join(root,'backend/src')+';'+path.join(root,'desktop/src')+';'+path.join(root,'packages/phonetic_core/src')}});
 const pending=new Map();let counter=0;let resolveReady,rejectReady;const ready=new Promise((resolve,reject)=>{resolveReady=resolve;rejectReady=reject;});
 createInterface({input:worker.stdout}).on('line',line=>{const v=JSON.parse(line);if(v.ready)resolveReady(v);else{const done=pending.get(v.id);pending.delete(v.id);done?.(v);}});worker.stderr.on('data',d=>process.stderr.write(d));worker.on('error',rejectReady);worker.on('exit',code=>rejectReady(Error('Bridge exited before ready: '+code)));
 const {out}=await ready;const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0},optimizeDeps:{noDiscovery:true,include:['vue']},plugins:[{name:'owned-m03-rpc',configureServer(s){s.middlewares.use('/__m03',(req,res)=>{let body='';req.on('data',d=>body+=d);req.on('end',()=>{const data=JSON.parse(body),id=++counter;pending.set(id,v=>{res.setHeader('Content-Type','application/json');res.end(JSON.stringify(v));});worker.stdin.write(JSON.stringify({...data,rpc_id:id})+'\n');});});}}]});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--mute-audio']});const context=await browser.newContext({viewport:{width:1440,height:1000}}),page=await context.newPage();const checks=[],errors=[],timings=[];const resultReads=[];page.on('request',req=>{if(req.url().endsWith('/__m03')){const b=req.postDataJSON();if(b?.op==='result')resultReads.push(b);}});page.on('pageerror',e=>errors.push(e.message));
 const click=name=>page.getByRole('button',{name,exact:true}).first().click();const plotted=()=>page.waitForFunction(()=>document.querySelectorAll('.egg-four-plots .scientific-plot>svg').length===4,{},{timeout:60000});



 try{
  let lastPreview;const submitted=[];
  page.on('response',async res=>{if(res.url().endsWith('/__m03')){const req=res.request().postDataJSON();if(req?.op==='egg_preview_update')lastPreview=(await res.json()).value;}});
  page.on('request',req=>{if(req.url().endsWith('/__m03')){const b=req.postDataJSON();if(b?.op==='egg')submitted.push(b.config);}});
  const idle=async()=>{await page.waitForTimeout(80);await page.waitForFunction(()=>document.querySelector('.egg-live-status')?.textContent==='实时预览'&&[...document.querySelectorAll('button')].some(b=>b.textContent==='保存 CSV / 三图'&&!b.disabled),{},{timeout:60000});await page.waitForTimeout(20);};
  await page.goto(server.resolvedUrls.local[0]+'tests/m03-r2.html');await click('EGG 信号分析');await click('打开 WAV 目录');await page.getByLabel('EGG 音频文件').selectOption({label:'3.wav'});await idle();
  const range=async(a,b)=>page.locator('.egg-bottom-bar .selection-controls').evaluate((el,{a,b})=>{const inputs=el.querySelectorAll('input');for(const [i,value] of [[1,b],[0,a]]){inputs[i].value=String(value);inputs[i].dispatchEvent(new Event('input',{bubbles:true}));inputs[i].dispatchEvent(new Event('change',{bubbles:true}));}},{a,b});
  await range(40,40.5);await idle();
  await page.getByLabel('Praat F0',{exact:true}).check();await idle();assert.equal(lastPreview.config.f0_policy,'audio-f0/2');assert(lastPreview.preview.praat.values.some(Number.isFinite));
  const before=lastPreview.preview;await page.getByLabel('REAPER F0',{exact:true}).check();await idle();const actual=lastPreview.preview.reaper;assert(actual.times.length>20);assert(actual.values.some(v=>v>30&&v<800));assert.notDeepEqual(actual,before.praat);
  for(const key of ['cq','sq','audio','egg','gci','goi'])assert.deepEqual(lastPreview.preview[key],before[key]);
  await page.getByLabel('GCI F0',{exact:true}).check();await idle();assert((await page.locator('.spec-pane').innerText()).includes('REAPER F0'));
  checks.push('real native REAPER is independently selectable beside Praat/GCI; new policy; EGG/CQ/wave data unchanged');
  await page.getByLabel('REAPER F0',{exact:true}).uncheck();await idle();assert.equal(lastPreview.preview.reaper,null);await page.getByLabel('REAPER F0',{exact:true}).check();await idle();assert.deepEqual(lastPreview.preview.reaper,actual);
  checks.push('REAPER off/on restores identical native time samples without a new source');
  for(const theme of ['light','dark'])for(const [width,height] of [[1440,1000],[960,720],[800,600]]){
    await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);await page.setViewportSize({width,height});await page.waitForTimeout(120);
    assert(await page.locator('.egg-checks').evaluate(e=>e.scrollWidth<=e.clientWidth+1));const b=await page.locator('.egg-bottom-bar').boundingBox();assert(b.y+b.height<=height);await page.screenshot({path:path.join(out,`r5-${theme}-${width}.png`)});
  }
  checks.push('6 theme/viewport layouts keep all 3 F0 controls and bottom transport within bounds');
  await page.setViewportSize({width:1440,height:1000});await click('保存 CSV / 三图');const view=page.getByRole('button',{name:/查看 .*CSV/});await view.waitFor({timeout:120000});await view.click();await page.getByRole('button',{name:'选择目录保存完整结果',exact:true}).waitFor();await click('选择目录保存完整结果');await page.getByRole('dialog').getByRole('status').filter({hasText:'已保存'}).waitFor();await click('返回分析');
  let names=await fs.readdir(path.join(out,'saved'));let jsons=await Promise.all(names.filter(n=>n.endsWith('.ptb.json')).map(async n=>JSON.parse(await fs.readFile(path.join(out,'saved',n),'utf8'))));const single=jsons.find(m=>m.config.mode==='single');assert(single);assert.equal(single.f0_analysis.praat.floor_hz,30);assert.equal(single.f0_analysis.praat.ceiling_hz,800);assert.equal(single.f0_analysis.reaper.backend,'native_reaper');assert(single.f0_analysis.reaper.binary_sha256);assert((await fs.readFile(path.join(out,'saved',single.export_names['egg_DATA.csv']),'utf8')).includes('F0_REAPER (Hz)'));
  checks.push('single-file native job saves CSV/three PNG and actual 30–800 Hz engine metadata');
  await click('批量分析');const dialog=page.getByRole('dialog',{name:'EGG 批量分析'});await dialog.getByRole('checkbox',{name:/全选/}).uncheck();await dialog.getByRole('checkbox',{name:'3.wav',exact:true}).check();await dialog.getByLabel('REAPER F0',{exact:true}).check();await dialog.screenshot({path:path.join(out,'r5-batch.png')});await click('提交所选文件');await dialog.waitFor({state:'detached'});
  await page.getByRole('button',{name:'保存本次批量结果（1）',exact:true}).waitFor({timeout:120000});await click('保存本次批量结果（1）');await page.getByRole('status').filter({hasText:'已保存 1 个批次任务'}).waitFor();
  names=await fs.readdir(path.join(out,'saved'));jsons=await Promise.all(names.filter(n=>n.endsWith('.ptb.json')).map(async n=>JSON.parse(await fs.readFile(path.join(out,'saved',n),'utf8'))));const batch=jsons.find(m=>m.config.mode==='batch');assert(batch);assert.equal(batch.f0_analysis.reaper.floor_hz,30);assert.equal(batch.f0_analysis.reaper.ceiling_hz,800);assert((await fs.readFile(path.join(out,'saved',batch.export_names['egg_DATA.csv']),'utf8')).includes('F0_REAPER (Hz)'));assert(submitted.every(c=>c.f0_policy==='audio-f0/2'&&c.keep_reaper_f0));
  checks.push('batch native job and CSV carry REAPER option with the same search bounds');
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(out,'r5-report.json'),JSON.stringify({success:true,checks,errors,submitted},null,2));console.log(JSON.stringify({out,checks},null,2));
 }catch(e){await page.screenshot({path:path.join(out,'r5-failed.png'),fullPage:true});console.error(out);throw e;}
 finally{await browser.close();await server.close();worker.stdin.write(JSON.stringify({op:'shutdown',rpc_id:++counter})+'\n');await new Promise(resolve=>worker.once('exit',resolve));}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
