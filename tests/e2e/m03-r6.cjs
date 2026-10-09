// M03-R6: real local API/REAPER-compatible core, public synthetic gain regions.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{spawn}=require('node:child_process'),{createInterface}=require('node:readline'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const worker=spawn(path.join(root,'.venv/m09-ui/Scripts/python.exe'),['-B','-X','utf8','scripts/m03_ui_bridge.py','--db-ranges'],{cwd:root,windowsHide:true,env:{...process.env,PYTHONPATH:['backend/src','desktop/src','packages/phonetic_core/src'].map(p=>path.join(root,p)).join(';')}});
 const pending=new Map();let counter=0,resolveReady,rejectReady;const ready=new Promise((r,j)=>{resolveReady=r;rejectReady=j;});
 createInterface({input:worker.stdout}).on('line',line=>{const v=JSON.parse(line);if(v.ready)resolveReady(v);else{const done=pending.get(v.id);pending.delete(v.id);done?.(v);}});worker.stderr.on('data',d=>process.stderr.write(d));worker.on('error',rejectReady);worker.on('exit',code=>rejectReady(Error('Bridge exited '+code)));
 const {out}=await ready,{createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0,watch:null},optimizeDeps:{noDiscovery:true,include:['vue']},plugins:[{name:'owned-m03-r6-rpc',configureServer(s){s.middlewares.use('/__m03',(req,res)=>{let body='';req.on('data',d=>body+=d);req.on('end',()=>{const data=JSON.parse(body),id=++counter;pending.set(id,v=>{res.setHeader('Content-Type','application/json');res.end(JSON.stringify(v));});worker.stdin.write(JSON.stringify({...data,rpc_id:id})+'\n');});});}}]});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true}),page=await browser.newPage({viewport:{width:1440,height:1000}});
 const checks=[],errors=[],responses=[],timings=[];page.on('pageerror',e=>errors.push(e.message));
 let holdNext=false,held=false;
 await page.route('**/__m03',async route=>{
  const request=route.request().postDataJSON();const response=await route.fetch();
  if(request.op==='egg_preview_update'){
   const value=await response.json();responses.push({config:request.config,result:value.value});
   if(holdNext){holdNext=false;held=true;await new Promise(r=>setTimeout(r,350));}
  }
  await route.fulfill({response});
 });
 const idle=async()=>{await page.waitForTimeout(80);await page.waitForFunction(()=>document.querySelector('.egg-live-status')?.textContent==='实时预览'&&[...document.querySelectorAll('.egg-page button')].some(b=>b.textContent==='保存 CSV / 三图'&&!b.disabled),{},{timeout:60000});};
 const range=()=>page.evaluate(()=>[...document.querySelectorAll('input[aria-label="EGG dB 下限"],input[aria-label="EGG dB 上限"]')].map(e=>Number(e.value)));
 const fill=async(label,value)=>{if(label==='EGG 选区起点'){await page.locator('.egg-bottom-bar .selection-controls').evaluate((el,a)=>{const inputs=el.querySelectorAll('input');for(const [i,v]of [[1,a+.5],[0,a]]){inputs[i].value=String(v);inputs[i].dispatchEvent(new Event('input',{bubbles:true}));inputs[i].dispatchEvent(new Event('change',{bubbles:true}));}},value);}else{await page.getByLabel(label,{exact:true}).fill(String(value));await page.getByLabel(label,{exact:true}).press('Tab');}};
 const auto=page.getByRole('button',{name:/^自动 dB/});
 try{
  await page.goto(server.resolvedUrls.local[0]+'tests/m03-r2.html');await page.getByRole('button',{name:'EGG 信号分析',exact:true}).click();assert.equal(await auto.getAttribute('aria-pressed'),'true');
  await page.getByRole('button',{name:'打开 WAV 目录',exact:true}).click();await page.getByLabel('EGG 音频文件').selectOption({label:'db-ranges.wav'});await idle();
  const loud=await range();assert.equal(loud[1]-loud[0],50);assert(await page.getByLabel('EGG dB 上限').isDisabled());
  await fill('EGG 选区起点',2.5);await idle();const quiet=await range();assert.equal(quiet[1]-quiet[0],50);assert(quiet[1]<=loud[1]-20,JSON.stringify({loud,quiet}));
  assert.deepEqual(responses.at(-1).result.config.spec_vmin,quiet[0]);assert.equal(responses.at(-1).result.config.spec_vmax,quiet[1]);
  checks.push('auto enabled before load; real PSD gain change updates range by >20 dB and returned PNG config matches');
  const spec=page.locator('.spec-pane .scientific-plot>svg');await spec.scrollIntoViewIfNeeded();const box=await spec.boundingBox();
  const t=Date.now();await page.mouse.move(box.x+box.width*.65,box.y+box.height*.5);await page.mouse.down();await page.mouse.move(box.x+box.width*.4,box.y+box.height*.5,{steps:12});await page.mouse.up();await idle();timings.push({gesture:'pan',ms:Date.now()-t});
  const latest=responses.at(-1).result;assert(latest.config.roi_start>2.5);assert.equal(latest.config.spec_vmin,(await range())[0]);
  await page.mouse.move(box.x+box.width*.5,box.y+box.height*.5);await page.mouse.wheel(0,-240);await idle();assert(responses.at(-1).result.config.roi_end-responses.at(-1).result.config.roi_start<.5);
  checks.push('actual pointer drag and wheel zoom update auto range together with current image');
  await auto.click();assert.equal(await auto.getAttribute('aria-pressed'),'false');await page.getByLabel('EGG dB 下限').fill('-90');await page.getByLabel('EGG dB 上限').fill('-20');await idle();
  await fill('EGG 选区起点',.7);await idle();assert.deepEqual(await range(),[-90,-20]);assert(await page.getByText('自动已关闭 · 手动色阶',{exact:true}).isVisible());
  await auto.click();await idle();assert.equal(await auto.getAttribute('aria-pressed'),'true');assert.equal((await range())[1]-(await range())[0],50);assert.notDeepEqual(await range(),[-90,-20]);
  checks.push('pressed/unpressed visible state; manual values stay fixed during navigation; re-enable restores auto');
  holdNext=true;await fill('EGG 选区起点',.9);for(let i=0;i<100&&!held;i++)await new Promise(r=>setTimeout(r,20));assert(held);
  await fill('EGG 选区起点',2.8);await idle();assert.equal(responses.at(-1).result.config.roi_start,2.8);assert.deepEqual(await range(),responses.at(-1).result.preview.suggested_db_range);assert.equal((await range())[1],quiet[1]);
  checks.push('delayed real response from loud interval cannot overwrite the newer quiet viewport');
  const beforeSilence=await range();await fill('EGG 选区起点',4.5);await idle();assert.deepEqual(await range(),beforeSilence);assert(await page.getByText('自动已开启 · 静音区间保留当前色阶',{exact:true}).isVisible());
  await auto.click();await page.getByRole('button',{name:'保存参数草稿',exact:true}).click();await page.reload();await page.getByRole('button',{name:'EGG 信号分析',exact:true}).first().click();assert.equal(await auto.getAttribute('aria-pressed'),'false');
  checks.push('silent interval keeps finite prior scale; manual/automatic mode persisted separately from scientific contract');
  assert.deepEqual(errors,[]);await page.screenshot({path:path.join(out,'r6-final.png')});await fs.writeFile(path.join(out,'r6-report.json'),JSON.stringify({success:true,checks,loud,quiet,timings,updates:responses.length,errors},null,2));console.log(JSON.stringify({out,checks,loud,quiet,timings}));
 }catch(e){await page.screenshot({path:path.join(out,'r6-failed.png')});await fs.writeFile(path.join(out,'r6-failed.json'),JSON.stringify({error:String(e),checks,errors,last:responses.at(-1)?.result?.config},null,2));throw e;}
 finally{await browser.close();await server.close();worker.stdin.write(JSON.stringify({op:'shutdown',rpc_id:++counter})+'\n');await new Promise(resolve=>worker.once('exit',resolve));}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
