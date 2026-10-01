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
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});const context=await browser.newContext({viewport:{width:1440,height:1000}}),page=await context.newPage();const checks=[],errors=[],timings=[];const resultReads=[];page.on('request',req=>{if(req.url().endsWith('/__m03')){const b=req.postDataJSON();if(b?.op==='result')resultReads.push(b);}});page.on('pageerror',e=>errors.push(e.message));
 const click=name=>page.getByRole('button',{name,exact:true}).first().click();const plotted=()=>page.waitForFunction(()=>document.querySelectorAll('.egg-four-plots .scientific-plot>svg').length===4,{},{timeout:60000});


 try{
  const idle=()=>page.waitForFunction(()=>document.querySelector('.egg-live-status')?.textContent==='实时预览'&&[...document.querySelectorAll('button')].some(b=>b.textContent==='保存 CSV / 三图'&&!b.disabled),{},{timeout:60000});
  const fill=async(label,value)=>{await page.getByLabel(label).fill(String(value));await page.getByLabel(label).press('Tab');};
  const timing=async(name,action)=>{const t=Date.now();await action();await idle();timings.push({name,ms:Date.now()-t});};
  await page.goto(server.resolvedUrls.local[0]+'tests/m03-r2.html');await click('EGG 信号分析');await click('打开 WAV 目录');
  await timing('load',()=>page.getByLabel('EGG 音频文件').selectOption({label:'3.wav'}));
  assert.equal(await page.getByLabel('EGG 高通频率').getAttribute('type'),'number');assert.equal(await page.getByLabel('EGG 低通频率').getAttribute('type'),'number');assert.equal(await page.getByLabel('EGG 低通频率').inputValue(),'2000');checks.push('numeric highpass and lowpass with 2000 Hz default');
  assert.equal(await page.getByText('声门活动',{exact:true}).count(),0);
  assert(await page.locator('.module-toolbar').getByRole('button',{name:'保存参数草稿'}).count());
  const before=await page.locator('.egg-history summary').innerText();
  await fill('EGG 选区起点',39.747);await fill('EGG 选区时长',1.349);await idle();
  await timing('micro-width',()=>fill('EGG 微观窗口',150));
  const geometry=await page.evaluate(()=>['.cq-pane','.spec-pane'].map(sel=>{const r=document.querySelector(sel+' clipPath rect');return {x:+r.getAttribute('x'),w:+r.getAttribute('width')};}));
  assert.deepEqual(geometry[0],geometry[1]);checks.push('left plot geometry aligned with permanent F0 axis');
  for(const area of ['cq','spec']){
   const svg=page.locator('.'+area+'-pane svg');const b=await svg.boundingBox();
   const x=b.x+b.width*.4,y=b.y+b.height*.5;
   await timing(area+'-click',()=>page.mouse.click(x,y));
   const lines=await page.evaluate(()=>['.cq-pane','.spec-pane'].map(sel=>document.querySelector(sel+' svg line[stroke="var(--danger)"]')?.getAttribute('x1')));assert.equal(lines[0],lines[1]);
  }
  const a=await page.locator('.audio-pane svg').boundingBox();await page.mouse.move(a.x+a.width/2,a.y+a.height/2);
  const old=await page.getByLabel('EGG 微观窗口').inputValue();await timing('micro-wheel',()=>page.mouse.wheel(0,-120));assert(+await page.getByLabel('EGG 微观窗口').inputValue()<+old);
  for(const area of ['cq','spec','audio','egg']){
   const b=await page.locator('.'+area+'-pane svg').boundingBox();const oldText=await page.locator('.audio-pane header').innerText();const oldStart=await page.getByLabel('EGG 选区起点').inputValue();
   await page.mouse.move(b.x+b.width/2,b.y+b.height/2);await page.mouse.down();await page.mouse.move(b.x+b.width/2+35,b.y+b.height/2,{steps:4});
   await page.waitForFunction(({oldText,oldStart})=>document.querySelector('.audio-pane header').innerText!==oldText||document.querySelector('input[aria-label="EGG 选区起点"]').value!==oldStart,{oldText,oldStart});
   await page.mouse.up();await idle();checks.push(area+' drag updates before release');
  }
  assert.equal(await page.locator('.egg-history summary').innerText(),before);checks.push('all gestures create zero persistent tasks');
  await page.locator('.egg-source select').focus();const wave=await page.locator('.egg-overview svg').first().evaluate(e=>{const p=e.querySelector('.wave-line')??e.querySelectorAll('path')[1];const nums=p.getAttribute('d').match(/-?\d+(?:\.\d+)?/g).map(Number);return {height:e.getBoundingClientRect().height,d:p.getAttribute('d')};});
  assert(wave.height<=93);const ys=[...wave.d.matchAll(/,(-?\d+(?:\.\d+)?)V(-?\d+(?:\.\d+)?)/g)].flatMap(m=>[+m[1],+m[2]]);assert(Math.abs(Math.max(...ys.map(y=>Math.abs(y-45)))-40.5)<1e-5);checks.push('overview display peak reaches 90 percent without changing audio');
  await page.screenshot({path:path.join(out,'realtime-four-plots.png'),fullPage:true});
  await fill('EGG 选区起点',40);await fill('EGG 选区时长',.5);await idle();await click('逆滤波 IF');
  await page.locator('.inverse-grid svg').first().waitFor({timeout:60000});assert.equal(await page.locator('.inverse-grid svg').count(),4);checks.push('IF finishes into result dialog automatically');
  const heights=await page.locator('.inverse-grid svg').evaluateAll(es=>es.map(e=>e.getBoundingClientRect().height));assert(heights.every(v=>v>=299));
  const download=page.waitForEvent('download');await click('保存四图 PNG');const d=await download;await d.saveAs(path.join(out,d.suggestedFilename()));assert(d.suggestedFilename().endsWith('.png'));
  await page.screenshot({path:path.join(out,'inverse-result.png')});checks.push('large translucent IF charts and real PNG download');
  await click('返回分析');
  let release,received;const responseHeld=new Promise(resolve=>received=resolve);let armed=true;
  await page.route('**/__m03',async route=>{if(armed&&route.request().postDataJSON()?.op==='egg_preview_update'){armed=false;const response=await route.fetch();received();await new Promise(resolve=>release=resolve);await route.fulfill({response});}else await route.continue();});
  await fill('EGG 微观窗口',125);await responseHeld;
  await page.getByLabel('EGG 音频文件').selectOption({label:'3-复测.wav'});await page.waitForFunction(()=>document.querySelector('.egg-source select').selectedOptions[0].textContent==='3-复测.wav');release();await idle();
  assert((await page.locator('.audio-pane header').innerText()).includes('0.2500'));checks.push('file switch rejects controlled late previous response');
  await page.unroute('**/__m03');
  let expired=false;await page.route('**/__m03',async route=>{if(!expired&&route.request().postDataJSON()?.op==='egg_preview_update'){expired=true;await route.fulfill({contentType:'application/json',body:JSON.stringify({error:'egg_preview_expired'})});}else await route.continue();});
  await fill('EGG 微观窗口',100);await idle();assert(expired);checks.push('expired session transparently reloads current real source');await page.unroute('**/__m03');
  await fill('EGG 选区起点',40);await fill('EGG 选区时长',.5);await idle();await click('逆滤波 IF');await page.getByLabel('EGG 音频文件').selectOption({label:'3.wav'});await idle();
  await page.waitForFunction(()=>!document.querySelector('.egg-history').innerText.includes('运行中'),{},{timeout:60000});assert.equal(await page.locator('dialog').count(),0);checks.push('late IF completion after source switch stays in history');
  await page.evaluate(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.textContent==='交换声道');b.click();b.click();});await idle();assert.equal(await page.getByRole('button',{name:'播放选区',exact:true}).isEnabled(),true);checks.push('rapid channel round trip preserves normalized playback');
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(out,'r2-report.json'),JSON.stringify({checks,timings,errors},null,2));console.log(JSON.stringify({out,checks,timings},null,2));
 }catch(e){console.log(await page.locator('.egg-page').innerText());await page.screenshot({path:path.join(out,'failed.png'),fullPage:true});throw e;}finally{await browser.close();await server.close();worker.stdin.write(JSON.stringify({op:'shutdown',rpc_id:++counter})+'\n');await new Promise(resolve=>worker.once('exit',resolve));}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
