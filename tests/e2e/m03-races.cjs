// Real owned backend/child + common AppShell, standalone headless Chrome.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{spawn}=require('node:child_process'),{createInterface}=require('node:readline'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const worker=spawn(path.join(root,'.venv/m09-ui/Scripts/python.exe'),['-X','utf8','scripts/m03_ui_bridge.py'],{cwd:root,windowsHide:true,env:{...process.env,PYTHONPATH:path.join(root,'backend/src')+';'+path.join(root,'desktop/src')}});
 const pending=new Map();let counter=0;let resolveReady,rejectReady;const ready=new Promise((resolve,reject)=>{resolveReady=resolve;rejectReady=reject;});
 createInterface({input:worker.stdout}).on('line',line=>{const v=JSON.parse(line);if(v.ready)resolveReady(v);else{const done=pending.get(v.id);pending.delete(v.id);done?.(v);}});worker.stderr.on('data',d=>process.stderr.write(d));worker.on('error',rejectReady);
 const {out}=await ready;const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0},plugins:[{name:'owned-m03-rpc',configureServer(s){s.middlewares.use('/__m03',(req,res)=>{let body='';req.on('data',d=>body+=d);req.on('end',()=>{const data=JSON.parse(body),id=++counter;pending.set(id,v=>{res.setHeader('Content-Type','application/json');res.end(JSON.stringify(v));});worker.stdin.write(JSON.stringify({...data,rpc_id:id})+'\n');});});}}]});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});const context=await browser.newContext({viewport:{width:1440,height:1000}}),page=await context.newPage();const checks=[],errors=[];page.on('pageerror',e=>errors.push(e.message));
 const click=name=>page.getByRole('button',{name,exact:true}).first().click();const plotted=()=>page.waitForFunction(()=>document.querySelectorAll('.egg-four-plots .scientific-plot>svg').length===4,{},{timeout:60000});
 try{
  await page.goto(server.resolvedUrls.local[0]+'tests/m03-live.html');await click('EGG 信号分析');await click('打开 WAV 目录');await page.getByLabel('EGG 音频文件').selectOption({label:'EGG ɑ̃˥.wav'});await page.waitForFunction(()=>!document.querySelector('.egg-source select').disabled);await click('更新分析');await plotted();
  let release,arrived;const held=new Promise(r=>arrived=r),gate=new Promise(r=>release=r);let delay=true;
  await page.route('**/__m03',async route=>{const data=route.request().postDataJSON();if(delay&&data.op==='result'){delay=false;const response=await route.fetch();arrived();await gate;await route.fulfill({response});}else await route.continue();});
  await page.locator('.egg-result-links button').filter({hasText:'交互分析'}).first().click();await held;
  await page.getByLabel('EGG 音频文件').selectOption({label:'silence.wav'});await page.waitForFunction(()=>!document.querySelector('.egg-source select').disabled);release();await page.waitForTimeout(900);
  assert.equal(await page.getByLabel('EGG 音频文件').locator('option:checked').textContent(),'silence.wav');assert.equal(await page.locator('.egg-four-plots svg').count(),0);checks.push('held old preview metadata cannot replace the newly selected file');
  await page.unroute('**/__m03');await click('更新分析');await plotted();await click('交换声道');assert.equal(await page.locator('.egg-four-plots svg').count(),0);assert(await page.getByRole('button',{name:'播放选区',exact:true}).isDisabled());await click('更新分析');await plotted();checks.push('channel swap invalidates plots and playback until recomputed');
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(out,'report.json'),JSON.stringify({success:true,checks,errors,schema_applied:[]},null,2));console.log(out);
 }catch(e){await page.screenshot({path:path.join(out,'failed.png')});await fs.writeFile(path.join(out,'report.json'),JSON.stringify({success:false,error:String(e),checks,errors},null,2));console.log(out);throw e;}
 finally{await browser.close();await server.close();worker.stdin.end(JSON.stringify({op:'shutdown'})+'\n');await new Promise(resolve=>worker.once('exit',resolve));}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
