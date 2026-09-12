// Real owned backend/child + common AppShell, standalone headless Chrome.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{spawn}=require('node:child_process'),{createInterface}=require('node:readline'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const worker=spawn(path.join(root,'.venv/m09-ui/Scripts/python.exe'),['-X','utf8','scripts/m03_ui_bridge.py','--long'],{cwd:root,windowsHide:true,env:{...process.env,PYTHONPATH:path.join(root,'backend/src')+';'+path.join(root,'desktop/src')}});
 const pending=new Map();let counter=0;let resolveReady,rejectReady;const ready=new Promise((resolve,reject)=>{resolveReady=resolve;rejectReady=reject;});
 createInterface({input:worker.stdout}).on('line',line=>{const v=JSON.parse(line);if(v.ready)resolveReady(v);else{const done=pending.get(v.id);pending.delete(v.id);done?.(v);}});worker.stderr.on('data',d=>process.stderr.write(d));worker.on('error',rejectReady);
 const {out}=await ready;const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0},plugins:[{name:'owned-m03-rpc',configureServer(s){s.middlewares.use('/__m03',(req,res)=>{let body='';req.on('data',d=>body+=d);req.on('end',()=>{const data=JSON.parse(body),id=++counter;pending.set(id,v=>{res.setHeader('Content-Type','application/json');res.end(JSON.stringify(v));});worker.stdin.write(JSON.stringify({...data,rpc_id:id})+'\n');});});}}]});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});const context=await browser.newContext({viewport:{width:1440,height:1000}}),page=await context.newPage();const checks=[],errors=[];page.on('pageerror',e=>errors.push(e.message));
 const click=name=>page.getByRole('button',{name,exact:true}).first().click();const plotted=()=>page.waitForFunction(()=>document.querySelectorAll('.egg-four-plots .scientific-plot>svg').length===4,{},{timeout:60000});
 try{
  await page.goto(server.resolvedUrls.local[0]+'tests/m03-live.html');await click('EGG 信号分析');await click('打开 WAV 目录');await page.getByLabel('EGG 音频文件').selectOption({label:'long.wav'});await page.waitForFunction(()=>!document.querySelector('.egg-source select').disabled);
  const overview=page.locator('.egg-overview'),both=overview.getByLabel('显示两个声道'),fit=overview.getByRole('button',{name:'适合窗口',exact:true});
  for(const [width,height,theme] of [[1440,900,'dark'],[800,600,'light']]){
   await page.setViewportSize({width,height});await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);await overview.scrollIntoViewIfNeeded();const graph=await overview.locator('.wave-track svg').boundingBox(),controls=await overview.locator('.overview-controls').boundingBox(),box=await overview.boundingBox(),fitBox=await fit.boundingBox(),bothBox=await both.boundingBox();assert(graph.y-box.y<15,'waveform starts at top');assert.equal(Math.round(graph.height),110);assert(controls.y>=graph.y+graph.height);assert(Math.abs(fitBox.y-bothBox.y)<15,'fit and checkbox share controls row');await overview.screenshot({path:path.join(out,`overview-${width}-${theme}.png`)});checks.push({width,theme,height:box.height,waveFirst:true,controlsTogether:true});
  }
  await both.check();assert.equal(await overview.locator('.wave-track svg').count(),2);await overview.getByLabel('放大波形').click();assert((await overview.locator('.mono').textContent()).includes('4'));await fit.click();assert((await overview.locator('.mono').textContent()).includes('2'));assert.equal(await overview.getByLabel('平移波形时间窗').inputValue(),'0');await overview.screenshot({path:path.join(out,'overview-stereo.png')});checks.push('both-channel toggle, zoom and fit retain 60-second overview with zero offset');
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(out,'report.json'),JSON.stringify({success:true,checks,errors,schema_applied:[]},null,2));console.log(out);
 }catch(e){await page.screenshot({path:path.join(out,'failed.png')});await fs.writeFile(path.join(out,'report.json'),JSON.stringify({success:false,error:String(e),checks,errors},null,2));console.log(out);throw e;}
 finally{await browser.close();await server.close();worker.stdin.end(JSON.stringify({op:'shutdown'})+'\n');await new Promise(resolve=>worker.once('exit',resolve));}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
