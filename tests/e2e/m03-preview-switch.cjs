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
 let release,armed=false,started=false,settled=false,pass=false;
 try{
  await page.goto(server.resolvedUrls.local[0]+'tests/m03-live.html');await click('EGG 信号分析');await click('打开 WAV 目录');
  const select=page.getByLabel('EGG 音频文件');
  const load=async label=>{await select.selectOption({label});await page.waitForFunction(()=>!document.querySelector('.egg-source select').disabled);};
  const waitFor=async predicate=>{const end=Date.now()+60000;while(!predicate()&&Date.now()<end)await page.waitForTimeout(20);assert(predicate(),'controlled response reached expected phase');};
  await page.route('**/__m03',async route=>{const body=route.request().postDataJSON();if(body.op==='result'&&armed){armed=false;started=true;await new Promise(resolve=>release=resolve);if(pass)await route.continue();else await route.fulfill({json:{error:'asset_expired'}});settled=true;return;}await route.continue();});
  for(const succeeds of [false,true]){
   await load('EGG ɑ̃˥.wav');armed=true;started=false;settled=false;pass=succeeds;await click('更新分析');await waitFor(()=>started);
   await load('silence.wav');release();await waitFor(()=>settled);await page.waitForTimeout(400);
   assert.equal(await page.locator('.egg-page>p[role=alert]').count(),0,'old preview response must not put an error on the newly selected file');
   assert.equal(await page.locator('.egg-four-plots .scientific-plot>svg').count(),0,'old preview must not replace new file empty analysis');
   assert.equal(await select.locator('option:checked').innerText(),'silence.wav');
   assert(await page.getByRole('button',{name:'更新分析',exact:true}).isEnabled());
   checks.push(`new file survives old preview ${succeeds?'success':'failure'} with no stale plots/error`);
  }
  await load('EGG ɑ̃˥.wav');armed=true;started=false;settled=false;pass=false;await click('更新分析');await waitFor(()=>started);release();await waitFor(()=>settled);
  await page.locator('.egg-page>p[role=alert]').filter({hasText:'asset_expired'}).waitFor();checks.push('current file preview failure remains visible');
  await click('更新分析');await plotted();assert.equal(await page.locator('.egg-page>p[role=alert]').count(),0);checks.push('normal real analysis recovers after current failure');
  await page.locator('.egg-page').getByRole('button',{name:'使用说明',exact:true}).click();const help=page.getByRole('dialog',{name:'EGG 使用说明'});assert((await help.innerText()).includes('固定 3 ms'));await help.screenshot({path:path.join(out,'help-light.png')});await page.evaluate(()=>document.documentElement.dataset.theme='dark');await help.screenshot({path:path.join(out,'help-dark.png')});checks.push('existing help displays clarified IF behavior in both themes');
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(out,'report.json'),JSON.stringify({success:true,checks,errors,schema_applied:[],faults:'Only controlled delay/asset_expired on a real preview result read. Normal task and scientific output remain real.'},null,2));console.log(out);
 }catch(e){await page.screenshot({path:path.join(out,'failed.png')});await fs.writeFile(path.join(out,'report.json'),JSON.stringify({success:false,error:String(e),checks,errors},null,2));console.log(out);throw e;}
 finally{release?.();await browser.close();await server.close();worker.stdin.end(JSON.stringify({op:'shutdown'})+'\n');await new Promise(resolve=>worker.once('exit',resolve));}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
