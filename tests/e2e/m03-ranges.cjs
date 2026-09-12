// Real owned backend/child + common AppShell, standalone headless Chrome.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{spawn}=require('node:child_process'),{createInterface}=require('node:readline'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const worker=spawn(path.join(root,'.venv/m09-ui/Scripts/python.exe'),['-X','utf8','scripts/m03_ui_bridge.py','--ranges'],{cwd:root,windowsHide:true,env:{...process.env,PYTHONPATH:path.join(root,'backend/src')+';'+path.join(root,'desktop/src')}});
 const pending=new Map();let counter=0;let resolveReady,rejectReady;const ready=new Promise((resolve,reject)=>{resolveReady=resolve;rejectReady=reject;});
 createInterface({input:worker.stdout}).on('line',line=>{const v=JSON.parse(line);if(v.ready)resolveReady(v);else{const done=pending.get(v.id);pending.delete(v.id);done?.(v);}});worker.stderr.on('data',d=>process.stderr.write(d));worker.on('error',rejectReady);
 const {out}=await ready;const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0},plugins:[{name:'owned-m03-rpc',configureServer(s){s.middlewares.use('/__m03',(req,res)=>{let body='';req.on('data',d=>body+=d);req.on('end',()=>{const data=JSON.parse(body),id=++counter;pending.set(id,v=>{res.setHeader('Content-Type','application/json');res.end(JSON.stringify(v));});worker.stdin.write(JSON.stringify({...data,rpc_id:id})+'\n');});});}}]});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});const context=await browser.newContext({viewport:{width:1440,height:1000}}),page=await context.newPage();const checks=[],errors=[];page.on('pageerror',e=>errors.push(e.message));
 const click=name=>page.getByRole('button',{name,exact:true}).first().click();const plotted=()=>page.waitForFunction(()=>document.querySelectorAll('.egg-four-plots .scientific-plot>svg').length===4,{},{timeout:60000});
 try{
  await page.goto(server.resolvedUrls.local[0]+'tests/m03-live.html');await click('EGG 信号分析');await click('打开 WAV 目录');
  const select=async name=>{await page.getByLabel('EGG 音频文件').selectOption({label:name});await page.waitForFunction(()=>!document.querySelector('.egg-source select').disabled);};
  const width=page.getByLabel('EGG 微观窗口');await select('wide.wav');
  await page.getByLabel('EGG 选区起点').fill('3');await page.getByLabel('EGG 选区起点').blur();
  for(const value of ['5','5000']){await width.fill(value);await width.blur();const start=Date.now();await click('更新分析');await plotted();checks.push({width_ms:Number(value),ready_ms:Date.now()-start});assert.equal(await width.inputValue(),value);await page.screenshot({path:path.join(out,'micro-'+value+'.png')});}
  assert((await page.locator('.audio-pane small').textContent()).includes('每 32 点'));
  for(const pathNode of await page.locator('.audio-pane svg path,.egg-pane svg path').all())assert((await pathNode.getAttribute('d')??'').length<300000);
  const beforeJobs=await page.locator('.task-row').count();await page.locator('.egg-pane svg').focus();await page.keyboard.press('-');await page.waitForTimeout(600);assert.equal(await width.inputValue(),'5000');assert.equal(await page.locator('.task-row').count(),beforeJobs);
  await page.keyboard.press('+');await page.locator('.egg-pane .plot-empty').waitFor();await plotted();assert.equal(await width.inputValue(),'4500');checks.push('wide waveform stays bounded; keyboard respects maximum then zooms inward');
  await width.fill('4');await width.blur();await click('更新分析');await page.getByRole('alert').filter({hasText:'微观窗口须在'}).waitFor();checks.push('invalid numeric range is explained before task submission');
  await width.fill('50');await width.blur();await select('long.wav');
  const pan=page.getByLabel('平移波形时间窗');await pan.focus();await page.keyboard.press('End');assert(Math.abs(Number(await pan.inputValue())-6.4)<.001);
  assert((await page.getByLabel('波形时间轴（秒）').textContent()).includes('66.400'));await page.getByLabel('EGG 选区起点').fill('65');await page.getByLabel('EGG 选区起点').blur();
  await click('更新分析');await plotted();assert.equal(await page.getByLabel('EGG 选区起点').inputValue(),'65');await page.screenshot({path:path.join(out,'long-eof.png')});checks.push('66.4-second source navigates to EOF and analyzes tail with global preprocessing');
  await select('oversized.wav');const count=await page.locator('.task-row').count();await click('更新分析');await page.getByRole('alert').filter({hasText:'当前 EGG 计算限 120 秒'}).waitFor();assert.equal(await page.locator('.task-row').count(),count);checks.push('120.8-second input explicitly refused without a new task');
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(out,'report.json'),JSON.stringify({success:true,checks,errors,schema_applied:[]},null,2));console.log(out);
 }catch(e){await page.screenshot({path:path.join(out,'failed.png')});await fs.writeFile(path.join(out,'report.json'),JSON.stringify({success:false,error:String(e),checks,errors},null,2));console.log(out);throw e;}
 finally{await browser.close();await server.close();worker.stdin.end(JSON.stringify({op:'shutdown'})+'\n');await new Promise(resolve=>worker.once('exit',resolve));}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
