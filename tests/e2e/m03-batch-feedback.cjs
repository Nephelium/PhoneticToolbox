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
  await page.goto(server.resolvedUrls.local[0]+'tests/m03-live.html');await click('EGG 信号分析');await click('打开 WAV 目录');
  const batch=page.getByRole('dialog',{name:'EGG 批量分析'});
  const openBatch=async()=>{await click('批量分析');await batch.getByRole('checkbox',{name:/全选/}).uncheck();await batch.getByRole('checkbox',{name:'EGG ɑ̃˥.wav',exact:true}).check();};
  let submissions=0,fontCalls=0,rejectMode='',release;
  await page.route('**/__m03',async route=>{const body=route.request().postDataJSON();if(body.op==='egg_fonts')fontCalls++;if(body.op==='egg'){
    submissions++;
    if(rejectMode==='hold')await new Promise(resolve=>release=resolve);
    if(rejectMode==='all'||rejectMode==='hold'||(rejectMode==='partial'&&submissions%2===0)){await route.fulfill({json:{error:'input_unavailable'}});return;}
  }await route.continue();});
  await openBatch();await batch.getByLabel('高通 Hz').fill('1000');await click('提交所选文件');
  await batch.getByRole('alert').filter({hasText:'高通'}).waitFor({timeout:5000});assert.equal(submissions,0);assert.equal(fontCalls,0);
  assert(await batch.getByRole('checkbox',{name:'EGG ɑ̃˥.wav',exact:true}).isChecked());
  await batch.getByLabel('高通 Hz').fill('25');await batch.getByLabel('静音阈值').fill('');await click('提交所选文件');await batch.getByRole('alert').filter({hasText:'静音阈值'}).waitFor();assert.equal(submissions,0);
  await batch.getByLabel('静音阈值').fill('1.1');await click('提交所选文件');await batch.getByRole('alert').filter({hasText:'静音阈值'}).waitFor();assert.equal(submissions,0);
  checks.push('invalid filter, empty and out-of-range silence threshold stay in dialog without any task/font RPC');
  await batch.getByLabel('静音阈值').fill('.01');await click('提交所选文件');await batch.waitFor({state:'detached'});
  const saved=page.getByRole('button',{name:'保存本次批量结果（1）',exact:true});await saved.waitFor({timeout:60000});assert.equal(submissions,1);
  checks.push('corrected parameters produce one real successful batch result');
  await openBatch();rejectMode='all';await click('提交所选文件');await batch.getByRole('alert').filter({hasText:'源文件已失效'}).waitFor();assert(await saved.count());assert(await batch.getByRole('checkbox',{name:'EGG ɑ̃˥.wav',exact:true}).isChecked());
  checks.push('controlled all-file admission rejection preserves prior batch save entry and selected files');
  rejectMode='hold';await click('提交所选文件');await page.waitForTimeout(250);assert(await batch.getByLabel('高通 Hz').isDisabled());assert(await batch.getByLabel('静音阈值').isDisabled());assert(await batch.getByRole('button',{name:'返回工作台'}).isDisabled());assert(await batch.getByRole('button',{name:'关闭对话框'}).isDisabled());await page.keyboard.press('Escape');assert(await batch.isVisible());release();await batch.getByRole('alert').filter({hasText:'源文件已失效'}).waitFor();
  await page.setViewportSize({width:800,height:600});await batch.screenshot({path:path.join(out,'batch-error-small.png')});const submit=await batch.getByRole('button',{name:'提交所选文件'}).boundingBox();assert(submit.y>=0&&submit.y+submit.height<=600);await batch.getByRole('alert').scrollIntoViewIfNeeded();assert(await batch.getByRole('alert').isVisible());checks.push('controlled pending rejection freezes parameters and close action; error and footer remain reachable at 800x600');
  await page.setViewportSize({width:800,height:440});const body=batch.locator('.dialog-body');assert(await body.evaluate(e=>e.scrollHeight>e.clientHeight));await body.evaluate(e=>e.scrollTop=0);await body.hover();await page.mouse.wheel(0,350);await page.waitForFunction(()=>document.querySelector('dialog .dialog-body').scrollTop>0);const smallSubmit=await batch.getByRole('button',{name:'提交所选文件'}).boundingBox();assert(smallSubmit.y+smallSubmit.height<=440);await page.evaluate(()=>document.documentElement.dataset.theme='dark');await batch.getByRole('alert').scrollIntoViewIfNeeded();await batch.screenshot({path:path.join(out,'batch-error-dark-scroll.png')});checks.push('ordinary wheel scrolls dialog content at 800x440 with accessible fixed footer in dark theme');
  await click('返回工作台');await page.setViewportSize({width:1440,height:1000});await openBatch();await batch.getByRole('checkbox',{name:'silence.wav',exact:true}).check();rejectMode='partial';submissions=0;await click('提交所选文件');await batch.waitFor({state:'detached'});await page.getByRole('status').filter({hasText:'未提交'}).filter({hasText:'源文件已失效'}).waitFor();await saved.waitFor({timeout:60000});assert.equal(submissions,2);await page.screenshot({path:path.join(out,'batch-partial.png')});
  checks.push('partial controlled rejection submits each selected file once and retains the real successful result with per-file failure reason');
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(out,'report.json'),JSON.stringify({success:true,checks,errors,schema_applied:[],faults:'Submission rejection and delay injected at browser RPC only; successful jobs use actual backend and science child.'},null,2));console.log(out);
 }catch(e){await page.screenshot({path:path.join(out,'failed.png')});await fs.writeFile(path.join(out,'report.json'),JSON.stringify({success:false,error:String(e),checks,errors},null,2));console.log(out);throw e;}
 finally{await browser.close();await server.close();worker.stdin.end(JSON.stringify({op:'shutdown'})+'\n');await new Promise(resolve=>worker.once('exit',resolve));}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
