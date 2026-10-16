// Real Vue page/native keyboard controls; isolated Chrome profile and synthetic ports.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url'),{spawnSync}=require('node:child_process'),{createHash}=require('node:crypto');
const root=path.resolve(__dirname,'../..');
async function main(){
 const out=path.join(root,'output/validation/m09-decimal-input','chrome-'+Date.now()),scratch=path.join(out,'scratch');await fs.mkdir(scratch,{recursive:true});
 const write=(name,data)=>fs.writeFile(path.join(out,name),JSON.stringify(data,null,2)+'\n');
 const marker={schema:'ptb-test-run/1',owner_pid:process.pid,state:'active',started_at:new Date().toISOString(),recipe:'8x8 in-memory PNG; actual M09 Vue page; per-key decimal typing, text caret, precision and task snapshot in light/dark themes'};await write('run.json',marker);
 const report={success:false,cases:[],errors:[]};let browser,server,page;
 try{
  const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
  server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0,watch:null},optimizeDeps:{noDiscovery:true,include:['vue']}});await server.listen();
  const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
  browser=await chromium.launchPersistentContext(path.join(scratch,'profile'),{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,viewport:{width:1440,height:1000}});report.browser=browser.browser().version();page=await browser.newPage();page.on('pageerror',e=>report.errors.push(e.message));
  for(const theme of ['light','dark']){
   await page.goto(server.resolvedUrls.local[0]+'tests/m09-decimal-input.html');await page.getByLabel('语谱图图片',{exact:true}).selectOption('synthetic-image');
   await page.evaluate(theme=>document.documentElement.dataset.theme=theme,theme);
   for(const [name,text] of [['时间起点（s）','0.125'],['时间终点（s）','0.5102'],['窗长（ms）','10.25']]){
    const input=page.getByLabel(name,{exact:true});await input.fill('');let typed='';
    for(const key of text){typed+=key;await input.press(key==='.'?'Period':key);assert.equal(await input.inputValue(),typed.endsWith('.')?typed.slice(0,-1):typed,`${theme} ${name} key ${key}`);}
    await input.blur();assert.equal(await input.inputValue(),text);report.cases.push({theme,name,typed:text});
   }
   const end=page.getByLabel('时间终点（s）',{exact:true});await end.fill('1');await end.press('Period');
   // A sibling control forces the whole page to update while the decimal is incomplete.
   await page.getByLabel('迭代次数',{exact:true}).evaluate(el=>{el.value='33';el.dispatchEvent(new Event('input',{bubbles:true}));});
   await end.press('2');assert.equal(await end.inputValue(),'1.2');
   await end.press('Backspace');await end.press('3');assert.equal(await end.inputValue(),'1.3');
   await end.fill('');await end.press('Period');await end.press('5');assert.equal(await end.inputValue(),'.5');await end.blur();assert.equal(await end.inputValue(),'0.5');
   await end.fill('0.5102');await end.press('ArrowLeft');await end.press('Backspace');await end.press('9');assert.equal(await end.inputValue(),'0.5192');
   await end.fill('0.123456789');assert.equal(await end.inputValue(),'0.12346');await end.blur();
   const win=page.getByLabel('窗长（ms）',{exact:true});await win.fill('10.256');assert.equal(await win.inputValue(),'10.26');await win.blur();
   const db=page.getByLabel('最小 dB（白）',{exact:true});await db.fill('');await db.pressSequentially('-30.5');assert.equal(await db.inputValue(),'-30.5');await db.blur();
   await page.getByRole('button',{name:'开始重建',exact:true}).click();
   assert.deepEqual(await page.evaluate(()=>{const c=qa.submitted.at(-1);return [c.time_start,c.time_end,c.win_length_ms,c.min_db,c.n_iter];}),[.125,.12346,10.26,-30.5,33]);
   report.cases.push({theme,name:'editing and snapshot',checks:'sibling render, backspace, leading decimal, mid-value editing, precision, negative decimal and numeric submission'});
   const text=page.getByLabel('Shared text seconds',{exact:true});await text.fill('1.23');await text.press('ArrowLeft');await text.press('4');assert.equal(await text.inputValue(),'1.243');
   assert.equal(await text.evaluate(el=>el.selectionStart),4);await page.evaluate(()=>qa.state.tick++);await text.press('5');assert.equal(await text.inputValue(),'1.2453');
   await text.fill('');await text.pressSequentially('1.23000');assert.equal(await text.inputValue(),'1.23000');await text.blur();assert.equal(await text.inputValue(),'1.23');
   report.cases.push({theme,name:'shared text caret and trailing zeros'});await page.locator('.calibration').screenshot({path:path.join(out,theme+'.png')});
  }
  assert.deepEqual(report.errors,[]);report.success=true;
 }catch(error){report.failure=String(error);if(page)await page.screenshot({path:path.join(out,'failure.png')}).catch(()=>{});throw error;}
 finally{
  await browser?.close();await server?.close();await write('report.json',report);
  const files=[];async function inventory(dir){for(const entry of await fs.readdir(dir,{withFileTypes:true})){assert(!entry.isSymbolicLink());const file=path.join(dir,entry.name);if(entry.isDirectory())await inventory(file);else try{const bytes=await fs.readFile(file);files.push({path:path.relative(scratch,file),bytes:bytes.length,sha256:createHash('sha256').update(bytes).digest('hex')});}catch(e){if(e.code!=='ENOENT')throw e;}}}await inventory(scratch);
  await write('scratch-manifest.json',{schema:'ptb-test-scratch-files/1',files});await write('run.json',{...marker,state:'ready',test_outcome:report.success?'completed':'failed'});
  const cleanup=spawnSync('powershell.exe',['-NoProfile','-ExecutionPolicy','Bypass','-File',path.join(root,'scripts/cleanup_test_scratch.ps1'),'-RunDirectory',out],{windowsHide:true,encoding:'utf8'});if(cleanup.status!==0)await write('cleanup.json',{status:'retained',reason:cleanup.stderr||cleanup.stdout});
  console.log(JSON.stringify({out,success:report.success,cases:report.cases.length,failure:report.failure}));
 }
}
main().catch(error=>{console.error(error);process.exitCode=1;});
