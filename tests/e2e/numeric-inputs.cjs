// Audit numeric declarations and probe actual Vue controls using owned Chrome.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url'),{spawnSync}=require('node:child_process'),{createHash}=require('node:crypto');
const root=path.resolve(__dirname,'../..');
async function main(){
 const out=path.join(root,'output/validation/numeric-inputs','chrome-'+Date.now()),scratch=path.join(out,'scratch');await fs.mkdir(scratch,{recursive:true});
 const write=(name,data)=>fs.writeFile(path.join(out,name),JSON.stringify(data,null,2)+'\n');
 const marker={schema:'ptb-test-run/1',owner_pid:process.pid,state:'active',started_at:new Date().toISOString(),recipe:'Source input declaration inventory; 18 actual Vue views with synthetic state only; decimal typing and sibling render, F0 draft validation in light/dark'};await write('run.json',marker);
 const report={success:false,cases:[],errors:[],inventory:[]};let browser,server,page;
 try{
  const {parse}=require(path.join(root,'frontend/node_modules/@vue/compiler-dom'));
  async function inventory(dir){for(const entry of await fs.readdir(dir,{withFileTypes:true})){const file=path.join(dir,entry.name);if(entry.isDirectory())await inventory(file);else if(file.endsWith('.vue')){
   const text=await fs.readFile(file,'utf8'),template=text.match(/<template>([\s\S]*)<\/template>/)?.[1];if(!template)continue;
   const ast=parse(template);function visit(n){if(n.type===1&&n.tag==='input'){
    const prop=name=>n.props.find(p=>p.type===6&&p.name===name)?.value?.content;
    const model=n.props.find(p=>p.type===7&&p.name==='model');const unit=n.props.find(p=>p.type===7&&p.name==='time-precision');
    if(prop('type')!=='range'&&(prop('type')==='number'||prop('inputmode')==='decimal'||unit))report.inventory.push({file:path.relative(root,file).replaceAll('\\','/'),line:text.slice(0,text.indexOf('<template>')).split('\n').length+n.loc.start.line-1,type:prop('type')??'text',unit:unit?.exp?.content??null,model:model?.exp?.content??null,source:n.loc.source});
   }for(const child of n.children??[])visit(child);}visit(ast);
  }}}await inventory(path.join(root,'frontend/src'));
  assert.deepEqual(report.inventory.filter(c=>c.type==='number'&&!c.model),[],'Number controls must retain an editing model across unrelated renders');
  const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
  server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0,watch:null},optimizeDeps:{noDiscovery:true,include:['vue']}});await server.listen();
  const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
  browser=await chromium.launchPersistentContext(path.join(scratch,'profile'),{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,viewport:{width:1440,height:1000}});report.browser=browser.browser().version();page=await browser.newPage();page.on('pageerror',e=>report.errors.push(e.message));
  const views=process.env.PTB_INPUT_VIEWS?.split(',')??['settings','joint','egg','eggRange','eggBatch','transport','offset','font','background','m02','m04','m05','m06','m07','m08','m09','m11','m12','m15','m16','m17','m18','pdf'];
  for(const theme of ['light','dark'])for(const view of views){
   await page.goto(server.resolvedUrls.local[0]+'tests/numeric-inputs.html?view='+view);await page.waitForFunction(()=>window.qa?.ready);await page.evaluate(theme=>document.documentElement.dataset.theme=theme,theme);
   if(view==='m07'){
    for(const role of ['源','目标']){
     const input=page.getByLabel(`${role} F0 第 1 点`,{exact:true});await input.fill('');for(const key of '150.25'){await input.press(key==='.'?'Period':key);if(key==='.')await page.getByLabel('F0 对齐',{exact:true}).evaluate(el=>{el.value=el.value==='normalize'?'onset':'normalize';el.dispatchEvent(new Event('change',{bubbles:true}));});await page.evaluate(()=>qa.rerender());await page.evaluate(()=>new Promise(requestAnimationFrame));}assert.equal(await input.inputValue(),'150.25',theme+' '+role+' F0 decimal');
     await input.press('ArrowLeft');await input.press('Backspace');await input.press('7');assert.equal(await input.inputValue(),'150.75');await page.evaluate(()=>qa.rerender());await input.press('6');assert.equal(await input.inputValue(),'150.765');
    }
    await page.getByRole('button',{name:'保存参数草稿',exact:true}).click();assert.deepEqual(await page.evaluate(()=>{const d=qa.readDraft();return [d.points.source[0],d.points.target[0]];}),[150.765,150.765]);
    const source=page.getByLabel('源 F0 第 1 点',{exact:true});await source.fill('');await source.blur();await page.getByRole('button',{name:'保存参数草稿',exact:true}).click();assert.equal(await page.evaluate(()=>qa.readDraft().points.source[0]),0);
    await source.fill('abc');await page.getByRole('button',{name:'保存参数草稿',exact:true}).click();assert.equal(await page.evaluate(()=>qa.readDraft().points.source[0]),0);assert(await page.getByText('每份 F0 至少需要一个有效点，值须在 0–1000 Hz 内，空白表示无声',{exact:true}).isVisible());
    report.cases.push({theme,view,name:'F0 source/target decimals, middle caret, sibling update, numeric save, blank and invalid input'});await page.locator('.m07-workspace').screenshot({path:path.join(out,theme+'-f0.png')});await source.fill('150');
   }
   const fields=page.locator('input[type="number"]:visible:not(:disabled), input[inputmode="decimal"]:visible:not(:disabled)');const count=await fields.count();
   for(let i=0;i<count;i++){
    const field=fields.nth(i),original=await field.inputValue(),label=await field.evaluate(el=>el.getAttribute('aria-label')??el.closest('label')?.textContent.trim()??'numeric');
    await field.fill('');await field.press('1');await field.press('Period');await page.evaluate(()=>qa.rerender());await field.press('2');assert.equal(await field.inputValue(),'1.2',`${theme} ${view} ${label}`);
    await field.press('Backspace');await field.press('3');assert.equal(await field.inputValue(),'1.3',`${theme} ${view} ${label} backspace`);
    report.cases.push({theme,view,label,typed:'1.3'});await field.fill(original);await field.blur();
   }
   if(view==='background'){
    const blur=page.getByLabel('背景图高斯模糊值',{exact:true});await blur.fill('0');await blur.blur();await blur.fill('-10');await blur.blur();assert.equal(await blur.inputValue(),'0');assert.equal(await page.evaluate(()=>qa.model.blurInput),0);
    report.cases.push({theme,view,name:'blur rejected below minimum restores display even when model remains zero'});
   }
   if(view==='m17'){
    const size=page.getByLabel('音标字号',{exact:true});await size.fill('26');await size.blur();await size.fill('26.4');await size.blur();assert.equal(await size.inputValue(),'26');
    report.cases.push({theme,view,name:'font size retains integer rounding on commit'});
   }
   if(view==='transport'){
    const start=page.locator('.selection-controls input').first();await start.fill('');assert.equal(await page.evaluate(()=>qa.wave.start),0);await start.press('1');await start.press('Period');await start.press('2');await start.blur();assert.equal(await page.evaluate(()=>qa.wave.start),1.2);
    await start.focus();await page.evaluate(()=>qa.wave.start=2.3456789);await page.waitForFunction(()=>document.querySelector('.selection-controls input').value==='2.34568');assert.equal(await page.evaluate(()=>qa.wave.start),2.3456789);await start.blur();
    report.cases.push({theme,view,name:'blank text stays local until commit; genuine focused external changes preserve scientific precision'});
   }
   if(!count)report.cases.push({theme,view,skipped:'No enabled numeric controls in this empty view'});
  }
  // Native M10 duration is the only number text field outside Vue; sliders are not typed controls.
  await page.route('**/vocal-tract/app.js',route=>route.fulfill({contentType:'text/javascript',body:''}));
  for(const theme of ['light','dark']){
   await page.goto(server.resolvedUrls.local[0]+'vocal-tract/index.html');await page.evaluate(async theme=>{
    const {setupKeyframes}=await import('/vocal-tract/keyframes.js');const frames=[{name:'synthetic',duration:1,f0:150,lip_width:1,params:[],source:{mode:'voiced'}}];window.nativeFrames=frames;
    setupKeyframes({initialFrames:frames,getPose:()=>frames[0],applyPose:()=>{},post:async()=>new Response('{}'),setBusy:()=>{},isReady:()=>true,isSilent:true,keepVowel:()=>true,viewer:{}});document.getElementById('panel-motion').hidden=false;document.documentElement.dataset.theme=theme;
   },theme);
   const duration=page.locator('#keyframeList input[type="number"]');await duration.fill('');for(const key of '1.25')await duration.press(key==='.'?'Period':key);assert.equal(await duration.inputValue(),'1.25');await duration.blur();assert.equal(await page.evaluate(()=>nativeFrames[0].duration),1.25);
   await duration.fill('0');await duration.blur();assert.equal(await duration.inputValue(),'1.25');report.cases.push({theme,view:'native-m10',name:'per-key decimal and original duration range rejection'});
  }
  assert.deepEqual(report.errors,[]);report.success=true;
 }catch(error){report.failure=String(error);if(page)await page.screenshot({path:path.join(out,'failure.png')}).catch(()=>{});throw error;}
 finally{
  await browser?.close();await server?.close();await write('report.json',report);
  const files=[];async function inventory(dir){for(const entry of await fs.readdir(dir,{withFileTypes:true})){assert(!entry.isSymbolicLink());const file=path.join(dir,entry.name);if(entry.isDirectory())await inventory(file);else try{const bytes=await fs.readFile(file);files.push({path:path.relative(scratch,file),bytes:bytes.length,sha256:createHash('sha256').update(bytes).digest('hex')});}catch(e){if(e.code!=='ENOENT')throw e;}}}await inventory(scratch);
  await write('scratch-manifest.json',{schema:'ptb-test-scratch-files/1',files});await write('run.json',{...marker,state:'ready',test_outcome:report.success?'completed':'failed'});
  const cleanup=spawnSync('powershell.exe',['-NoProfile','-ExecutionPolicy','Bypass','-File',path.join(root,'scripts/cleanup_test_scratch.ps1'),'-RunDirectory',out],{windowsHide:true,encoding:'utf8'});if(cleanup.status!==0)await write('cleanup.json',{status:'retained',reason:cleanup.stderr||cleanup.stdout});
  console.log(JSON.stringify({out,success:report.success,declarations:report.inventory.length,cases:report.cases.length,failure:report.failure}));
 }
}
main().catch(error=>{console.error(error);process.exitCode=1;});
