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
  await page.getByLabel('EGG 音频文件').selectOption({label:'EGG ɑ̃˥.wav'});await page.waitForFunction(()=>!document.querySelector('.egg-source select').disabled);await click('更新分析');await plotted();
  const main=page.locator('#main-content'),plot=page.locator('.cq-pane svg');
  const scroll=()=>main.evaluate(e=>e.scrollTop),reset=()=>main.evaluate(e=>e.scrollTop=0);
  const duration=await page.getByLabel('EGG 选区时长').inputValue(),jobs=await page.locator('.task-row').count();
  for(const [width,height,theme] of [[1280,800,'light'],[1024,600,'dark'],[800,500,'light'],[1440,450,'dark']]){
   await page.setViewportSize({width,height});await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);await reset();await plot.hover();const before=await scroll();await page.mouse.wheel(0,240);await page.waitForTimeout(250);assert((await scroll())>before,'ordinary wheel over EGG must scroll');
   assert.equal(await page.getByLabel('EGG 选区时长').inputValue(),duration);assert.equal(await page.locator('.task-row').count(),jobs);
   await main.hover({position:{x:15,y:100}});for(let i=0;i<8;i++)await page.mouse.wheel(0,600);await page.waitForTimeout(300);
   const box=await page.locator('.egg-overview').boundingBox();assert(box.y<height&&box.y+box.height>0,'overview reachable');await page.screenshot({path:path.join(out,`scroll-${width}-${height}-${theme}.png`)});checks.push({width,height,theme,scrollTop:await scroll(),ordinaryWheelNoTask:true});
  }
  await page.setViewportSize({width:1280,height:800});await reset();await plot.hover();await page.keyboard.down('Control');await page.mouse.wheel(0,-120);await page.keyboard.up('Control');await page.locator('.cq-pane .plot-empty').waitFor();await plotted();assert(Number(await page.getByLabel('EGG 选区时长').inputValue())<Number(duration));checks.push('Ctrl-wheel still zooms and recomputes real EGG result');
  await click('批量分析');const dialog=page.locator('dialog'),body=dialog.locator('.dialog-body'),footer=dialog.locator('.dialog-actions');await page.setViewportSize({width:800,height:500});
  assert(await footer.isVisible());const footerBox=await footer.boundingBox();assert(footerBox.y>=0&&footerBox.y+footerBox.height<=500);await body.hover();await page.mouse.wheel(0,600);await page.waitForTimeout(200);assert((await body.evaluate(e=>e.scrollTop))>0);const afterFooter=await footer.boundingBox();assert.equal(afterFooter.y,footerBox.y);await page.screenshot({path:path.join(out,'dialog-scroll.png')});checks.push('batch dialog body scrolls with footer fixed');await dialog.getByLabel('关闭对话框').click();
  for(const label of ['参数显示','语谱图转音频','参数估计']){await click(label);await main.hover({position:{x:15,y:100}});await page.mouse.wheel(0,1000);await page.waitForTimeout(200);assert.equal(await main.evaluate(e=>getComputedStyle(e).overflowY),'auto');checks.push(label+' outer scroll fallback available');}
  await page.goto(server.resolvedUrls.local[0]+'tests/m02-export.html');await page.getByRole('button',{name:'元音ɑ̃.wav',exact:true}).click();await page.locator('.parameter-chart').waitFor();const chart=page.locator('.parameter-chart').first();await chart.hover();const beforePath=await chart.innerHTML();const beforeY=await page.evaluate(()=>scrollY);await page.mouse.wheel(0,160);await page.waitForTimeout(200);assert((await page.evaluate(()=>scrollY))>beforeY);assert.equal(await chart.innerHTML(),beforePath);await chart.hover();await page.keyboard.down('Control');await page.mouse.wheel(0,-120);await page.keyboard.up('Control');await page.waitForTimeout(200);assert.notEqual(await chart.innerHTML(),beforePath);checks.push('M02 actual component synthetic table: ordinary wheel scrolls without chart change; Ctrl-wheel zooms');
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(out,'report.json'),JSON.stringify({success:true,checks,errors,schema_applied:[]},null,2));console.log(out);
 }catch(e){await page.screenshot({path:path.join(out,'failed.png')});await fs.writeFile(path.join(out,'report.json'),JSON.stringify({success:false,error:String(e),checks,errors},null,2));console.log(out);throw e;}
 finally{await browser.close();await server.close();worker.stdin.end(JSON.stringify({op:'shutdown'})+'\n');await new Promise(resolve=>worker.once('exit',resolve));}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
