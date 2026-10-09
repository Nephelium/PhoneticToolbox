// M08-R2: owned Chrome, real desktop adapter and Praat, isolated synthetic data.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{spawn}=require('node:child_process'),{createInterface}=require('node:readline'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const worker=spawn(path.join(root,'.venv/m09-ui/Scripts/python.exe'),['-X','utf8','tests/support/m08_host_bridge.py'],{cwd:root,windowsHide:true,env:{...process.env,PYTHONPATH:['backend/src','packages/phonetic_core/src','desktop/src','scripts'].map(p=>path.join(root,p)).join(';')}});
 let readyResolve,readyReject,counter=0;const pending=new Map(),ready=new Promise((r,j)=>{readyResolve=r;readyReject=j});
 createInterface({input:worker.stdout}).on('line',line=>{const d=JSON.parse(line);if(d.ready)readyResolve(d);else{pending.get(d.id)?.(d);pending.delete(d.id)}});worker.stderr.on('data',d=>process.stderr.write(d));worker.once('exit',code=>readyReject(Error('Host exited '+code)));
 const {out}=await ready;console.log(out);const requests=[];
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0,watch:null},optimizeDeps:{entries:['tests/m08-host.html']},plugins:[{name:'m08-r2-host',configureServer(s){s.middlewares.use('/__m08_host',(req,res)=>{let raw='';req.on('data',d=>raw+=d);req.on('end',()=>{const id=++counter,message=JSON.parse(raw);requests.push(message);pending.set(id,data=>{res.setHeader('Content-Type','application/json');res.end(JSON.stringify(data))});worker.stdin.write(JSON.stringify({...message,id})+'\n')})})}}]});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--mute-audio']}),page=await browser.newPage({viewport:{width:1800,height:1100}}),checks=[],layouts=[],errors=[];
 page.on('pageerror',e=>errors.push(e.message));page.setDefaultTimeout(45000);
 const click=name=>page.getByRole('button',{name,exact:true}).click(),idle=()=>page.waitForFunction(()=>document.querySelector('.m08-page')?.getAttribute('aria-busy')==='false');
 async function decimal(label,initial,tail){const e=page.getByLabel(label,{exact:true});await e.fill(initial);await e.press('End');for(const char of tail){await e.press(char==='.'?'Period':char);await page.waitForTimeout(50);}await page.waitForTimeout(1100);assert.equal(await e.inputValue(),initial+tail,label);}
 async function layout(width,height,theme,empty=false){
  await page.setViewportSize({width,height});await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);await page.waitForTimeout(350);
  const value=await page.locator('.plots').evaluate(e=>{const boxes=[...e.children].map(n=>{const r=n.getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height};});const m=document.querySelector('.m08-page');return {boxes,client:m.clientWidth,scroll:m.scrollWidth};});
  assert(value.scroll<=value.client+2,JSON.stringify(value));
  if(width>=1600){const b=value.boxes;assert(Math.abs(b[0].h-b[1].h)<1&&Math.abs(b[2].h-b[3].h)<1);assert(Math.abs(b[0].w-b[3].w)<1);}
  const chart=page.locator('[aria-label="历史 F0 对比"]');assert.equal(await chart.locator('svg.history-plot').count(),empty?0:1);
  if(empty)await chart.getByText('生成音频后显示实际 F0 对比。',{exact:true}).waitFor();
  await page.screenshot({path:path.join(out,`m08-r2-${empty?'empty':'results'}-${width}-${theme}.png`)});layouts.push({width,height,theme,empty,...value});
 }
 let success=false;
 try{
  await page.goto(server.resolvedUrls.local[0]+'tests/m08-host.html');await page.locator('nav').getByRole('button',{name:'变速变调',exact:true}).click();await click('打开音频目录');
  const select=page.getByLabel('M08 音频');await page.waitForFunction(()=>document.querySelector('[aria-label="M08 音频"]')?.options.length>1);
  const option=await select.locator('option').evaluateAll(es=>es.find(o=>o.textContent.endsWith('.wav')).value);await select.selectOption(option);await idle();await page.locator('svg.m08-curve').waitFor();
  await click('编辑拐点表');await click('添加行');await decimal('时间 1','0','.3');await click('取消');checks.push('incremental decimal time survives reactive redraw and job polling');
  await layout(1800,1100,'light',true);await layout(1800,1100,'dark',true);checks.push('empty history uses shared placeholder and equal outer panels');
  await decimal('语速倍率','0','.8');await decimal('F0 下限','50','.5');await decimal('F0 上限','350','.5');await click('应用范围');await decimal('参考线 Hz','200','.25');await click('添加参考线');checks.push('speed, axes and reference line retain partial decimals');
  await click('批量变速变调');await decimal('音高倍率','1','.25');await decimal('音高偏移 Hz','-2','.5');await click('单文件与基频');
  await click('合成当前视野');await page.getByText('输出 1.250 s',{exact:false}).waitFor();await page.waitForFunction(()=>document.querySelectorAll('.history li').length===1);await idle();
  await page.getByText('生成结果已保留在任务历史；点击保存合成音或批量保存，选择输出目录。',{exact:true}).waitFor();assert.equal((await fs.readdir(path.join(out,'saved'))).length,0);checks.push('generation explicitly distinguishes task history from chosen WAV export');
  await click('编辑拐点表');await click('添加行');await page.getByLabel('时间 0',{exact:true}).fill('0.1');await decimal('时间 1','0','.3');await page.getByLabel('时间 2',{exact:true}).fill('0.9');
  for(let i=0;i<3;i++){await page.getByLabel('频率 '+i,{exact:true}).fill('120,160,200');await page.getByLabel('连接 '+i,{exact:true}).selectOption('full');}
  await click('保存更改');await click('批量生成');await page.waitForFunction(()=>document.querySelectorAll('.history li').length===28);await idle();
  const linear=requests.findLast(r=>r.body.action==='create'&&r.body.body.config.action==='linear');assert.deepEqual(linear.body.body.config.points.map(p=>p.time),[.1,.3,.9]);checks.push('three decimal knot times reach real Praat task unchanged');
  await page.locator('.wave-viewport').first().getByRole('button',{name:'适合窗口',exact:true}).click();
  await click('删除本批次音频');
  const boxes=await page.getByRole('dialog').locator('.file-list input[type=checkbox]').evaluateAll(es=>es.map(e=>{const r=e.getBoundingClientRect();return {w:r.width,h:r.height};}));
  assert.equal(boxes.length,28);for(const b of boxes)assert(b.w===16&&b.h===16,JSON.stringify(boxes));await page.screenshot({path:path.join(out,'m08-r2-delete-checkboxes.png')});await click('取消');checks.push('28 native checkboxes keep 16 x 16 geometry independent of file-name length');
  // Single export carries the actual resolved native directory and final file name.
  await click('保存合成音');await idle();await page.locator('.save-location').getByText(path.join(out,'saved'),{exact:true}).waitFor();
  const saved=await fs.readdir(path.join(out,'saved'));assert.equal(saved.length,1);await page.locator('.save-location summary').click();await page.locator('.save-location').getByText(saved[0],{exact:true}).waitFor();checks.push('actual exported directory and collision-resolved file name remain visible');
  for(const [width,height]of [[1800,1100],[1280,800],[900,760]])for(const theme of ['light','dark'])await layout(width,height,theme);
  await page.setViewportSize({width:1800,height:1100});
  const download=page.waitForEvent('download');await click('保存对比图');await(await download).saveAs(path.join(out,'m08-r2-comparison.png'));assert((await fs.stat(path.join(out,'m08-r2-comparison.png'))).size>1000);checks.push('history chart PNG still exports');
  assert.deepEqual(errors,[]);success=true;
 }catch(e){await page.screenshot({path:path.join(out,'m08-r2-failed.png')});throw e;}
 finally{await fs.writeFile(path.join(out,'m08-r2-report.json'),JSON.stringify({success,checks,layouts,errors,requests},null,2));await browser.close();await server.close();worker.stdin.end();console.log(JSON.stringify({out,success,checks}));}
}
main().catch(e=>{console.error(e);process.exitCode=1});
