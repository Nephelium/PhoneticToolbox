// M07-R1: actual production host on isolated synthetic files, muted WebAudio.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{spawn}=require('node:child_process'),{createInterface}=require('node:readline'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const worker=spawn(path.join(root,'.venv/m09-ui/Scripts/python.exe'),['-B','-X','utf8','tests/support/m07_host_bridge.py'],{cwd:root,windowsHide:true,env:{...process.env,PYTHONPATH:['backend/src','packages/phonetic_core/src','desktop/src','scripts'].map(p=>path.join(root,p)).join(';')}});
 let readyResolve,readyReject,counter=0;const pending=new Map(),ready=new Promise((r,j)=>{readyResolve=r;readyReject=j;});
 createInterface({input:worker.stdout}).on('line',line=>{const d=JSON.parse(line);if(d.ready)readyResolve(d);else{pending.get(d.id)?.(d);pending.delete(d.id);}});worker.stderr.on('data',d=>process.stderr.write(d));worker.once('exit',code=>readyReject(Error('Host exited '+code)));
 const {out}=await ready;console.log(out);
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0,watch:null},optimizeDeps:{entries:['tests/m07-host.html']},plugins:[{name:'m07-r1-host',configureServer(s){s.middlewares.use('/__m07_host',(req,res)=>{let raw='';req.on('data',d=>raw+=d);req.on('end',()=>{const id=++counter;pending.set(id,data=>{res.setHeader('Content-Type','application/json');res.end(JSON.stringify(data));});worker.stdin.write(JSON.stringify({...JSON.parse(raw),id})+'\n');});});}}]});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--mute-audio']}),page=await browser.newPage({viewport:{width:1920,height:1080}}),checks=[],layouts=[],errors=[];
 page.on('pageerror',e=>errors.push(e.message));page.setDefaultTimeout(60000);
 await page.addInitScript(()=>{window.audioStarts=[];const create=AudioContext.prototype.createBufferSource;AudioContext.prototype.createBufferSource=function(){const node=create.call(this),start=node.start;node.start=function(when,offset,duration){window.audioStarts.push({offset,duration,frames:this.buffer.length,sample:this.buffer.getChannelData(0)[10]});return start.call(this,when,offset,duration);};return node;};});
 const scope=page.getByLabel('发声类型连续统工作区',{exact:true});const click=name=>scope.getByRole('button',{name,exact:true}).click();
 const idle=()=>page.waitForFunction(()=>document.querySelector('.m07-page')?.getAttribute('aria-busy')==='false');
 const curves=n=>page.waitForFunction(n=>document.querySelectorAll('.f0-plot path[data-curve=synthesis]').length===n,n);
 const playing=()=>page.evaluate(async()=>{const {playback}=await import('/src/state/audio.ts');return playback.playing;});
 let success=false;
 try{
  await page.goto(server.resolvedUrls.local[0]+'tests/m07-host.html');await page.locator('nav').getByRole('button',{name:'发声类型合成',exact:true}).click();await scope.waitFor();
  assert.equal(await scope.locator('.wave-viewport').count(),3);await page.screenshot({path:path.join(out,'r1-empty.png')});
  await click('打开音频目录');await scope.getByRole('combobox',{name:'源音频',exact:true}).selectOption({label:'input0.wav'});await scope.getByRole('combobox',{name:'目标音频',exact:true}).selectOption({label:'input1.wav'});
  await click('提取 F0');await idle();await scope.getByText('分析完成，控制点尚未修改',{exact:true}).waitFor();
  assert((await scope.getByLabel('F0 图例',{exact:true}).innerText()).includes('源 F0'));assert((await scope.getByLabel('F0 图例',{exact:true}).innerText()).includes('目标 F0'));
  await scope.getByLabel('连续统步数',{exact:true}).fill('3');await click('生成全部六组');await idle();await curves(3);
  assert.equal(await scope.locator('.result-row').count(),1);assert.equal(await scope.locator('.history-select').filter({hasText:/ · 3 步/}).count(),6);
  assert((await scope.locator('.current-audio').innerText()).includes('整组'));assert.equal(await playing(),false);
  checks.push('six real groups persist; one selected result group, all-step default waveform without unsolicited playback, three permanent equal-format audio panels');
  const row=scope.locator('.result-row');let starts=await page.evaluate(()=>audioStarts.length);
  await row.getByRole('button',{name:'step02',exact:true}).click();await curves(1);await page.waitForFunction(n=>audioStarts.length===n,starts+1);
  assert((await scope.getByLabel('F0 图例',{exact:true}).innerText()).includes('step02'));assert.equal(await row.getByRole('button',{name:'step02',exact:true}).getAttribute('aria-pressed'),'true');
  const single=await page.evaluate(()=>audioStarts.at(-1));await scope.getByLabel('合成音频',{exact:true}).getByRole('button',{name:'停止',exact:true}).click();
  await row.getByRole('button',{name:'整组试听',exact:true}).click();await curves(3);const whole=await page.evaluate(()=>audioStarts.at(-1));assert.equal(whole.frames,single.frames*3);
  const axisEnds=await scope.locator('.f0-plot path[data-curve=synthesis]').evaluateAll(nodes=>nodes.map(n=>Number(n.dataset.axisEnd)));assert.deepEqual(axisEnds,[100,100,100]);
  checks.push('single click starts exact single-step WebAudio; whole click starts three-step PCM and overlays independent 0–100% curves');
  await scope.getByLabel('合成音频',{exact:true}).getByRole('button',{name:'停止',exact:true}).click();
  const history=scope.locator('.history-select');await history.filter({hasText:'源到目标 · 仅发声类型变化'}).click();await curves(3);assert.equal(await scope.locator('.result-row').count(),1);assert((await scope.locator('.result-row').innerText()).includes('源到目标 · 仅发声类型变化'));
  assert(await history.filter({hasText:/2026\/\d\d\/\d\d，\d\d:\d\d:\d\d/}).count()>0);
  const box=await scope.locator('.task-history').evaluate(e=>({height:e.clientHeight,scroll:e.scrollHeight}));assert(box.height<=185&&box.scroll>box.height);await scope.locator('.task-history').evaluate(e=>e.scrollTop=e.scrollHeight);assert(await scope.locator('.task-history').evaluate(e=>e.scrollTop)>0);
  await history.filter({hasText:'提取 F0'}).click();assert.equal(await scope.locator('.result-row').count(),0);assert.equal(await scope.locator('.f0-plot path[data-curve=synthesis]').count(),0);
  await history.filter({hasText:'目标到源 · F0 与发声类型同时变化'}).click();await curves(3);
  checks.push('scrollable compact history includes exact design and task time; generation task shows its one group, analysis has no synthesis group');
  const before=await scope.locator('.history-select[aria-pressed=true]').innerText();await click('刷新任务历史');await page.waitForFunction(()=>!document.querySelector('.history-heading button').disabled);assert.equal(await scope.locator('.history-select[aria-pressed=true]').innerText(),before);
  await scope.locator('.result-row').getByRole('button',{name:'step01',exact:true}).click();await scope.locator('.result-row').getByRole('button',{name:'step03',exact:true}).click();await curves(1);assert((await scope.locator('.current-audio').innerText()).includes('step03.wav'));assert((await scope.getByLabel('F0 图例',{exact:true}).innerText()).includes('step03'));
  await scope.getByLabel('合成音频',{exact:true}).getByRole('button',{name:'停止',exact:true}).click();
  checks.push('history refresh preserves selected task; quick step switches leave the latest waveform and F0 only');
  await click('输出位置');await click('保存完整组');await scope.getByText('已保存本组完整文件和参数清单',{exact:true}).waitFor();assert.equal((await fs.readdir(path.join(out,'saved'))).length,1);
  await scope.getByLabel('源 F0 第 4 点',{exact:true}).fill('NaN');assert(await scope.getByRole('button',{name:'生成当前',exact:true}).isDisabled());await click('应用编辑');await scope.getByText(/每份 F0 至少需要一个有效点/).waitFor();await scope.getByLabel('源 F0 第 4 点',{exact:true}).fill('130');await click('应用编辑');await idle();
  checks.push('native complete-group save and invalid-control rejection remain available');
  for(const [width,height] of [[1920,1080],[1440,900],[1280,800]])for(const theme of ['light','dark']){
   await page.setViewportSize({width,height});await page.evaluate(theme=>document.documentElement.dataset.theme=theme,theme);await page.waitForTimeout(120);
   const geo=await scope.locator('.m07-plots>.module-section').evaluateAll(nodes=>nodes.map(n=>{const r=n.getBoundingClientRect();return {label:n.getAttribute('aria-label'),x:r.x,y:r.y,width:r.width,height:r.height,bottom:r.bottom};}));
   assert(Math.abs(geo[0].y-geo[1].y)<2&&Math.abs(geo[2].y-geo[3].y)<2);assert(geo[0].x===geo[2].x&&geo[1].x===geo[3].x);
   if(width===1920)assert(geo.every(g=>g.bottom<=height),JSON.stringify(geo));
   await page.screenshot({path:path.join(out,`r1-${width}-${theme}.png`),fullPage:true});layouts.push({width,height,theme,geo});
  }
  await page.setViewportSize({width:1920,height:1080});await page.getByLabel('关闭 发声类型合成',{exact:true}).click();await page.getByRole('button',{name:'保存草稿并关闭',exact:true}).click();await scope.waitFor({state:'detached'});await page.locator('nav').getByRole('button',{name:'发声类型合成',exact:true}).click();await scope.waitFor();await click('刷新任务历史');await curves(3);assert.equal(await scope.locator('.result-row').count(),1);assert.equal(await scope.locator('.history-select').filter({hasText:/ · 3 步/}).count(),6);
  checks.push('reopen reads old managed results and their F0 snapshots; six light/dark layouts and four-panel geometry verified');
  await click('打开音频目录');await scope.getByRole('combobox',{name:'源音频',exact:true}).selectOption({label:'input0.wav'});await scope.getByRole('combobox',{name:'目标音频',exact:true}).selectOption({label:'input1.wav'});await click('提取 F0');await idle();await click('应用编辑');await idle();
  await scope.getByLabel('连续统步数',{exact:true}).fill('9');await scope.getByLabel('连续统类型',{exact:true}).selectOption('3');
  let delayed=false;await page.route('**/__m07_host',async route=>{const body=route.request().postDataJSON();if(body?.body?.op==='m07_f0'&&!delayed){delayed=true;const response=await route.fetch();await new Promise(resolve=>setTimeout(resolve,800));await route.fulfill({response});}else await route.continue();});
  await click('生成当前');await idle();await scope.locator('.history-select').filter({hasText:'源到目标 · 仅发声类型变化 · 3 步'}).click();await curves(3);await page.waitForTimeout(1000);assert.equal(await scope.locator('.f0-plot path[data-curve=synthesis]').count(),3);assert((await scope.locator('.result-row').innerText()).includes('仅发声类型变化 · 3 步'));
  await scope.locator('.history-select').filter({hasText:' · 9 步'}).click();await curves(9);assert.equal(await scope.locator('.result-audios button').count(),10);await scope.locator('.result-audios').getByRole('button',{name:'step09',exact:true}).click();await curves(1);assert((await scope.getByLabel('F0 图例',{exact:true}).innerText()).includes('step09'));await scope.locator('.result-audios').getByRole('button',{name:'整组试听',exact:true}).click();await curves(9);await scope.getByLabel('合成音频',{exact:true}).getByRole('button',{name:'停止',exact:true}).click();
  assert(delayed);await page.unroute('**/__m07_host');await page.evaluate(()=>document.documentElement.dataset.theme='light');await page.waitForTimeout(1000);await page.screenshot({path:path.join(out,'r1-nine-step-light.png')});checks.push('real nine-step group and single-step nine playback; injected late F0 response cannot replace the newly selected older task');

  assert.deepEqual(errors,[]);success=true;
 }catch(e){await fs.writeFile(path.join(out,'r1-failure.txt'),await scope.innerText());await page.screenshot({path:path.join(out,'r1-failure.png'),fullPage:true});throw e;}
 finally{await fs.writeFile(path.join(out,'r1-report.json'),JSON.stringify({success,checks,layouts,errors,scope:'Windows Chrome, actual production desktop adapter/host; QWebChannel transport substitute and muted WebAudio'},null,2));console.log(JSON.stringify({out,success,checks}));await browser.close();await server.close();worker.stdin.end();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
