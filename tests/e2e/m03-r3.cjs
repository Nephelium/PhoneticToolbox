// Real owned backend/child + common AppShell, standalone headless Chrome.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{spawn}=require('node:child_process'),{createInterface}=require('node:readline'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const worker=spawn(path.join(root,'.venv/m09-ui/Scripts/python.exe'),['-X','utf8','scripts/m03_ui_bridge.py','--real-only'],{cwd:root,windowsHide:true,env:{...process.env,PYTHONPATH:path.join(root,'backend/src')+';'+path.join(root,'desktop/src')+';'+path.join(root,'packages/phonetic_core/src')}});
 const pending=new Map();let counter=0;let resolveReady,rejectReady;const ready=new Promise((resolve,reject)=>{resolveReady=resolve;rejectReady=reject;});
 createInterface({input:worker.stdout}).on('line',line=>{const v=JSON.parse(line);if(v.ready)resolveReady(v);else{const done=pending.get(v.id);pending.delete(v.id);done?.(v);}});worker.stderr.on('data',d=>process.stderr.write(d));worker.on('error',rejectReady);worker.on('exit',code=>rejectReady(Error('Bridge exited before ready: '+code)));
 const {out}=await ready;const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0},optimizeDeps:{noDiscovery:true,include:['vue']},plugins:[{name:'owned-m03-rpc',configureServer(s){s.middlewares.use('/__m03',(req,res)=>{let body='';req.on('data',d=>body+=d);req.on('end',()=>{const data=JSON.parse(body),id=++counter;pending.set(id,v=>{res.setHeader('Content-Type','application/json');res.end(JSON.stringify(v));});worker.stdin.write(JSON.stringify({...data,rpc_id:id})+'\n');});});}}]});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});const context=await browser.newContext({viewport:{width:1440,height:1000}}),page=await context.newPage();const checks=[],errors=[],timings=[];const resultReads=[];page.on('request',req=>{if(req.url().endsWith('/__m03')){const b=req.postDataJSON();if(b?.op==='result')resultReads.push(b);}});page.on('pageerror',e=>errors.push(e.message));
 const click=name=>page.getByRole('button',{name,exact:true}).first().click();const plotted=()=>page.waitForFunction(()=>document.querySelectorAll('.egg-four-plots .scientific-plot>svg').length===4,{},{timeout:60000});



 try{
  const idle=()=>page.waitForFunction(()=>document.querySelector('.egg-live-status')?.textContent==='实时预览'&&[...document.querySelectorAll('button')].some(b=>b.textContent==='保存 CSV / 三图'&&!b.disabled),{},{timeout:60000});
  const fill=async(label,value)=>{await page.getByLabel(label).fill(String(value));await page.getByLabel(label).press('Tab');};
  await page.goto(server.resolvedUrls.local[0]+'tests/m03-r2.html');await click('EGG 信号分析');await click('打开 WAV 目录');
  await page.getByLabel('EGG 音频文件').selectOption({label:'3.wav'});await idle();await plotted();
  assert.equal(await page.locator('.workbench-left').getByLabel('EGG 音频文件').count(),1);
  for(const name of ['交换声道','保存 CSV / 三图','逆滤波 IF'])assert.equal(await page.locator('.module-toolbar').getByRole('button',{name,exact:true}).count(),1);
  assert.equal(await page.locator('.module-toolbar').getByLabel('EGG LP 阶数').count(),1);checks.push('selector in sidebar, all requested actions in top toolbar');
  const geometry=[];
  for(const mode of ['light','dark'])for(const width of [230,300,420,520]){
    await page.evaluate(({mode,width})=>{document.documentElement.dataset.theme=mode;const h=document.querySelector('.module-workbench .panel-resize-handle');h.dispatchEvent(new KeyboardEvent('keydown',{key:'Home',bubbles:true}));for(let i=200;i<width;i+=10)h.dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true}));}, {mode,width});
    await page.waitForTimeout(120);
    const g=await page.locator('.workbench-left').evaluate(el=>({width:el.clientWidth,scroll:el.scrollWidth,height:el.querySelector('.egg-side-controls').getBoundingClientRect().height,fields:[...el.querySelectorAll('input[type=number],select')].map(e=>({w:e.getBoundingClientRect().width,h:e.getBoundingClientRect().height,x:e.getBoundingClientRect().x})),cols:[...el.querySelectorAll('.egg-field-grid')].map(e=>getComputedStyle(e).gridTemplateColumns)}));
    assert(g.scroll<=g.width+1,JSON.stringify(g));assert(g.fields.every(f=>f.h<=30),JSON.stringify(g));geometry.push({mode,width,...g});
    await page.screenshot({path:path.join(out,`r3-sidebar-${mode}-${width}.png`)});
  }
  assert(geometry.find(g=>g.width>400).cols.some(c=>c.split(' ').length===2));checks.push('8 sidebar layout cases, compact heights, narrow/wide field pairs, no horizontal overflow');
  await fill('EGG 选区起点',40);await fill('EGG 选区时长',.5);await idle();
  await click('自动 dB');await idle();const db=[+await page.getByLabel('EGG dB 下限').inputValue(),+await page.getByLabel('EGG dB 上限').inputValue()];assert.equal(db[1]-db[0],50);
  await fill('EGG dB 下限',db[0]-7);await idle();assert.equal(+await page.getByLabel('EGG dB 下限').inputValue(),db[0]-7);
  await fill('EGG 微观窗口',100);await idle();assert.equal(+await page.getByLabel('EGG dB 下限').inputValue(),db[0]-7);checks.push('automatic actual-PSD 50 dB range, manual override persists through subsequent updates');
  await fill('EGG dB 下限',10);await click('自动 dB');await idle();assert.equal(+await page.getByLabel('EGG dB 下限').inputValue(),db[0]);checks.push('auto dB repairs an invalid manual range using matching PSD');
  await page.getByLabel('GCI F0',{exact:true}).check();await idle();
  const f0Axes=await page.locator('.spec-pane .scientific-plot>svg').evaluate(svg=>{const frame=svg.querySelector(':scope>rect'),x=+frame.getAttribute('x')+(+frame.getAttribute('width'))+7;return [...svg.querySelectorAll(':scope>text')].filter(t=>Math.abs(+t.getAttribute('x')-x)<.01).map(t=>+t.textContent);});
  assert.equal(f0Axes.length,5);assert.notDeepEqual(f0Axes,[50,163,275,388,500]);assert(f0Axes[0]>300&&f0Axes[4]>f0Axes[0]);checks.push('real GCI F0 axis follows current voiced range instead of 50-500 Hz');
  await page.evaluate(()=>{const h=document.querySelector('.module-workbench .panel-resize-handle');h.dispatchEvent(new KeyboardEvent('keydown',{key:'Home',bubbles:true}));for(let i=0;i<10;i++)h.dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true}));});
  await click('逆滤波 IF');await page.locator('.inverse-grid .scientific-plot>svg').first().waitFor({timeout:60000});
  assert.equal(await page.locator('.inverse-actions').getByRole('button',{name:'选择目录保存完整结果'}).count(),1);assert.equal(await page.locator('.inverse-actions').getByRole('button',{name:'返回分析'}).count(),1);
  assert.equal(await page.locator('dialog .dialog-actions').count(),0);checks.push('result save/return in top row, no bottom footer');
  const outputs=[];
  for(const [name,count] of [['全部分开 · 6 图',6],['音频 + IF · 4 图',4],['EGG + IF · 4 图',4],['音频 + EGG + IF · 2 图',2]]){
    await click(name);await page.waitForTimeout(160);assert.equal(await page.locator('.inverse-grid .scientific-plot>svg').count(),count);
    const bounds=await page.locator('.inverse-grid .scientific-plot>svg').evaluateAll(svgs=>svgs.map(svg=>{const r=svg.querySelector(':scope>rect');return [+r.getAttribute('x'),+r.getAttribute('width')];}));
    for(const bound of bounds)assert.deepEqual(bound,bounds[0],'all panel plot boundaries align');
    const exported=await page.evaluate(async()=>{const {scientificPlotsSvg}=await import('/src/design/plot-export.ts');const snapshot=scientificPlotsSvg([...document.querySelectorAll('.inverse-grid>section')]);const xml=new DOMParser().parseFromString(snapshot.text,'image/svg+xml'),root=xml.documentElement;return {width:snapshot.width,height:snapshot.height,titles:[...root.children].filter(e=>e.tagName==='text').map(e=>({text:e.textContent,anchor:e.getAttribute('text-anchor'),x:+e.getAttribute('x')})),xml:snapshot.text};});
    assert.equal(exported.titles.length,count);assert(exported.titles.every(t=>t.anchor==='middle'));assert(!exported.xml.includes('归一化分析音频'));assert(!exported.xml.includes('plot-legend'));
    assert(exported.xml.includes('#174b82'));assert(exported.xml.includes('#a63f10'));assert(exported.xml.includes('#633d91'));assert(exported.xml.includes('stroke-dasharray="8 4"'));assert(exported.xml.includes('stroke-dasharray="1 5"'));
    const download=page.waitForEvent('download');await click(`保存当前 ${count} 图 PNG`);const d=await download;await d.saveAs(path.join(out,d.suggestedFilename()));await fs.writeFile(path.join(out,d.suggestedFilename()+'.svg'),exported.xml);
    await page.screenshot({path:path.join(out,`r3-${count}-${outputs.length}.png`)});outputs.push({name,count,file:d.suggestedFilename(),width:exported.width,height:exported.height,titles:exported.titles});
  }
  checks.push('all 4 modes save actual 6/4/4/2-panel PNGs, aligned plot boundaries, centered title-only headers, distinct print colors and strokes');
  const download=page.waitForEvent('download');await page.getByRole('button',{name:'保存此图 PNG',exact:true}).first().click();const d=await download;await d.saveAs(path.join(out,'single.png'));checks.push('single-panel PNG');
  await page.getByLabel('显示 EGG 波形',{exact:true}).check();assert.equal(await page.getByRole('heading',{name:'滤波 EGG',exact:true}).count(),1);
  assert.equal(await page.locator('dialog .audio-transport').count(),2);
  await page.getByRole('heading',{name:'滤波 EGG',exact:true}).scrollIntoViewIfNeeded();await page.screenshot({path:path.join(out,'r3-full-egg.png')});
  await page.getByLabel('显示 EGG 波形',{exact:true}).uncheck();assert.equal(await page.getByRole('heading',{name:'滤波 EGG',exact:true}).count(),0);checks.push('full selected EGG can be shown and hidden, audio/IF playback retained');
  await click('选择目录保存完整结果');await page.waitForFunction(()=>document.querySelector('dialog').innerText.includes('已保存'));
  const saved=await fs.readdir(path.join(out,'saved'));const meta=JSON.parse(await fs.readFile(path.join(out,'saved',saved.find(f=>f.endsWith('.ptb.json'))),'utf8'));assert.equal(meta.inverse_view.full_egg_values.length,22050);assert.equal(meta.inverse_view.sample_rate_hz,44100);checks.push('full EGG persisted in real immutable task snapshot');
  await click('返回分析');
  await page.setViewportSize({width:960,height:720});await page.waitForTimeout(150);assert(await page.getByLabel('EGG 音频文件').isVisible());await page.screenshot({path:path.join(out,'r3-small-window.png')});
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(out,'r3-report.json'),JSON.stringify({checks,geometry,db,outputs,saved,errors},null,2));console.log(JSON.stringify({out,checks,db},null,2));
 }catch(e){console.log(await page.locator('.egg-page').innerText());await page.screenshot({path:path.join(out,'failed.png'),fullPage:true});throw e;}finally{await browser.close();await server.close();worker.stdin.write(JSON.stringify({op:'shutdown',rpc_id:++counter})+'\n');await new Promise(resolve=>worker.once('exit',resolve));}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
