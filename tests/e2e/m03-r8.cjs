// Actual local API/compatible child/native REAPER and standalone owned Chrome.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {spawn}=require('node:child_process'),{createInterface}=require('node:readline'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const worker=spawn(path.join(root,'.venv/m14/Scripts/python.exe'),['-B','-X','utf8','scripts/m03_ui_bridge.py','--reaper','--managed-scratch'],{cwd:root,windowsHide:true,env:{...process.env,PYTHONPATH:['backend/src','desktop/src','packages/phonetic_core/src'].map(p=>path.join(root,p)).join(';')}});
 const pending=new Map();let counter=0,resolveReady,rejectReady;
 const ready=new Promise((resolve,reject)=>{resolveReady=resolve;rejectReady=reject;});
 createInterface({input:worker.stdout}).on('line',line=>{
  let value;try{value=JSON.parse(line);}catch{process.stdout.write(line+'\n');return;}
  if(value.ready)resolveReady(value);else{const done=pending.get(value.id);pending.delete(value.id);done?.(value);}
 });
 worker.stderr.on('data',d=>process.stderr.write(d));worker.on('error',rejectReady);worker.on('exit',code=>rejectReady(Error('Bridge exited '+code)));
 let server,browser,page,out,scratch;const checks=[],errors=[],responses=[];
 try{
  ({out,scratch}=await ready);
  const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
  server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0,watch:null},optimizeDeps:{noDiscovery:true,include:['vue']},plugins:[{name:'owned-m03-r8-rpc',configureServer(s){s.middlewares.use('/__m03',(req,res)=>{let body='';req.on('data',d=>body+=d);req.on('end',()=>{const data=JSON.parse(body),id=++counter;pending.set(id,value=>{res.setHeader('Content-Type','application/json');res.end(JSON.stringify(value));});worker.stdin.write(JSON.stringify({...data,rpc_id:id})+'\n');});});}}]});
  await server.listen();
  const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
  browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--mute-audio']});
  page=await browser.newPage({viewport:{width:1440,height:1000}});page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/__m03',async route=>{const response=await route.fetch();if(route.request().postDataJSON().op==='egg_preview_update'){const value=await response.json();if(value.value)responses.push(value.value);}await route.fulfill({response});});
  const idle=async()=>{await page.waitForTimeout(80);await page.waitForFunction(()=>document.querySelector('.egg-live-status')?.textContent==='实时预览'&&[...document.querySelectorAll('button')].some(b=>b.textContent==='保存 CSV / 三图'&&!b.disabled),{},{timeout:60000});};
  const fill=async(label,value)=>{await page.getByLabel(label,{exact:true}).fill(String(value));await page.getByLabel(label,{exact:true}).press('Tab');await idle();};
  const overlay=()=>page.locator('.cq-pane .plot-highlight').evaluate(el=>{const parent=el.parentElement,frame=parent.parentElement.querySelector('defs rect'),line=parent.querySelector('line[data-marker-label="微观中心"]');return {x:+el.getAttribute('x'),width:+el.getAttribute('width'),height:+el.getAttribute('height'),frameX:+frame.getAttribute('x'),frameWidth:+frame.getAttribute('width'),center:+line.getAttribute('x1'),fill:getComputedStyle(el).fill};});
  const checkOverlay=async()=>{const config=responses.at(-1).config,value=await overlay(),span=config.roi_end-config.roi_start;assert(Math.abs(value.width/value.frameWidth-config.micro_width_ms/1000/span)<1e-6);assert(Math.abs(value.x+value.width/2-value.center)<1e-6);const a=await page.locator('.spec-pane .plot-highlight').getAttribute('width');assert(Math.abs(+a-value.width)<1e-6);for(const pane of ['audio','egg'])assert.equal(await page.locator(`.${pane}-pane line[data-marker-label="微观中心 · 0 ms"]`).count(),1);return value;};
  await page.goto(server.resolvedUrls.local[0]+'tests/m03-r2.html');
  await page.getByRole('button',{name:'EGG 信号分析',exact:true}).click();await page.getByRole('button',{name:'打开音频目录',exact:true}).click();await page.getByLabel('EGG 音频文件').selectOption({label:'EGG ɑ̃˥.wav'});await idle();
  assert.equal(await page.locator('.cq-pane .plot-legend,.audio-pane .plot-legend,.egg-pane .plot-legend,.egg-colorbar').count(),0);
  assert.equal(await page.locator('.spec-pane .plot-legend').count(),0);
  for(const pane of ['audio','egg'])assert((await page.locator(`.${pane}-pane h2`).innerText()).endsWith('50.00 ms'));
  await checkOverlay();checks.push('specified legends/colorbar removed; all four graphs gain plot space; 50 ms titles and clipped theme overlays with center/zero lines');
  const boxes=await Promise.all(['EGG dB 下限','EGG dB 上限'].map(label=>page.getByLabel(label,{exact:true}).boundingBox()));assert(Math.abs(boxes[0].y-boxes[1].y)<1);
  const autoBox=await page.getByRole('button',{name:/^自动 dB/}).boundingBox(),windowBox=await page.locator('.egg-spectral-grid input').first().boundingBox();assert(Math.abs(autoBox.y-windowBox.y)<1);assert(autoBox.x>windowBox.x);
  await fill('EGG 微观窗口',100);await checkOverlay();for(const pane of ['audio','egg'])assert((await page.locator(`.${pane}-pane h2`).innerText()).endsWith('100.00 ms'));
  const svg=page.locator('.cq-pane .scientific-plot>svg'),box=await svg.boundingBox();await page.mouse.click(box.x+box.width*.7,box.y+box.height*.5);await idle();await checkOverlay();
  const micro=await page.locator('.audio-pane .scientific-plot>svg').boundingBox(),beforeWidth=responses.at(-1).config.micro_width_ms;await page.mouse.move(micro.x+micro.width*.5,micro.y+micro.height*.5);await page.mouse.wheel(0,-120);await idle();assert(responses.at(-1).config.micro_width_ms<beforeWidth);await checkOverlay();
  const beforeCenter=responses.at(-1).config.micro_center;await page.mouse.down();await page.mouse.move(micro.x+micro.width*.6,micro.y+micro.height*.5,{steps:8});await page.mouse.up();await idle();assert.notEqual(responses.at(-1).config.micro_center,beforeCenter);await checkOverlay();
  await fill('EGG 微观窗口',100);
  checks.push('auto dB shares spectral-window row; dB limits share next row; width and seek update both left overlays and both right titles');
  await page.getByLabel('Praat F0',{exact:true}).check();await page.getByLabel('REAPER F0',{exact:true}).check();await idle();const baseline=responses.at(-1);
  await fill('EGG F0 下限 Hz',80);await fill('EGG F0 上限 Hz',260);const custom=responses.at(-1);assert.equal(custom.config.f0_min_hz,80);assert.equal(custom.config.f0_max_hz,260);assert(custom.preview.praat.values.some(v=>v>80&&v<260));assert(custom.preview.reaper.values.some(v=>v>80&&v<260));assert.notDeepEqual(custom.preview.praat,baseline.preview.praat);
  for(const key of ['cq','sq','gci_f0','audio','egg','gci','goi'])assert.deepEqual(custom.preview[key],baseline.preview[key]);
  checks.push('actual UI submits shared 80–260 Hz; both native detectors return voiced estimates; EGG/event/waveform data unchanged');
  await page.getByLabel('EGG F0 下限 Hz',{exact:true}).fill('300');await page.getByLabel('EGG F0 下限 Hz',{exact:true}).press('Tab');await page.getByText(/F0 上下限须在/).waitFor();await fill('EGG F0 下限 Hz',80);
  await page.getByRole('button',{name:/^自动 dB/}).click();await fill('EGG dB 下限',-90);await fill('EGG dB 上限',-20);const scale=responses.at(-1).config;assert.equal(scale.spec_vmin,-90);assert.equal(scale.spec_vmax,-20);
  await page.getByRole('button',{name:/^自动 dB/}).click();await idle();assert.equal(responses.at(-1).config.spec_vmax-responses.at(-1).config.spec_vmin,50);
  checks.push('invalid limits reject with recovery; manual/automatic dB still work after relocation');
  const layouts=[];
  for(const theme of ['light','dark'])for(const [width,height] of [[1920,1080],[1440,900],[800,600]]){
   await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);await page.setViewportSize({width,height});await page.waitForTimeout(150);
   const dB=await Promise.all(['EGG dB 下限','EGG dB 上限'].map(label=>page.getByLabel(label,{exact:true}).boundingBox()));assert(Math.abs(dB[0].y-dB[1].y)<1);
   assert(await page.locator('.egg-spectral-grid').evaluate(el=>el.scrollWidth<=el.clientWidth+1));assert(await page.locator('.egg-f0-grid').evaluate(el=>el.scrollWidth<=el.clientWidth+1));
   for(const pane of ['cq','spec','audio','egg'])assert(await page.locator(`.${pane}-pane .scientific-plot>svg`).evaluate(el=>el.getBoundingClientRect().height>100));
   const fill=(await overlay()).fill;assert(fill.startsWith('rgba(')&&!fill.endsWith(', 0)'));layouts.push({theme,width,height,fill});await page.screenshot({path:path.join(out,`r8-${theme}-${width}.png`),fullPage:true});
  }
  checks.push('6 actual light/dark/viewport layouts: controls stay aligned, highlight transparent, graph height remains readable');
  await page.setViewportSize({width:1440,height:1000});await page.getByRole('button',{name:'保存 CSV / 三图',exact:true}).click();const view=page.getByRole('button',{name:/查看 .*CSV/});await view.waitFor({timeout:120000});await view.click();await page.getByRole('button',{name:'选择目录保存完整结果',exact:true}).click();await page.getByRole('dialog').getByRole('status').filter({hasText:'已保存'}).waitFor();await page.getByRole('button',{name:'返回分析',exact:true}).click();
  const saved=path.join(scratch,'saved'),names=await fs.readdir(saved);const metadata=JSON.parse(await fs.readFile(path.join(saved,names.find(name=>name.endsWith('.ptb.json'))),'utf8'));for(const detector of ['praat','reaper'])assert.deepEqual([metadata.f0_analysis[detector].floor_hz,metadata.f0_analysis[detector].ceiling_hz],[80,260]);
  await page.getByRole('button',{name:'保存参数草稿',exact:true}).click();await page.reload();await page.getByRole('button',{name:'EGG 信号分析',exact:true}).first().click();assert.equal(await page.getByLabel('EGG F0 下限 Hz',{exact:true}).inputValue(),'80');assert.equal(await page.getByLabel('EGG F0 上限 Hz',{exact:true}).inputValue(),'260');
  checks.push('actual export provenance records both search ranges; parameter draft restores both shared limits');
  await page.getByRole('button',{name:'帮助',exact:true}).click();await page.waitForFunction(()=>document.querySelector('.manual-document')?.textContent?.includes('F0 下限 Hz'));const text=await page.locator('.manual-document').innerText();assert(text.includes('总窗长')&&text.includes('半透明矩形')&&text.includes('10–2000 Hz'));await page.screenshot({path:path.join(out,'r8-manual.png')});
  for(const key of ['analysis-roi','micro-events','praat-only','gci-only','spec-50ms','spec-5ms','three-f0','shared-f0-range']){
   const figure=page.locator(`.manual-document figure[data-asset-id="m03-r8-${key}"]`).first();await figure.scrollIntoViewIfNeeded();
   await figure.locator('img').evaluate(async img=>{await img.decode();assertSize();function assertSize(){if(img.naturalWidth!==2560||img.naturalHeight!==1600)throw Error('Unexpected manual screenshot dimensions');}});
  }
  assert.equal(await page.locator('.manual-document figure[data-asset-id^="m03-long-export-controls"]').count(),0);
  await page.screenshot({path:path.join(out,'r8-manual-natural-range.png')});checks.push('module help opens rebuilt EGG manual; all 8 natural-recording screenshots decode at 2560×1600; former synthetic screenshot nodes removed');
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(out,'r8-report.json'),JSON.stringify({success:true,checks,layouts,updates:responses.length,errors},null,2));console.log(JSON.stringify({out,checks,layouts}));
 }catch(error){if(page&&out){await page.screenshot({path:path.join(out,'r8-failed.png'),fullPage:true});await fs.writeFile(path.join(out,'r8-failed.json'),JSON.stringify({error:String(error),checks,errors,last:responses.at(-1)?.config},null,2));}throw error;}
 finally{
  if(browser)await browser.close();if(server)await server.close();
  if(worker.exitCode===null){const done=new Promise(resolve=>worker.once('exit',resolve));worker.stdin.write(JSON.stringify({op:'shutdown',rpc_id:++counter})+'\n');await done;}
  if(out){
   let receipt=JSON.parse(await fs.readFile(path.join(out,'cleanup.json'),'utf8'));
   // The owner's exit is the safe retry boundary after asynchronous child teardown.
   if(receipt.status==='retained'){
    const cleanup=spawn('powershell.exe',['-NoProfile','-ExecutionPolicy','Bypass','-File',path.join(root,'scripts/cleanup_test_scratch.ps1'),'-RunDirectory',out],{cwd:root,windowsHide:true});
    cleanup.stdout.on('data',d=>process.stdout.write(d));cleanup.stderr.on('data',d=>process.stderr.write(d));await new Promise(resolve=>cleanup.once('exit',resolve));receipt=JSON.parse((await fs.readFile(path.join(out,'cleanup.json'),'utf8')).replace(/^\uFEFF/,''));
   }
   console.log(JSON.stringify({cleanup:receipt}));assert.equal(receipt.status,'deleted');
  }
 }
}
main().catch(error=>{console.error(error);process.exitCode=1;});
