// M04-D: independent Chrome, real TaskBridge/HTTP/MKL child. No EXE or DDL.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{spawn}=require('node:child_process'),{createInterface}=require('node:readline'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const worker=spawn(path.join(root,'.venv/m09-ui/Scripts/python.exe'),['-X','utf8','scripts/m04_ui_bridge.py'],{cwd:root,windowsHide:true,env:{...process.env,PYTHONPATH:path.join(root,'backend/src')+';'+path.join(root,'desktop/src')+';'+path.join(root,'packages/phonetic_core/src')}});
 const pending=new Map();let counter=0,resolveReady,rejectReady;const ready=new Promise((r,j)=>{resolveReady=r;rejectReady=j;});
 createInterface({input:worker.stdout}).on('line',line=>{const v=JSON.parse(line);if(v.ready)resolveReady(v);else{const done=pending.get(v.id);pending.delete(v.id);done?.(v);}});worker.stderr.on('data',d=>process.stderr.write(d));worker.on('error',rejectReady);worker.once('exit',code=>rejectReady(Error('bridge exited '+code)));
 const {out}=await ready;const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0},optimizeDeps:{entries:['tests/m04-live.html']},plugins:[{name:'owned-m04-rpc',configureServer(s){s.middlewares.use('/__m04',(req,res)=>{let body='';req.on('data',d=>body+=d);req.on('end',()=>{const data=JSON.parse(body),id=++counter;pending.set(id,v=>{res.setHeader('Content-Type','application/json');res.end(JSON.stringify(v));});worker.stdin.write(JSON.stringify({...data,rpc_id:id})+'\n');});});}}]});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});const context=await browser.newContext({viewport:{width:1440,height:1000}}),page=await context.newPage(),checks=[],errors=[],requests=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(r.url().endsWith('/__m04'))requests.push(r.postDataJSON());});
 const click=name=>page.getByRole('button',{name,exact:true}).first().click();const plotted=()=>page.locator('.lpc-spectrum:visible svg').waitFor({timeout:60000});
 const range=async(a,b)=>{const t=page.locator('.lpc-transport');await t.getByRole('spinbutton',{name:/^起点/}).fill(String(a));await t.getByRole('spinbutton',{name:/^终点/}).fill(String(b));await t.getByRole('spinbutton',{name:/^终点/}).blur();};
 const source=async name=>{await page.getByLabel('LPC 音频文件').selectOption({label:name});await page.locator('.lpc-page .wave-track svg').first().waitFor();await page.waitForFunction(()=>!document.querySelector('.lpc-files').innerText.includes('正在读取音频'));};


 const report={checks,errors,requests,layouts:[],png:[]};
 const transport=()=>page.locator('.lpc-transport');
 const axes=()=>page.locator('.lpc-spectrum svg').evaluate(el=>[...el.querySelectorAll('text[text-anchor="end"]')].filter(t=>t.getAttribute('x')!==String(el.viewBox.baseVal.width-4)).map(t=>t.textContent));
 const previewReady=async()=>{await page.waitForTimeout(80);await page.locator('.spectrogram-canvas[aria-busy=false] canvas:visible').waitFor({timeout:45000});};
 const snapshot=async(name)=>{await page.screenshot({path:path.join(out,name+'.png'),fullPage:true});report.layouts.push({name,...await page.evaluate(()=>({width:innerWidth,height:innerHeight,bodyOverflow:document.documentElement.scrollWidth>innerWidth,transport:document.querySelector('.lpc-transport')?.getBoundingClientRect().toJSON()}))});};
 const savePng=async(name)=>{const p=page.waitForEvent('download');await click('保存 PNG 图片');const d=await p;await d.saveAs(path.join(out,name+'.png'));report.png.push(name+'.png');return await fs.readFile(path.join(out,name+'.png'));};
 try{
  await page.goto(server.resolvedUrls.local[0]+'tests/m04-live.html');await click('LPC 谱图');await click('打开 WAV 目录');await source('LPC ɑ̃˥.wav');
  assert.equal(await page.getByRole('button',{name:'清除选区',exact:true}).count(),0);assert.equal(await page.locator('.workbench-left .audio-transport').count(),0);
  assert.equal(await transport().getByRole('button',{name:'全部',exact:true}).count(),1);assert.equal(await transport().getByLabel('播放音量').count(),1);
  await range(.1,.2);await click('开始分析');await plotted();const before=requests.filter(r=>r.op==='lpc').length;
  const buttons=await page.locator('.lpc-spectrum .actions button').allTextContents();assert.deepEqual(buttons.slice(-3),['适合频率范围','动态纵轴','固定纵轴']);
  const fixedAxes=await axes();assert.deepEqual(fixedAxes.slice(0,5),['-5','5','15','25','35']);const fixed=await savePng('fixed');
  await click('动态纵轴');assert.equal(await page.getByRole('button',{name:'动态纵轴',exact:true}).getAttribute('aria-pressed'),'true');
  const dynamicAxes=await axes();assert.notDeepEqual(dynamicAxes,fixedAxes);assert.equal(await page.locator('.lpc-page .notice').count(),0);
  const dynamic=await savePng('dynamic');assert(!fixed.equals(dynamic));
  const jobs=await page.evaluate(async()=> (await(await fetch('/__m04',{method:'POST',body:JSON.stringify({op:'lpc_jobs'})})).json()).value);
  const target=requests.filter(r=>r.op==='result').at(-1);const job=jobs.find(j=>j.id===target.job),json=job.result_manifest.files.find(f=>f.name==='lpc.ptb.json');
  const data=await page.evaluate(async({job,id})=>{const v=(await(await fetch('/__m04',{method:'POST',body:JSON.stringify({op:'result',job,id})})).json()).value;return JSON.parse(atob(v.base64));},{job:job.id,id:json.id});
  await fs.writeFile(path.join(out,'task.json'),JSON.stringify(data,null,2));
  const visible=data.spectrum.magnitude_db.filter((_,i)=>data.spectrum.frequencies_hz[i]<=data.config.freq_max_hz),y=[Math.min(...visible)-5,Math.max(...visible)+5];
  report.axes={fixed:fixedAxes,dynamic:dynamicAxes,expected:y};assert(Math.abs(Number(dynamicAxes[0])-y[0])<.51);assert(Math.abs(Number(dynamicAxes[4])-y[1])<.51);assert.equal(data.config.dynamic_y,false);assert.equal(data.spectrum.frequencies_hz.length,1024);
  await page.getByLabel('放大 LPC 频谱').click();const zoomed=await savePng('dynamic-zoomed');assert(dynamic.equals(zoomed));
  await click('固定纵轴');const again=await savePng('fixed-return');assert(fixed.equals(again));
  await page.getByLabel('LPC 幅度下限').fill('-20');await page.getByLabel('LPC 幅度上限').fill('60');
  const custom=await savePng('fixed-custom');assert(!custom.equals(fixed));assert.equal(await page.locator('.lpc-page .notice').count(),0);
  assert.equal(requests.filter(r=>r.op==='lpc').length,before);checks.push('Axis buttons update without jobs; independent bounds match spectrum; 5 current-axis PNG exports; zoom-independent PNG; immutable task snapshot');
  await page.getByLabel('LPC 幅度下限').fill('');assert(await page.getByRole('button',{name:'保存 PNG 图片',exact:true}).isDisabled());await page.getByLabel('LPC 幅度下限').fill('-5');await page.getByLabel('LPC 幅度上限').fill('35');checks.push('Invalid fixed bounds block export and recover without recomputation');
  await click('选择目录保存完整结果');await page.getByRole('status').filter({hasText:'已保存 3'}).waitFor();checks.push('Original task PNG/WAV/JSON bundle stays available');
  await click('波形');await page.getByLabel('显示语谱图（Praat）').check();await previewReady();
  const c=requests.filter(r=>r.op==='spectrogram').length;await range(.2,.4);await previewReady();await page.waitForTimeout(180);assert.equal(requests.filter(r=>r.op==='spectrogram').length,c);checks.push('Numeric selection preserves preview without recalculating unchanged viewport');
  let held=false,release;const gate=new Promise(r=>release=r);const slow=async route=>{const req=route.request().postDataJSON();if(req.op==='spectrogram'&&!held){held=true;const response=await route.fetch();await gate;await route.fulfill({response});}else await route.continue();};
  await page.route('**/__m04',slow);await page.getByLabel('放大波形',{exact:true}).click();await page.waitForFunction(()=>document.querySelector('.spectrogram-canvas')?.getAttribute('aria-busy')==='true');
  const centerBefore=await page.locator('.wave-viewport').evaluate(e=>e.getBoundingClientRect().width);
  await page.getByLabel('放大波形',{exact:true}).click();await page.getByLabel('平移波形时间窗').evaluate(e=>{e.value='0.7';e.dispatchEvent(new Event('input',{bubbles:true}));});await page.waitForTimeout(150);release();await previewReady();await page.unroute('**/__m04',slow);
  const last=requests.filter(r=>r.op==='spectrogram').at(-1);assert(Math.abs(last.view.start-.7)<1e-9);assert(Math.abs(last.view.end-1.2)<1e-9);assert((await page.locator('.spectrogram-canvas canvas').getAttribute('aria-label')).includes('0.700至1.200'));assert.equal(await page.locator('.wave-viewport').evaluate(e=>e.getBoundingClientRect().width),centerBefore);
  const settled=requests.filter(r=>r.op==='spectrogram').length;await page.waitForTimeout(750);assert.equal(requests.filter(r=>r.op==='spectrogram').length,settled);checks.push('Delayed old request discarded; newest zoom/pan renders; stable dimensions and settled request count');
  let busy=0;const reject=async route=>{const req=route.request().postDataJSON();if(req.op==='spectrogram'&&busy++===0)await route.fulfill({json:{error:'preview_busy'}});else await route.continue();};await page.route('**/__m04',reject);await page.getByLabel('放大波形',{exact:true}).click();await previewReady();await page.unroute('**/__m04',reject);checks.push('Busy preview retries and renders actual Praat data');
  const fail=async route=>{if(route.request().postDataJSON().op==='spectrogram')await route.fulfill({json:{error:'preview_failed'}});else await route.continue();};await page.route('**/__m04',fail);await page.getByLabel('放大波形',{exact:true}).click();await page.locator('.spectrogram-view [role=alert]').waitFor();await page.unroute('**/__m04',fail);await page.locator('.spectrogram-view').getByRole('button',{name:'重试',exact:true}).click();await previewReady();checks.push('Preview failure ends loading; retry recovers real canvas');
  await source('second.wav');await previewReady();assert.equal(requests.filter(r=>r.op==='spectrogram').at(-1).id,requests.filter(r=>r.op==='read').at(-1).id);checks.push('Equal-duration file switch refreshes correct source');
  await click('全部');assert.equal(await transport().getByRole('spinbutton',{name:/^起点/}).inputValue(),'0');assert.equal(await transport().getByRole('spinbutton',{name:/^终点/}).inputValue(),'2');
  await range(.1,.4);await transport().getByRole('button',{name:'播放选区',exact:true}).click();await transport().getByRole('button',{name:'暂停',exact:true}).waitFor();await transport().getByRole('button',{name:'停止',exact:true}).click();await transport().getByLabel('播放音量').evaluate(e=>{e.value='0.4';e.dispatchEvent(new Event('input',{bubbles:true}));});assert((await transport().textContent()).includes('40%'));checks.push('Shared original-time selection/all/play/stop/seek/volume controls');
  for(const [w,h] of [[1920,1080],[1280,720],[900,680]]){await page.setViewportSize({width:w,height:h});await previewReady();await snapshot('wave-'+w);assert.equal(report.layouts.at(-1).bodyOverflow,false);assert(report.layouts.at(-1).transport.top>=0&&report.layouts.at(-1).transport.bottom<=h);}
  await page.evaluate(()=>document.documentElement.dataset.theme='dark');await snapshot('wave-dark');await page.reload();await click('LPC 谱图');await page.locator('.history-links button').filter({hasText:job.id.slice(0,8)}).click();await plotted();assert(Math.abs(Number(await transport().getByRole('spinbutton',{name:/^终点/}).inputValue())-.1)<1e-8);await transport().getByRole('button',{name:'播放选区',exact:true}).click();await transport().getByRole('button',{name:'暂停',exact:true}).waitFor();await transport().getByRole('button',{name:'停止',exact:true}).click();checks.push('Historical clip remains playable without loading original audio');assert.deepEqual(errors,[]);report.success=true;
 }catch(e){report.failure=String(e.stack);await page.screenshot({path:path.join(out,'r2-failed.png'),fullPage:true});throw e;}
 finally{await fs.writeFile(path.join(out,'r2-report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({out,checks,errors,failure:report.failure}));await browser.close();await server.close();worker.stdin.write(JSON.stringify({op:'shutdown',rpc_id:++counter})+'\n');await new Promise(r=>worker.once('exit',r));}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
