const fs=require('node:fs/promises'),path=require('node:path'),os=require('node:os'),assert=require('node:assert/strict');
const [origin,out,scratch]=process.argv.slice(2);
const wasm=process.env.PTB_VOCAL_WASM==='1';
const {chromium}=require(path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
(async()=>{
 const credentials=process.env.PTB_VOCAL_QA_ACCOUNT?JSON.parse(await fs.readFile(process.env.PTB_VOCAL_QA_ACCOUNT,'utf8')):{username:'vocal-qa',password:'vocal-qa-local-password'};
 const ctx=await chromium.launchPersistentContext(path.join(scratch,'chrome'),{headless:true,executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',viewport:{width:1600,height:1050},acceptDownloads:true,downloadsPath:path.join(scratch,'downloads'),args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
 const page=await ctx.newPage(),errors=[],checks=[],httpErrors=[],vocalRequests=[],runtimeResponses=[];let frame;page.setDefaultTimeout(90000);
 if(process.env.PTB_VOCAL_QA_FIELDS){
  // This owned loopback harness has no account-settings database. Only settings
  // use this explicitly labelled fixture; auth, project and video routes are real.
  const fields=JSON.parse(await fs.readFile(process.env.PTB_VOCAL_QA_FIELDS,'utf8')),preferences=new Map();
  await page.route('**/api/v1/preferences**',async route=>{
   const module=new URL(route.request().url()).pathname.split('/')[4];
   if(!module)return route.fulfill({json:{modules:[...preferences.values()],allowed_fields:fields}});
   const body=route.request().postDataJSON(),before=preferences.get(module)??{module,revision:0,values:{}},value={module,revision:before.revision+1,values:{...before.values,...body.changes},updated_at:new Date().toISOString()};preferences.set(module,value);await route.fulfill({json:value});
  });
 }
 page.on('request',r=>{if(r.url().includes('/vocal/sessions')&&r.method()==='POST'){const data=r.postDataJSON();if(data?.op)vocalRequests.push(data.op);}});
 page.on('response',r=>{if(/\/(browser-science|vocal-wasm)\//.test(r.url()))runtimeResponses.push({url:r.url(),bytes:Number(r.headers()['content-length']||0)});});
 page.on('pageerror',e=>errors.push(e.message));page.on('response',r=>{if(r.url().includes('/vocal/')&&r.status()>=400)httpErrors.push({status:r.status(),url:r.url()});});
 const check=s=>{checks.push(s);console.log(s);};const audioEvidence={};
 const audioStatus=()=>frame.evaluate(async()=>{const {request}=await import('./platform.js');return request('status');});
 const ready=async()=>{await page.locator('iframe[title="生理参数合成"]').waitFor();frame=page.frames().find(f=>f.url().includes('vocal-tract/index.html'))||await page.waitForEvent('framenavigated',{predicate:f=>f.url().includes('vocal-tract/index.html')});await frame.waitForFunction(()=>document.body?.dataset.engineState==='ready');await pose();};
 const pose=async()=>{await frame.waitForFunction(()=>document.body.dataset.posePending==='false');};
 const state=()=>frame.evaluate(async()=>{const {viewer}=await import('./app.js');return {count:viewer.state.section_count,params:viewer.state.params,areas:viewer.state.tube_areas,meshes:viewer.state.meshes.length};});
 const pressRange=async(selector,key)=>{const before=await frame.locator('body').getAttribute('data-pose-revision');await frame.locator(selector).focus();await frame.locator(selector).press(key);await frame.waitForFunction(old=>document.body.dataset.poseRevision!==old,before);await pose();};
 try{
  await page.goto(origin+'/app/');await page.getByLabel('登录名').fill(credentials.username);await page.getByLabel('密码',{exact:true}).fill(credentials.password);await page.getByRole('button',{name:'登录',exact:true}).click();await page.locator('#tool-sidebar').waitFor();const coldStart=Date.now();await page.locator('#tool-sidebar').getByRole('button',{name:'生理参数合成',exact:true}).click();await ready();
  if(await page.getByRole('button',{name:'知道了',exact:true}).isVisible().catch(()=>false))await page.getByRole('button',{name:'知道了',exact:true}).click();
  if(wasm){audioEvidence.cold_start_ms=Date.now()-coldStart;const meta=await frame.evaluate(async()=>{const {request}=await import('./platform.js');return request('meta');});assert.equal(meta.execution,'browser-wasm');assert.equal(vocalRequests.length,0);check('actual Wasm native engine, shared Python state machine and zero server scientific requests');}
  assert.equal(await frame.locator('#sectionCount').inputValue(),'33');assert.equal((await state()).count,33);assert((await state()).meshes>=9);
  const options=await frame.locator('#sectionCount option').evaluateAll(nodes=>nodes.map(n=>({value:n.value,disabled:n.disabled})));assert.deepEqual(options,[{value:'33',disabled:false},{value:'65',disabled:false},{value:'129',disabled:true},{value:'257',disabled:true}]);assert.match(await frame.locator('.web-sampling-note a').getAttribute('href'),/^https:\/\/www\.phonetictoolbox\.com\//);check('33 native sections by default; 65 selectable; 129/257 disabled with desktop link');
  await frame.locator('#sectionCount').selectOption('65');await frame.waitForFunction(async()=>{const {viewer}=await import('./app.js');return viewer.state.section_count===65;});await pose();assert.equal((await state()).count,65);await frame.locator('#sectionCount').selectOption('33');await frame.waitForFunction(async()=>{const {viewer}=await import('./app.js');return viewer.state.section_count===33;});await pose();check('33 / 65 geometry recomputes and returns to 33');
  const before=await state();await pressRange('#p-TCX','ArrowRight');const after=await state();assert.notDeepEqual(after.params,before.params);await frame.locator('#undoButton').click();await pose();await frame.locator('#redoButton').click();await pose();check('real organ parameter edit / undo / redo');
  await frame.locator('#threeButton').click();await frame.locator('[data-mode="airway"]').click();await frame.locator('#fullModelToggle').check();await frame.locator('[data-mode="overlay"]').click();await frame.locator('#sagittalButton').click();check('3D / complete model / airway / overlay / sagittal views');
  await frame.locator('#savePreset').click();await frame.locator('#presetInputButton').click();await frame.locator('#presetInput').fill('qa-a');await frame.locator('#presetInputSave').click();await frame.locator('#presetDialog').waitFor({state:'hidden'});await frame.locator('#customPreset').click();assert((await frame.locator('#presetChart').innerText()).includes('qa-a'));await frame.locator('#presetCancel').click();check('custom named shape saves and appears in shared IPA library');
  await frame.locator('#tab-sound').click();assert(await frame.locator('#liveButton').isDisabled());assert.match(await frame.locator('.web-disabled-action').getAttribute('title'),/服务器性能.*桌面版/);
  const fixedPose=await state();await frame.locator('#oneSecondButton').click();await frame.waitForFunction(()=>document.body.dataset.vowelBusy==='true');
  for(const selector of ['#p-TCX','#sectionCount','#sourceMode','#f0Range','#undoButton'])assert(await frame.locator(selector).isDisabled());
  await frame.waitForFunction(()=>document.querySelector('#audioStatus').textContent.includes('元音试听'));await frame.waitForFunction(()=>document.querySelector('#levelMeter').dataset.rms>0);await frame.waitForFunction(()=>document.body.dataset.vowelBusy==='false');await pose();assert.deepEqual((await state()).params,fixedPose.params);assert(!(await frame.locator('#p-TCX').isDisabled()));assert(await frame.locator('#liveButton').isDisabled());check((wasm?'browser Wasm':'native server')+' 0.5-second PCM; pose locked during generation and playback, restored afterward; continuous phonation disabled with tooltip');
  audioEvidence.preview=await audioStatus();assert.equal(audioEvidence.preview.audio.underflows,0);
  await frame.locator('[data-analysis="monitor"]').click();await frame.waitForFunction(()=>Number(document.querySelector('#waveCanvas').dataset.sampleCount)>0||document.querySelector('#spectrogramCanvas').width>0);check('acknowledged output waveform and spectrogram');
  await frame.locator('#sourceMode').selectOption('whisper');await pose();await frame.locator('#sourceMode').selectOption('voiceless');await pose();await frame.locator('#sourceMode').selectOption('voiced');await pose();check('voiced / whisper / voiceless source controls');
  await frame.locator('#tab-motion').click();await frame.locator('#captureFrame').click();await frame.waitForFunction(()=>document.querySelectorAll('.pose-card').length===1);await frame.locator('.pose-card input[type="number"]').fill('0.1');await frame.locator('.pose-card input[type="number"]').press('Tab');assert.equal(await frame.locator('#captureAspiration').count(),0);await frame.locator('#captureSilence').click();await frame.waitForFunction(()=>document.querySelectorAll('.pose-card').length===2);check('ordinary / silent keyframes; aspiration uses VOT');
  const pending=page.waitForEvent('download');await frame.locator('#exportFrames').click();const docDownload=await pending,doc=JSON.parse(await fs.readFile(await docDownload.path(),'utf8'));assert.equal(doc.frames.length,2);assert.equal(doc.section_count,33);assert.equal(doc.version,5);const docPath=path.join(scratch,'roundtrip.ptb-vocal.json');await fs.writeFile(docPath,JSON.stringify(doc));
  await frame.locator('#clearFrames').click();await frame.waitForFunction(()=>document.querySelectorAll('.pose-card').length===0);const choose=page.waitForEvent('filechooser');await frame.locator('#importFrames').click();await(await choose).setFiles(docPath);await frame.waitForFunction(()=>document.querySelectorAll('.pose-card').length===2);check('version 5 JSON download / clear / import validates and restores all frames');
  for(const input of await frame.locator('.pose-card input[type="number"]').all()){await input.fill('1.7');await input.press('Tab');}
  assert(await frame.locator('#playFrames').isDisabled());assert(await frame.locator('#exportVideo').isDisabled());assert.match(await frame.locator('#frameStatus').innerText(),/5 秒以内/);
  const restore=page.waitForEvent('filechooser');await frame.locator('#importFrames').click();await(await restore).setFiles(docPath);await frame.waitForFunction(()=>!document.querySelector('#playFrames').disabled);check('web sequence over five seconds cannot play or export; document restore remains available');
  await pose();await frame.locator('#playFrames').click();await frame.waitForFunction(()=>document.querySelector('#frameStatus').textContent.includes('播放结束'));audioEvidence.animation=await audioStatus();assert.equal(audioEvidence.animation.audio.underflows,0);check('generated synchronized motion and audio complete');
  await frame.locator('#exportVideo').click();const videoDownload=page.waitForEvent('download',{timeout:180000});await frame.locator('#videoStart').click();const video=await videoDownload;const mp4=await fs.readFile(await video.path());assert.equal(mp4.toString('ascii',4,8),'ftyp');await video.saveAs(path.join(scratch,'action.mp4'));await frame.waitForFunction(()=>document.querySelector('#videoStatus').textContent.includes('已保存'));await frame.locator('#videoClose').click();check('real WebCodecs packets -> bounded server MP4 H264/AAC -> browser download');
  await page.reload();await page.locator('#tool-sidebar').waitFor();await page.locator('#tool-sidebar').getByRole('button',{name:'生理参数合成',exact:true}).click();await ready();await frame.locator('#tab-motion').click();assert.equal(await frame.locator('.pose-card').count(),3);check('refresh restores this tab/account/project sequence');
  for(const theme of ['light','dark']){await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);await page.waitForTimeout(100);await page.screenshot({path:path.join(out,theme+'.png')});}check('light / dark actual workbench screenshots');
  if(wasm){
   audioEvidence.wasm=await frame.evaluate(async()=>{
    const {request}=await import('./platform.js'),sequence=await request('keyframes/load'),frames=sequence.frames.filter(f=>!f.aspiration&&!f.silent).slice(0,1);
    const timings=[];for(let i=0;i<5;i++){const begin=performance.now();await request('meta');timings.push(performance.now()-begin);}
    frames[0].duration=2.5;frames.push(structuredClone(frames[0]));const start=performance.now();const prepared=await request('animation/prepare',{frames,pitch_curve:[]});const sequence_ms=performance.now()-start;
    const raw=await request('animation/audio',{id:prepared.id});
    frames[0].f0=171;const rendering=request('animation/prepare',{frames,pitch_curve:[]}).then(()=>({cancelled:false}),e=>({cancelled:true,error:String(e)}));
    await new Promise(r=>setTimeout(r,25));const stop=performance.now();await request('animation/stop');const cancel_ms=performance.now()-stop,cancelled=await rendering;
    const restored=await request('presets/load'),recovered=await request('keyframes/load'),meta=await request('meta');
    return {meta_ms:timings,sequence_ms,sequence_samples:raw.samples,sequence_duration:prepared.duration,cancel_ms,cancelled,restored_custom:restored.some(p=>p.name==='qa-a'),restored_frames:recovered.frames.length,execution:meta.execution};
   });
   assert.equal(audioEvidence.wasm.sequence_samples,240000);assert.equal(audioEvidence.wasm.sequence_duration,5);assert(audioEvidence.wasm.cancelled.cancelled);assert(audioEvidence.wasm.restored_custom);assert.equal(audioEvidence.wasm.restored_frames,3);assert.equal(audioEvidence.wasm.execution,'browser-wasm');
   assert(vocalRequests.every(op=>op.startsWith('video/')));check('five-second local synthesis, immediate cancellation and acknowledged state recovery; server used only for video encoding');
   const probe=await page.evaluate(()=>new Promise((resolve,reject)=>{
    const worker=new Worker(new URL('./wasm-qa/vocal-wasm-probe.worker.js',location.href));
    let last=performance.now(),maxGap=0;const timer=setInterval(()=>{const now=performance.now();maxGap=Math.max(maxGap,now-last);last=now;},16);
    worker.onmessage=e=>{clearInterval(timer);worker.terminate();e.data.error?reject(Error(e.data.error)):resolve({...e.data,main_thread_max_timer_gap_ms:maxGap});};
    worker.onerror=e=>{clearInterval(timer);worker.terminate();reject(Error(e.message));};worker.postMessage({});
   }));
   await fs.writeFile(path.join(scratch,'wasm-browser.json'),probe.result);delete probe.result;audioEvidence.scientific_probe=probe;check('independent actual-browser/native recipe outputs and non-blocking main-thread measurement');
  }
  assert.deepEqual(errors,[]);assert.deepEqual(httpErrors,[]);await fs.writeFile(path.join(out,'browser.json'),JSON.stringify({success:true,checks,errors,httpErrors,audioEvidence,mp4_bytes:mp4.length,vocalRequests,runtimeResponses},null,2));
 }catch(error){await page.screenshot({path:path.join(out,'failure.png')});await fs.writeFile(path.join(out,'failure.txt'),JSON.stringify({error:String(error),errors,httpErrors,body:await page.locator('body').innerText(),frame:frame?await frame.locator('body').innerText():null},null,2));throw error;}
 finally{try{const response=await ctx.request.get(origin+'/api/v1/auth/me');if(response.ok()){const session=await response.json();const loggedOut=await ctx.request.post(origin+'/api/v1/auth/logout',{headers:{Origin:origin,'X-CSRF-Token':session.csrf_token}});assert.equal(loggedOut.status(),204);}}finally{await ctx.close();}}
})().catch(e=>{console.error(e);process.exitCode=1;});
