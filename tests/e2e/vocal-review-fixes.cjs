// Current app, real Chrome/Wasm, synthetic account and an isolated browser profile.
const fs=require('node:fs/promises'),path=require('node:path'),os=require('node:os'),assert=require('node:assert/strict');
const {execFileSync}=require('node:child_process');
const {chromium}=require(path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
const [origin,out,scratch]=process.argv.slice(2);
(async()=>{
 const ctx=await chromium.launchPersistentContext(path.join(scratch,'chrome'),{channel:'chrome',headless:true,viewport:{width:1600,height:1050},acceptDownloads:true,args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
 const page=await ctx.newPage(),errors=[],checks=[],evidence={};let frame;page.setDefaultTimeout(45000);
 page.on('pageerror',e=>errors.push(e.message));
 const fields=JSON.parse(await fs.readFile(process.env.PTB_VOCAL_QA_FIELDS,'utf8')),saved=new Map();
 await page.route('**/api/v1/preferences**',async route=>{const module=new URL(route.request().url()).pathname.split('/')[4];if(!module)return route.fulfill({json:{modules:[...saved.values()],allowed_fields:fields}});if(route.request().method()==='PATCH'){const value=route.request().postDataJSON();saved.set(module,{module,revision:(saved.get(module)?.revision||0)+1,values:{...saved.get(module)?.values,...value.values}});}return route.fulfill({json:saved.get(module)||{module,revision:0,values:{}}});});
 const call=(op,body={})=>frame.evaluate(async([op,body])=>(await import('./platform.js')).request(op,body),[op,body]);
 const settled=()=>frame.waitForFunction(()=>document.body?.dataset.engineState==='ready'&&document.body?.dataset.posePending==='false');
 const check=name=>{checks.push(name);console.log('PASS',name);};
 async function pcm(body){const clip=await call('animation/prepare',body),audio=await call('animation/audio',{id:clip.id});const bytes=Buffer.from(audio.base64,'base64');return {clip,y:new Float32Array(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength))};}
 try{
  await page.goto(origin+'/app/');await page.getByLabel('登录名',{exact:true}).fill('vocal-qa');await page.getByLabel('密码',{exact:true}).fill('vocal-qa-local-password');await page.getByRole('button',{name:'登录',exact:true}).click();await page.locator('#tool-sidebar').getByRole('button',{name:'生理参数合成',exact:true}).click();
  frame=await(await page.locator('iframe[title="生理参数合成"]').elementHandle()).contentFrame();await settled();assert.equal((await call('meta')).execution,'browser-wasm');
  for(const speaker of ['JD2','W02']){
   if((await call('meta')).speaker!==speaker){await frame.locator('#speakerSelect').selectOption(speaker);await frame.waitForURL(new RegExp('speaker='+speaker));await settled();}
   const meta=await call('meta');
   await frame.locator('#presets [data-preset="t"]').click();await settled();
   await frame.locator('#tab-sound').click();await frame.locator('#sourceMode').selectOption('voiceless');await settled();
   assert.equal(await frame.locator('#sourceMode').inputValue(),'voiceless');
   await frame.locator('#tab-organs').click();
   await frame.locator('#presets [data-preset="a"]').click();await settled();
   assert.equal(await frame.locator('#sourceMode').inputValue(),'voiced');
   assert.equal(Number(await frame.locator('#f0Range').inputValue()),speaker==='W02'?200:120);
   check(speaker+' actual preset buttons restore voiced vowel and model F0');
   const pose={params:meta.presets.a,lip_width:1,f0:meta.default_f0,duration:.3,frame_id:'qa'},perturbation={jitter:0,shimmer:0,seed:23};
   const dry=await pcm({frames:[pose],perturbation});
   for(const [parameter,value] of [['f0',300],['pressure_pa',1200],['lip_width',1.4]]){
    const wet=await pcm({frames:[pose],perturbation,features:[{id:parameter,enabled:true,parameter,mode:'fixed',value,selection:{kind:'time',start:.1,end:.2}}]});
    assert.deepEqual(wet.y.slice(0,4800),dry.y.slice(0,4800));assert(wet.y.slice(4800,9600).some((x,i)=>x!==dry.y[4800+i]));
   }
   check(speaker+' real Wasm audition PCM is unchanged before all three regional boundaries');
   const controlled=await call('animation/prepare',{frames:[pose],perturbation,global_controls:{f0:{mode:'fixed',value:240}},features:[{id:'pitch',enabled:true,parameter:'f0',mode:'fixed',value:280,selection:{kind:'time',start:.1,end:.2}}]});
   assert.equal(controlled.timeline.find(p=>p.time===.1).f0,280);assert.equal(controlled.timeline.find(p=>p.time===.2).f0,240);
   assert.equal((await call('animation/picture',{id:controlled.id,index:3})).state.f0,280);
   check(speaker+' composed global/regional F0 timeline agrees with actual picture state');
   await call('live',{active:true});await call('deactivate');
   await call('animation/prepare',{frames:[{...pose,params:meta.presets.i,manual_root:true}],perturbation});
   await call('live',{active:true});await call('deactivate');
   check(speaker+' manual-root sequence preparation followed by live start remains usable');
   evidence[speaker]={execution:meta.execution,regional_prefix_samples:4800,timeline_points:controlled.timeline.length};
  }
  const probe=await page.evaluate(()=>new Promise((resolve,reject)=>{const worker=new Worker('./wasm-qa/vocal-wasm-probe.worker.js');const timer=setTimeout(()=>{worker.terminate();reject(Error('review ABI probe timed out'));},120000);worker.onmessage=e=>{clearTimeout(timer);worker.terminate();e.data.error?reject(Error(e.data.error)):resolve(e.data);};worker.onerror=e=>{clearTimeout(timer);worker.terminate();reject(Error(e.message));};worker.postMessage({recipe:'vocal_review_cases.py'});}));
  evidence.nativeState=JSON.parse(probe.result);assert(evidence.nativeState.every(p=>p.root_restored&&p.compensation_matches&&p.cancel_restored&&p.exception_restored&&p.keep_vowel_all_exits_restored&&p.stop_pose_adopted));check('isolated real Wasm ABI restores preprocessing modes on all exits and adopts stopped poses');
  const current=await call('meta'),motionFrames=[{params:current.presets.i,f0:220,lip_width:1.2,duration:1,manual_root:true,source:current.source_presets.voiced},{params:current.presets.u,f0:280,lip_width:1,duration:1,manual_root:true,source:current.source_presets.voiced}];
  const motion=await call('animation/prepare',{frames:motionFrames,keep_vowel:true});await call('animation/play',{id:motion.id});
  const stopDeadline=Date.now()+30000;while((await call('animation/frame',{id:motion.id})).time<=.25){assert(Date.now()<stopDeadline);await new Promise(r=>setTimeout(r,40));}
  const stopped=await call('animation/stop');assert(stopped.pose.f0>220&&stopped.pose.f0<280);assert.equal(stopped.pose.manual_root,true);
  evidence.stopped={f0:stopped.pose.f0,manual_root:stopped.pose.manual_root};check('real AudioContext stop sends the final clock and receives the interpolated pose');
  // Fault injection aborts exactly one real IndexedDB profile write. The file
  // picker, document validation, worker, rollback and subsequent retry stay real.
  await frame.locator('#tab-motion').click();await frame.locator('#captureFrame').click();await frame.waitForFunction(()=>document.querySelectorAll('.pose-card').length===1);
  await frame.locator('#tab-sound').click();await frame.locator('#jitterRange').evaluate(e=>{e.value='0';e.dispatchEvent(new Event('change',{bubbles:true}));});
  let previous;for(let n=0;n<100;n++){previous=await call('keyframes/load');if(previous.perturbation?.jitter===0)break;await new Promise(r=>setTimeout(r,100));}assert.equal(previous.perturbation.jitter,0);
  const incoming={...previous,frames:motionFrames.map((p,i)=>({...p,frame_id:'motion-'+i})),perturbation:{jitter:80,shimmer:70,seed:2}},doc=await call('document/create',incoming);
  const durable=()=>page.evaluate(()=>new Promise((resolve,reject)=>{const open=indexedDB.open('ptb-research-files-v1');open.onerror=()=>reject(open.error);open.onsuccess=()=>{const db=open.result,r=db.transaction('values').objectStore('values').getAll();r.onsuccess=()=>{db.close();resolve(r.result.filter(v=>v.key.endsWith('vocal-profile-v1')));};};}));
  const before=await durable();await page.evaluate(()=>{const original=IDBObjectStore.prototype.put;window.qaFailProfile=true;IDBObjectStore.prototype.put=function(value,...args){if(window.qaFailProfile&&value.key?.endsWith('vocal-profile-v1')){window.qaFailProfile=false;this.transaction.abort();return {};}return original.call(this,value,...args);};});
  await frame.locator('#tab-motion').click();
  const importFile=async()=>{const picker=page.waitForEvent('filechooser');await frame.locator('#importFrames').click();await(await picker).setFiles({name:'review.ptb-vocal.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(doc))});};
  await importFile();await frame.locator('#frameStatus').filter({hasText:'导入失败'}).waitFor();assert.equal(await page.evaluate(()=>qaFailProfile),false);assert.deepEqual(await durable(),before);assert.deepEqual(await call('keyframes/load'),previous);
  assert.deepEqual(await frame.evaluate(()=>['jitterRange','shimmerRange','perturbationSeed'].map(id=>+document.getElementById(id).value)),[0,0,1]);
  await importFile();await frame.locator('#frameStatus').filter({hasText:'已导入'}).waitFor();assert.deepEqual((await call('keyframes/load')).perturbation,incoming.perturbation);check('real file import and IndexedDB quota failure preserve complete state and allow retry');
  const prepared=await call('animation/prepare',{...incoming,keep_vowel:true});await frame.locator('#playFrames').click();await frame.waitForFunction(()=>+document.getElementById('motionProgress').value>.15);await frame.locator('#stopFrames').click();await frame.locator('#frameStatus').filter({hasText:'已停止，保留当前姿势'}).waitFor();
  const held=await call('animation/frame',{id:prepared.id});assert(held.time>.25&&held.time<2);assert.equal(Number(await frame.locator('#f0Range').inputValue()),Math.round(held.state.f0));assert.equal(await frame.locator('#f0Output').textContent(),Math.round(held.state.f0)+' Hz');assert.equal(held.state.manual_root,true);
  await frame.locator('#captureFrame').click();await frame.waitForFunction(()=>document.querySelectorAll('.pose-card').length===3);const captured=(await call('keyframes/load')).frames.at(-1);for(const key of ['params','manual_root','lip_width','tongue_shape','lip_shape','larynx_height','source'])assert.deepEqual(captured[key],held.state[key]);check('visible play/stop controls and recaptured pose agree exactly with the shared core');
  await frame.locator('#clearFrames').click();
  // Export through the visible controls. Capture the actual canvas label used by
  // WebCodecs and read back the resulting MP4; no alternate drawing function.
  await frame.locator('#tab-motion').click();await frame.locator('#captureFrame').click();await frame.waitForFunction(()=>document.querySelectorAll('.pose-card').length===1);
  await frame.locator('.global-articulation summary').click();await frame.locator('#globalParameter').selectOption('f0');await frame.locator('#addGlobalControl').click();await frame.getByLabel('基频调整方式',{exact:true}).selectOption('fixed');await frame.locator('#global-value-f0').fill('240');await frame.locator('#global-value-f0').press('Tab');
  const end=Date.now()+30000;while((await call('keyframes/load')).global_controls?.f0?.value!==240){assert(Date.now()<end);await new Promise(r=>setTimeout(r,100));}
  await frame.evaluate(()=>{window.qaVideoLabels=[];const original=CanvasRenderingContext2D.prototype.fillText;CanvasRenderingContext2D.prototype.fillText=function(text,...args){if(String(text).startsWith('F0 '))window.qaVideoLabels.push(String(text));return original.call(this,text,...args);};});
  await frame.locator('#exportVideo').click();const download=page.waitForEvent('download');await frame.locator('#videoStart').click();const file=await download;await frame.locator('#videoStatus').filter({hasText:'已保存'}).waitFor();await file.saveAs(path.join(scratch,'corrected-f0.mp4'));
  const labels=await frame.evaluate(()=>qaVideoLabels);assert(labels.length>0&&labels.every(label=>label==='F0 240 Hz'),JSON.stringify(labels));const video=await fs.readFile(path.join(scratch,'corrected-f0.mp4'));assert(video.subarray(4,8).toString()==='ftyp');
  const decoded=JSON.parse(execFileSync(path.join(process.cwd(),'.venv/m05/Scripts/python.exe'),['-E','-s','-B','-X','utf8',path.join(process.cwd(),'tests/support/decode_vocal_review_video.py'),path.join(scratch,'corrected-f0.mp4'),path.join(out,'video-frame.png')],{encoding:'utf8',windowsHide:true}));
  evidence.video={bytes:video.length,labels,...decoded};
  await frame.locator('#videoClose').click();check('actual MP4 export uses 240 Hz composed label instead of authored F0');
  const beforeDraft=await call('keyframes/load'),beforeDraftBytes=await durable();
  await page.evaluate(()=>{window.qaFailProfile=true;});
  await frame.locator('#captureFrame').click();await frame.locator('#frameStatus').filter({hasText:'保存失败'}).waitFor();
  assert.equal(await page.evaluate(()=>qaFailProfile),false);assert.deepEqual(await call('keyframes/load'),beforeDraft);
  const draftDownload=page.waitForEvent('download');await frame.locator('#exportFrames').click();
  const draftFile=await draftDownload;await draftFile.saveAs(path.join(scratch,'unsaved-draft.json'));
  await frame.locator('#frameStatus').filter({hasText:'已导出'}).waitFor();
  const exportedDraft=JSON.parse(await fs.readFile(path.join(scratch,'unsaved-draft.json'),'utf8'));
  assert.equal(exportedDraft.frames.length,beforeDraft.frames.length+1);
  for(const field of ['features','pitch_curve','pressure_curve','global_controls','perturbation'])assert.deepEqual(exportedDraft[field]??(field==='perturbation'?null:field==='global_controls'?{}:[]),beforeDraft[field]??(field==='perturbation'?null:field==='global_controls'?{}:[]));
  assert.deepEqual(await durable(),beforeDraftBytes);assert.deepEqual(await call('keyframes/load'),beforeDraft);
  check('real IndexedDB failure preserves persisted bytes while exporting the complete unsaved draft');
  await frame.locator('#tab-organs').click();await frame.locator('#presets [data-preset="a"]').click();await settled();
  for(const region of [1,2,3]){
   await frame.locator('[data-side-region="'+region+'"]').click();
   const section=await frame.evaluate(async()=>{const {viewer}=await import('./app.js'),s=viewer.state;return {slider:+document.getElementById('sliceRange').value,section:s.section,upper:s.upper,expectedUpper:s.airway_sections[s.section].upper,lower:s.lower,expectedLower:s.airway_sections[s.section].lower,label:document.getElementById('slicePosition').textContent,expectedLabel:s.centerline[s.section][2].toFixed(1)+' cm'};});
   assert.equal(section.slider,section.section);assert.deepEqual(section.upper,section.expectedUpper);assert.deepEqual(section.lower,section.expectedLower);assert.equal(section.label,section.expectedLabel);
  }
  check('real Chrome side-region buttons synchronize viewer section, profiles, slider and position label');
  await frame.locator('#savePreset').click();await frame.locator('#presetInputButton').click();await frame.locator('#presetInput').fill('b');await frame.locator('#presetInputSave').click();await frame.locator('#presetDialog').waitFor({state:'hidden'});
  await frame.locator('#tab-motion').click();await frame.locator('#clearFrames').click();
  await frame.locator('.ipa-sequence').evaluate(e=>e.open=true);await frame.locator('#sequenceIpa').fill('ba');await frame.locator('#ipaTiming').selectOption('uniform');await frame.locator('#ipaDuration').fill('3');await frame.locator('#generateIpa').click();
  await frame.waitForFunction(()=>document.querySelectorAll('.pose-card').length===2);
  const longStop=await call('keyframes/load');assert.deepEqual(longStop.frames.map(f=>f.duration),[3,3]);assert.deepEqual(longStop.frames[0].stop_timing,{closure:2.992,release:.008,aspiration:0,vot:-.03});
  const longClip=await call('animation/prepare',longStop);assert.equal(longClip.duration,6);assert.equal((await call('animation/audio',{id:longClip.id})).samples,288000);
  await frame.locator('.pose-card').first().locator('.stop-timing > summary').click();await frame.getByLabel('闭塞毫秒，姿势 1',{exact:true}).fill('2990');await frame.getByLabel('闭塞毫秒，姿势 1',{exact:true}).press('Tab');
  await frame.locator('.stop-landmarks').first().filter({hasText:'结束 2998 ms'}).waitFor();assert.equal((await call('keyframes/load')).frames[0].duration,2.998);
  check('real Wasm generates and renders six-second ba from three-second poses, retaining negative VOT and editable long closure');
  for(const theme of ['light','dark']){await page.locator('button[title=设置]').click();await page.locator('[data-preview-mode='+theme+']').click();await page.locator('#tab-M10').click();await frame.waitForFunction(theme=>document.documentElement.dataset.theme===theme,theme);await page.screenshot({path:path.join(out,'review-'+theme+'.png')});}
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(out,'review-fixes.json'),JSON.stringify({success:true,checks,evidence,errors},null,2));
 }catch(error){await page.screenshot({path:path.join(out,'failure.png')});await fs.writeFile(path.join(out,'failure.json'),JSON.stringify({error:String(error),errors,checks,state:frame?await frame.locator('body').innerText():null},null,2));throw error;}
 finally{await ctx.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
