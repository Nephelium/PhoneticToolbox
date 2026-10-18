// Real UI and Wasm, owned loopback service and Chrome profile.
const fs=require('node:fs/promises'),path=require('node:path'),os=require('node:os'),assert=require('node:assert/strict');
const {chromium}=require(path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
const [origin,out,scratch]=process.argv.slice(2);
(async()=>{
 const ctx=await chromium.launchPersistentContext(path.join(scratch,'chrome'),{channel:'chrome',headless:true,viewport:{width:1600,height:1050},acceptDownloads:true,args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
 const page=await ctx.newPage(),errors=[],checks=[],evidence={};let frame;page.setDefaultTimeout(45000);
 page.on('pageerror',e=>errors.push(e.message));
 await page.addInitScript(()=>{window.qaSound=[];window.qaOps=[];window.qaSaves=[];const start=AudioBufferSourceNode.prototype.start;AudioBufferSourceNode.prototype.start=function(...args){const y=this.buffer?.getChannelData(0),tail=y?.subarray(Math.floor(y.length/2));window.qaSound.push({samples:this.buffer?.length,state:this.context.state,rms:tail?Math.sqrt(tail.reduce((sum,v)=>sum+v*v,0)/tail.length):0});return start.apply(this,args);};const post=Worker.prototype.postMessage;Worker.prototype.postMessage=function(value,...args){if(value?.op)window.qaOps.push(value.op);if(value?.op==='keyframes')window.qaSaves.push(value.body.global_controls);return post.call(this,value,...args);};});
 const fields=JSON.parse(await fs.readFile(process.env.PTB_VOCAL_QA_FIELDS,'utf8')),saved=new Map();
 await page.route('**/api/v1/preferences**',async route=>{const module=new URL(route.request().url()).pathname.split('/')[4];if(!module)return route.fulfill({json:{modules:[...saved.values()],allowed_fields:fields}});if(route.request().method()==='PATCH'){const value=route.request().postDataJSON();saved.set(module,{module,revision:(saved.get(module)?.revision||0)+1,values:{...saved.get(module)?.values,...value.values}});}return route.fulfill({json:saved.get(module)||{module,revision:0,values:{}}});});
 const check=name=>{checks.push(name);console.log('PASS',name);};
 // Run bridge polling in the page's main world: importing platform.js in
 // Playwright's isolated waitForFunction world creates a second request-ID counter.
 const waitState=async(fn,arg)=>{const end=Date.now()+45000;while(Date.now()<end){if(await frame.evaluate(fn,arg))return;await new Promise(r=>setTimeout(r,100));}throw Error('Saved state did not settle');};
 const settled=()=>frame.waitForFunction(()=>document.body.dataset.posePending==='false');
 const call=(op,body={})=>frame.evaluate(async([op,body])=>(await import('./platform.js')).request(op,body),[op,body]);
 const snap=()=>frame.evaluate(async()=>{const {viewer}=await import('./app.js');return {shape:viewer.state.tongue_shape,lip:viewer.state.lip_shape,params:viewer.state.params,limited:viewer.state.limited,handles:Object.fromEntries(viewer.handles().map(h=>[h.name,h.p]))};});
 const drag=async(name,dx,dy)=>{const node=frame.locator('[data-handle="'+name+'"]'),p=await node.evaluate(e=>{const t=e.getScreenCTM();return {x:t.e,y:t.f};}),box=await page.locator('iframe[title="生理参数合成"]').boundingBox();await page.mouse.move(p.x+box.x,p.y+box.y);await page.mouse.down();await page.mouse.move(p.x+box.x+dx,p.y+box.y+dy,{steps:8});await page.mouse.up();await settled();};
 try{
  await page.goto(origin+'/app/');await page.getByLabel('登录名',{exact:true}).fill('vocal-qa');await page.getByLabel('密码',{exact:true}).fill('vocal-qa-local-password');await page.getByRole('button',{name:'登录',exact:true}).click();await page.locator('#tool-sidebar').getByRole('button',{name:'生理参数合成',exact:true}).click();
  frame=await(await page.locator('iframe[title="生理参数合成"]').elementHandle()).contentFrame();await frame.waitForFunction(()=>['ready','failed'].includes(document.body?.dataset.engineState));assert.equal(await frame.locator('body').getAttribute('data-engine-state'),'ready',await frame.locator('#loadIndicator').innerText());await settled();
  evidence.meta=await call('meta');assert.equal(evidence.meta.execution,'browser-wasm');assert.equal(Number(await frame.locator('#f0Range').inputValue()),120);

  assert.equal(await frame.locator('#captureAspiration').count(),0);
  await frame.locator('#tab-motion').click();await frame.locator('#sequenceIpa').fill('ta');await frame.locator('#generateIpa').click();await frame.waitForFunction(()=>document.querySelectorAll('.pose-card').length===2);await frame.locator('.pose-card').first().locator('.stop-timing > summary').click();
  const edit=async(title,value)=>{const input=frame.getByLabel(title+'毫秒，姿势 1',{exact:true});await input.fill(String(value));await input.press('Tab');await call('keyframes/load');};
  await edit('闭塞',150);await edit('释放',8);await edit('VOT',-100);
  let seq=await call('keyframes/load');assert.equal(seq.frames[0].stop_timing.aspiration,0);assert(Math.abs(seq.frames[0].duration-.158)<1e-12);assert.equal(seq.frames[0].source.vibration,1);
  assert.equal(await frame.getByLabel('送气毫秒，姿势 1',{exact:true}).inputValue(),'0');assert((await frame.locator('.stop-landmarks').first().innerText()).includes('浊音起始 50 ms'));
  for(const selector of ['.sequence-help','.stop-help']){const help=frame.locator(selector).first();assert.equal(await help.evaluate(e=>e.open),false);await help.locator(':scope > summary').click();assert(await help.locator('p').isVisible());await help.locator(':scope > summary').click();assert.equal(await help.evaluate(e=>e.open),false);}
  assert.equal(await frame.locator('.stop-help > summary').first().evaluate(e=>getComputedStyle(e,'::after').content),'"+"');assert(!(await frame.locator('#panel-motion').textContent()).includes('起声'));check('timing and editing explanations default collapsed and expand; phonetic terminology displayed');
  const render=async()=>{const seq=await call('keyframes/load'),clip=await call('animation/prepare',seq),audio=await call('animation/audio',{id:clip.id}),buffer=Buffer.from(audio.base64,'base64');return {seq,clip,y:new Float32Array(buffer.buffer.slice(buffer.byteOffset,buffer.byteOffset+buffer.byteLength)),sr:audio.sample_rate};};
  const negative=await render();assert.equal(negative.y.length,17184);const rms=(y,a,b)=>Math.sqrt(y.slice(Math.round(a*48000),Math.round(b*48000)).reduce((s,v)=>s+v*v,0)/(Math.round((b-a)*48000)));
  const x=negative.y.slice(3600,6720),lag=Math.round(negative.sr/negative.seq.frames[0].f0);let aa=0,bb=0,ab=0;for(let i=0;i<x.length-lag;i++){aa+=x[i]*x[i];bb+=x[i+lag]*x[i+lag];ab+=x[i]*x[i+lag];}const periodic=ab/Math.sqrt(aa*bb);assert(periodic>.8);
  const picture=(await call('animation/picture',{id:negative.clip.id,index:3})).state;assert.equal(picture.stop_phase,'闭塞前浊');assert.equal(picture.source.vibration,1);
  check('negative VOT edits reach real Wasm prevoicing PCM, closed articulation and exact 358 ms length');
  await edit('VOT',50);let positive=await render();assert.equal(positive.y.length,19200);assert.equal(positive.seq.frames[0].stop_timing.aspiration,.042);assert(rms(negative.y,.075,.14)>10*rms(positive.y,.075,.14));evidence.prevoice={periodic,negative_rms:rms(negative.y,.075,.14),positive_rms:rms(positive.y,.075,.14)};
  await edit('送气',70);await edit('释放',20);seq=await call('keyframes/load');assert(Math.abs(seq.frames[0].stop_timing.vot-.078)<1e-12);assert(Math.abs(seq.frames[0].stop_timing.aspiration-.058)<1e-12);assert.equal((await render()).y.length,20544);
  await edit('VOT',0);assert.equal((await render()).y.length,17760);await edit('VOT',-40);seq=await call('keyframes/load');assert.equal(seq.frames[0].stop_timing.aspiration,0);assert.equal((await render()).y.length,17760);
  await edit('闭塞',30);assert((await frame.locator('.stop-error').first().innerText()).includes('前浊'));assert.equal((await call('keyframes/load')).frames[0].stop_timing.closure,.15);
  check('positive/zero/negative VOT and release/aspiration edits share one timeline; invalid shortening leaves saved timing intact');
  const download=page.waitForEvent('download');await frame.locator('#exportFrames').click();const doc=JSON.parse(await fs.readFile(await(await download).path(),'utf8'));assert.equal(doc.stop_timing_method,'m10/stop-timing/2');
  const legacy={...doc,stop_timing_method:'m10/stop-timing/1'},validated=await call('document/validate',legacy);assert(validated.geometry_notice.includes('前浊'));assert.deepEqual(validated.frames,doc.frames);check('current method export and old method import preserve authored durations with recomputation notice');
  await frame.locator('#playFrames').click();await frame.locator('#frameStatus').filter({hasText:'播放结束'}).waitFor();assert((await page.evaluate(()=>qaSound)).some(p=>p.samples>0&&p.state==='running'));check('actual browser playback completes edited sequence');
  for(const theme of ['light','dark']){await page.locator('button[title=设置]').click();await page.locator('[data-preview-mode='+theme+']').click();await page.locator('#tab-M10').click();await frame.waitForFunction(theme=>document.documentElement.dataset.theme===theme,theme);await page.screenshot({path:path.join(out,'stop-timing-'+theme+'.png')});}
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(out,'stop-timing.json'),JSON.stringify({success:true,checks,evidence,errors},null,2));
 }catch(error){await page.screenshot({path:path.join(out,'failure.png')});await fs.writeFile(path.join(out,'failure.json'),JSON.stringify({error:String(error),errors,checks,state:frame?await frame.locator('body').innerText():null},null,2));throw error;}
 finally{await ctx.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
