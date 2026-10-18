/** Real hosted fallback, including saved controls that the old source cannot run. */
const fs=require('node:fs/promises'),path=require('node:path'),os=require('node:os'),assert=require('node:assert/strict');
const {chromium}=require(path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
const [origin,out,scratch]=process.argv.slice(2);
(async()=>{
 const ctx=await chromium.launchPersistentContext(path.join(scratch,'fallback-chrome'),{channel:'chrome',headless:true,viewport:{width:1600,height:1050},acceptDownloads:true,args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
 const page=await ctx.newPage(),errors=[],operations=[],checks=[];let frame;page.setDefaultTimeout(120000);
 page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(r.url().includes('/vocal/sessions/')&&r.method()==='POST')operations.push(r.postDataJSON().op);});
 await page.route('**/app/vocal-wasm/**',r=>r.abort('failed'));
 try{
  await page.goto(origin+'/app/');await page.getByLabel('登录名',{exact:true}).fill(process.env.PTB_M10_QA_USERNAME);await page.getByLabel('密码',{exact:true}).fill(process.env.PTB_M10_QA_PASSWORD);await page.getByRole('button',{name:'登录',exact:true}).click();await page.locator('#tool-sidebar').waitFor();
  const me=await(await ctx.request.get(origin+'/api/v1/auth/me')).json(),headers={Origin:origin,'X-CSRF-Token':me.csrf_token};
  const saved=await(await ctx.request.get(origin+'/api/v1/preferences')).json(),prior=saved.modules.find(m=>m.module==='M10');
  const setting={jitter:23,shimmer:30,seed:17},patch=await ctx.request.patch(origin+'/api/v1/preferences/M10',{headers,data:{expected_revision:prior?.revision??0,changes:{frame:{perturbation:setting}}}});assert.equal(patch.status(),200);
  await page.reload();await page.locator('#tool-sidebar').waitFor();await page.locator('#tool-sidebar').getByRole('button',{name:'生理参数合成',exact:true}).click();
  frame=await(await page.locator('iframe[title="生理参数合成"]').elementHandle()).contentFrame();const wait=fn=>frame.waitForFunction(fn,undefined,{polling:100});await wait(()=>document.body.dataset.engineState==='ready'&&document.body.dataset.posePending==='false');
  const notice=page.getByRole('button',{name:'知道了',exact:true});if(await notice.isVisible())await notice.click();
  const call=(op,body={})=>frame.evaluate(async([op,body])=>(await import('./platform.js')).request(op,body),[op,body]);
  const meta=await call('meta');assert.notEqual(meta.execution,'browser-wasm');assert.equal(meta.section_count,33);assert.equal(meta.max_animation_duration,5);
  await frame.locator('#tab-sound').click();assert(await frame.locator('#oneSecondButton').isDisabled());assert(await frame.locator('#liveButton').isDisabled());assert.match(await frame.locator('#perturbationHint').innerText(),/服务器声源不支持/);
  await frame.locator('#tab-motion').click();await frame.locator('#captureFrame').click();await wait(()=>document.querySelectorAll('.pose-card').length===1);assert(await frame.locator('#playFrames').isDisabled());
  const download=page.waitForEvent('download');await frame.locator('#exportFrames').click();const doc=JSON.parse(await fs.readFile(await(await download).path(),'utf8'));assert.equal(doc.version,6);assert.deepEqual(doc.perturbation,setting);checks.push('unsupported saved source is preserved and exportable; voice generation requires explicit recovery');
  await frame.locator('#tab-sound').click();await frame.locator('#perturbationReset').click();await wait(()=>!document.querySelector('#oneSecondButton').disabled);checks.push('explicit model-default action restores supported server source');
  const pings=[];let pending=Promise.resolve(),inflight=false;const ping=()=>{if(inflight)return;inflight=true;const start=Date.now();pending=ctx.request.get(origin+'/app/?vocal_availability_probe=1',{timeout:10000}).then(r=>pings.push({status:r.status(),ms:Date.now()-start})).finally(()=>inflight=false);};ping();const timer=setInterval(ping,300);
  try{await frame.locator('#oneSecondButton').click();await wait(()=>document.body.dataset.vowelBusy==='true');await wait(()=>document.body.dataset.vowelBusy==='false');}finally{clearInterval(timer);await pending;}
  const status=await call('status');assert.equal(status.audio.generated_samples,24000);assert.equal(status.audio.underflows,0);assert(pings.length&&pings.every(p=>p.status===200));assert.deepEqual(errors,[]);
  await page.screenshot({path:path.join(out,'fallback.png')});await fs.writeFile(path.join(out,'fallback.json'),JSON.stringify({success:true,checks,status,operations,entry_responses_during_native_preview:pings,errors},null,2));console.log('PASS real runtime failure, explicit source recovery, native 24000-sample audio and responsive HTTPS entry');
 }catch(e){await page.screenshot({path:path.join(out,'failure.png')});await fs.writeFile(path.join(out,'failure.json'),JSON.stringify({error:String(e),errors,operations,frame:frame?await frame.locator('body').innerText():null},null,2));throw e;}
 finally{try{const me=await ctx.request.get(origin+'/api/v1/auth/me');if(me.ok()){const s=await me.json();await ctx.request.post(origin+'/api/v1/auth/logout',{headers:{Origin:origin,'X-CSRF-Token':s.csrf_token}});}}finally{await ctx.close();}}
})().catch(e=>{console.error(e);process.exitCode=1;});
