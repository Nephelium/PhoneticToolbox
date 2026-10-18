/** Exercise the desktop section choices on the hosted Wasm engine. */
const fs=require('node:fs/promises'),path=require('node:path'),os=require('node:os'),assert=require('node:assert/strict');
const {chromium}=require(path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
const [origin,out,scratch]=process.argv.slice(2);
(async()=>{
 const ctx=await chromium.launchPersistentContext(path.join(scratch,'sections-chrome'),{channel:'chrome',headless:true,viewport:{width:1600,height:1050}});
 const page=await ctx.newPage(),errors=[],serverOps=[],sections=[];let frame;page.setDefaultTimeout(120000);
 page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(r.url().includes('/vocal/sessions/'))serverOps.push(r.url());});
 try{
  await page.goto(origin+'/app/');await page.getByLabel('登录名',{exact:true}).fill(process.env.PTB_M10_QA_USERNAME);await page.getByLabel('密码',{exact:true}).fill(process.env.PTB_M10_QA_PASSWORD);await page.getByRole('button',{name:'登录',exact:true}).click();await page.locator('#tool-sidebar').waitFor();
  await page.locator('#tool-sidebar').getByRole('button',{name:'生理参数合成',exact:true}).click();frame=await(await page.locator('iframe[title="生理参数合成"]').elementHandle()).contentFrame();
  await frame.waitForFunction(()=>document.body.dataset.engineState==='ready'&&document.body.dataset.posePending==='false',undefined,{polling:100});assert.equal(await frame.locator('body').getAttribute('data-execution'),'browser-wasm');
  await frame.evaluate(()=>{window.qaSectionSnapshots=[];addEventListener('message',e=>{if(e.source===parent&&e.data?.type==='m10-response'&&e.data.ok&&e.data.value?.centerline)window.qaSectionSnapshots.push(e.data.value);});});
  for(const count of [129,257,33,65]){
   await frame.locator('#sectionCount').selectOption(String(count));await frame.waitForFunction(n=>window.qaSectionSnapshots.at(-1)?.centerline.length===n&&document.body.dataset.posePending==='false',count,{polling:100});
   const state=await frame.evaluate(()=>window.qaSectionSnapshots.at(-1));assert.equal(state.section_count,count);assert.equal(state.airway_sections.length,count);assert(state.centerline.flat().every(Number.isFinite));assert(state.tube_areas.every(x=>Number.isFinite(x)&&x>=0));assert(state.tube_lengths.every(x=>Number.isFinite(x)&&x>0));sections.push({count,airway_sections:state.airway_sections.length,tube_sections:state.tube_areas.length,compute_ms:state.compute_ms});
  }
  assert.deepEqual(serverOps,[]);assert.deepEqual(errors,[]);await page.screenshot({path:path.join(out,'sections.png')});await fs.writeFile(path.join(out,'sections.json'),JSON.stringify({success:true,sections,serverOps,errors},null,2));console.log('PASS hosted browser 129/257/33/65 sections, finite geometry and no server science');
 }catch(e){await page.screenshot({path:path.join(out,'failure.png')});await fs.writeFile(path.join(out,'failure.json'),JSON.stringify({error:String(e),errors,sections},null,2));throw e;}
 finally{try{const me=await ctx.request.get(origin+'/api/v1/auth/me');if(me.ok()){const s=await me.json();await ctx.request.post(origin+'/api/v1/auth/logout',{headers:{Origin:origin,'X-CSRF-Token':s.csrf_token}});}}finally{await ctx.close();}}
})().catch(e=>{console.error(e);process.exitCode=1;});
