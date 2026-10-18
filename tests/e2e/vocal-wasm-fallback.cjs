/* Actual missing-runtime recovery, using only the owned loopback QA server. */
const fs=require('node:fs/promises'),path=require('node:path'),os=require('node:os'),assert=require('node:assert/strict');
const {chromium}=require(path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
const [origin,out,scratch]=process.argv.slice(2);
(async()=>{
 const context=await chromium.launchPersistentContext(path.join(scratch,'fallback-chrome'),{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,viewport:{width:1600,height:1050},args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
 const page=await context.newPage(),operations=[],errors=[];page.setDefaultTimeout(90000);page.on('pageerror',e=>errors.push(e.message));
 const allowed=JSON.parse(await fs.readFile(process.env.PTB_VOCAL_QA_FIELDS,'utf8')),preferences=new Map();
 await page.route('**/api/v1/preferences**',async route=>{const module=new URL(route.request().url()).pathname.split('/')[4];if(!module)return route.fulfill({json:{modules:[...preferences.values()],allowed_fields:allowed}});const body=route.request().postDataJSON(),value={module,revision:(preferences.get(module)?.revision??0)+1,values:body.changes,updated_at:new Date().toISOString()};preferences.set(module,value);await route.fulfill({json:value});});
 await page.route('**/app/vocal-wasm/**',route=>route.abort('failed'));
 page.on('request',r=>{if(r.url().includes('/vocal/sessions')&&r.method()==='POST'&&r.postDataJSON()?.op)operations.push(r.postDataJSON().op);});
 try{
  await page.goto(origin+'/app/');await page.getByLabel('登录名').fill('vocal-qa');await page.getByLabel('密码',{exact:true}).fill('vocal-qa-local-password');await page.getByRole('button',{name:'登录',exact:true}).click();await page.locator('#tool-sidebar').waitFor();await page.locator('#tool-sidebar').getByRole('button',{name:'生理参数合成',exact:true}).click();
  const iframe=page.locator('iframe[title="生理参数合成"]');await iframe.waitFor();const frame=page.frames().find(f=>f.url().includes('vocal-tract/index.html'))||await page.waitForEvent('framenavigated',{predicate:f=>f.url().includes('vocal-tract/index.html')});await frame.waitForFunction(()=>document.body.dataset.engineState==='ready'&&document.body.dataset.posePending==='false');
  if(await page.getByRole('button',{name:'知道了',exact:true}).isVisible().catch(()=>false))await page.getByRole('button',{name:'知道了',exact:true}).click();
  const meta=await frame.evaluate(async()=>{const {request}=await import('./platform.js');return request('meta');});assert.notEqual(meta.execution,'browser-wasm');assert.equal(meta.section_count,33);
  await frame.locator('#tab-sound').click();await frame.locator('#oneSecondButton').click();await frame.waitForFunction(()=>document.body.dataset.vowelBusy==='true');await frame.waitForFunction(()=>document.body.dataset.vowelBusy==='false');
  const status=await frame.evaluate(async()=>{const {request}=await import('./platform.js');return request('status');});assert.equal(status.audio.generated_samples,24000);assert.equal(status.audio.underflows,0);assert(operations.includes('preview'));assert(operations.includes('audio/pull'));assert.deepEqual(errors,[]);
  await fs.writeFile(path.join(out,'fallback.json'),JSON.stringify({success:true,missing_runtime:true,server_operations:operations,status,errors},null,2));console.log('actual unavailable Wasm runtime -> original native server -> complete 24000-sample browser playback');
 }finally{await context.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
