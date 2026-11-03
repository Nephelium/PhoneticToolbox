// Actual loopback HTTP, fresh PostgreSQL and isolated browser profiles.
// PTB_VERIFY_FRONTEND_DIST binds the surrounding fixture to the final frozen candidate.
const fs=require('node:fs/promises'),path=require('node:path'),os=require('node:os'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..');
async function main(){
 const [origin,out,scratch]=process.argv.slice(2);assert.equal(path.dirname(scratch),out);
 const {chromium}=require(path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const contexts=[],errors=[],checks=[],requests=[];
 const catalogs={};for(const code of ['zh-CN','en','ja']){
  catalogs[code]={};for(const ns of ['account','workbench','components','lpc-spectrum','vocal-tract'])Object.assign(catalogs[code],JSON.parse(await fs.readFile(path.join(root,'frontend/src/i18n/locales',code,ns+'.json'),'utf8')));
 }
 const lang=async page=>await page.locator('html').getAttribute('lang');
 const text=async(page,key)=>catalogs[await lang(page)][key];
 async function browser(name,edge=false){
  const context=await chromium.launchPersistentContext(path.join(scratch,name),{executablePath:edge?'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe':'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,viewport:{width:1920,height:1080},locale:'ja-JP'});contexts.push(context);
  await context.route('https://www.phonetictoolbox.com/announcements/v1.json',route=>route.fulfill({status:200,contentType:'application/json',body:JSON.stringify({schema:'ptb-announcements/1',items:[{id:'isolated-language-check',title:'Language QA notice',body:'测试公告原文保持。',publishedAt:'2026-10-01T00:00:00Z',platforms:['web']}]})}));
  await context.addInitScript(()=>{
   window.__ptbI18nQa={workerPosts:0};const original=Worker.prototype.postMessage;
   Worker.prototype.postMessage=function(...args){window.__ptbI18nQa.workerPosts++;return Reflect.apply(original,this,args);};
  });
  const page=await context.newPage();page.setDefaultTimeout(30000);page.on('pageerror',e=>errors.push(e.message));page.on('dialog',d=>void d.accept());
  page.on('request',r=>requests.push({browser:name,method:r.method(),path:new URL(r.url()).pathname}));
  await page.goto(origin+'/server/');await page.locator('input[autocomplete=username]').waitFor();return page;
 }
 async function locale(page,code,account=false){
  await page.waitForFunction(()=>!!document.querySelector('.language-switch select:not(:disabled)'));
  const changed=await page.locator('.language-switch select').first().inputValue()!==code;
  const wait=account&&changed?page.waitForResponse(r=>new URL(r.url()).pathname==='/api/v1/preferences/APP'&&r.request().method()==='PATCH'&&r.status()===200&&r.request().postDataJSON()?.changes?.uiLocale===code):Promise.resolve();
  await page.locator('.language-switch select').first().selectOption(code);await page.waitForFunction(c=>document.documentElement.lang===c,code);await page.waitForFunction(()=>!document.querySelector('.language-switch select:disabled'));await wait;
  await page.waitForFunction(c=>document.documentElement.dataset.fontLocale===c&&document.fonts.status==='loaded',code);
 }
 async function login(page,name='synthetic'){
  const unread=await page.evaluate(()=>!JSON.parse(localStorage.getItem('ptb.v3.announcements.seen.v1')??'[]').includes('isolated-language-check'));
  await page.locator('input[autocomplete=username]').fill(name);await page.locator('input[autocomplete=current-password]').fill('synthetic-password-only');
  await page.getByRole('button',{name:await text(page,'account.log_in'),exact:true}).click();await page.locator('#tool-sidebar').waitFor();
  const panel=page.locator('dialog[open] .announcements-panel');if(unread){await panel.waitFor({state:'visible'});assert((await panel.innerText()).includes('测试公告原文保持。'));await page.locator('dialog[open]').getByRole('button',{name:await text(page,'components.got_it'),exact:true}).click();}
  await page.waitForFunction(()=>!document.querySelector('.language-switch select:disabled'));
 }
 async function lpc(page){await page.locator('.nav-item[data-module-id=M04]').click();await page.locator('.lpc-page').waitFor({state:'visible'});}
 async function logout(page){
  await page.getByRole('button',{name:await text(page,'workbench.files_and_storage'),exact:true}).click();
  await page.getByRole('button',{name:await text(page,'account.log_out'),exact:true}).click();await page.locator('input[autocomplete=username]').waitFor();
 }
 async function preferences(page){return await page.evaluate(async()=>{const r=await fetch('/api/v1/preferences');if(!r.ok)throw Error(String(r.status));return (await r.json()).modules;});}
 try{
  const a=await browser('chrome-language-a');assert.equal(await lang(a),'zh-CN');await locale(a,'en');await login(a);assert.equal(await lang(a),'zh-CN');checks.push('new account starts Chinese despite anonymous English and Japanese browser locale');
  await lpc(a);const input=a.locator('.lpc-analysis-settings input[type=number]').first();await Promise.all([a.waitForResponse(r=>new URL(r.url()).pathname==='/api/v1/preferences/M04'&&r.request().method()==='PATCH'&&r.status()===200),input.fill('72')]);
  const bytes=Buffer.alloc(44+16000*2);bytes.write('RIFF',0);bytes.writeUInt32LE(bytes.length-8,4);bytes.write('WAVEfmt ',8);bytes.writeUInt32LE(16,16);bytes.writeUInt16LE(1,20);bytes.writeUInt16LE(1,22);bytes.writeUInt32LE(16000,24);bytes.writeUInt32LE(32000,28);bytes.writeUInt16LE(2,32);bytes.writeUInt16LE(16,34);bytes.write('data',36);bytes.writeUInt32LE(bytes.length-44,40);for(let i=0;i<16000;i++)bytes.writeInt16LE(Math.round(8000*Math.sin(2*Math.PI*220*i/16000)),44+i*2);
  const wav=path.join(scratch,'language-synthetic.wav');await fs.writeFile(wav,bytes);await a.locator('.lpc-page .research-upload input[type=file]').setInputFiles(wav);
  const files=a.locator('.lpc-files select').first();await a.waitForFunction(()=>document.querySelector('.lpc-files select')?.options.length===2);await files.selectOption({label:'language-synthetic.wav'});
  await a.waitForFunction(()=>document.querySelector('.lpc-files small')?.textContent.includes('16000 Hz'));const waveform=a.locator('.lpc-page .wave-viewport').first();await waveform.waitFor();
  await waveform.evaluate(e=>e.dataset.i18nProbe='same-loaded-waveform');await a.locator('.lpc-page').evaluate(e=>e.dataset.i18nProbe='same-module');
  const fileId=await files.inputValue(),scientificBefore=requests.filter(r=>r.browser==='chrome-language-a'&&r.method==='POST'&&/\/(?:jobs|uploads)(?:\/|$)/.test(r.path)).length,workerPosts=await a.evaluate(()=>window.__ptbI18nQa.workerPosts);
  for(const code of ['en','ja','zh-CN','en','ja']){
   await locale(a,code,true);assert.equal(await input.inputValue(),'72');assert.equal(await files.inputValue(),fileId);assert.equal(await waveform.getAttribute('data-i18n-probe'),'same-loaded-waveform');assert.equal(await a.locator('.lpc-page').getAttribute('data-i18n-probe'),'same-module');
  }
  assert.equal(requests.filter(r=>r.browser==='chrome-language-a'&&r.method==='POST'&&/\/(?:jobs|uploads)(?:\/|$)/.test(r.path)).length,scientificBefore);assert.equal(await a.evaluate(()=>window.__ptbI18nQa.workerPosts),workerPosts);checks.push('five language switches retain actual imported WAV, parameter draft, DOM and file identity without new uploads, scientific jobs or Worker posts');
  const cdp=await a.context().newCDPSession(a);await cdp.send('Performance.enable');
  async function counters(){await cdp.send('HeapProfiler.collectGarbage');const dom=await cdp.send('Memory.getDOMCounters'),metrics=await cdp.send('Performance.getMetrics');return {...dom,heapBytes:metrics.metrics.find(m=>m.name==='JSHeapUsedSize').value};}
  const before=await counters(),warmSwitchMs=[];
  for(let cycle=0;cycle<5;cycle++)for(const code of ['zh-CN','en','ja']){const start=performance.now();await locale(a,code,true);warmSwitchMs.push(performance.now()-start);}
  const after=await counters();assert.equal(await a.evaluate(()=>window.__ptbI18nQa.workerPosts),workerPosts);assert(after.nodes<=before.nodes+30,JSON.stringify({before,after}));assert(after.jsEventListeners<=before.jsEventListeners+2,JSON.stringify({before,after}));assert(after.heapBytes<=before.heapBytes+5_000_000,JSON.stringify({before,after}));
  const catalogLoads={};for(const code of ['en','ja']){catalogLoads[code]=requests.filter(r=>r.browser==='chrome-language-a'&&r.method==='GET'&&new RegExp('/assets/'+code+'-[^/]+\\.js$').test(r.path)).length;assert.equal(catalogLoads[code],1);}
  await fs.writeFile(path.join(out,'performance.json'),JSON.stringify({scope:'Owned Chrome headless integration; post-GC DOM/listener/heap counters after warm-up, not final EXE startup timings',before,after,workerPosts,catalogLoads,warmSwitchMs},null,2));checks.push('15 warm switches reuse language bundles and worker results without accumulating DOM nodes or listeners');
  for(const code of ['en','ja','zh-CN']){
   await locale(a,code,true);for(const width of [820,1280,1920]){
    await a.setViewportSize({width,height:1080});const layout=await a.evaluate(()=>({viewport:innerWidth,document:document.documentElement.scrollWidth,controls:[...document.querySelectorAll('.topbar .language-switch select,.topbar button')].filter(e=>e.offsetWidth).map(e=>{const r=e.getBoundingClientRect();return [r.left,r.right];})}));
    assert(layout.document<=layout.viewport+1,JSON.stringify(layout));for(const [left,right]of layout.controls)assert(left>=-1&&right<=width+1,JSON.stringify(layout));
   }await a.screenshot({path:path.join(out,'web-'+code+'.png')});
  }checks.push('three languages at 820, 1280 and 1920 pixels keep top controls accessible');
  const moduleSelectors={M01:'.m01-page',M02:'.m02-page',M03:'.egg-page',M04:'.lpc-page',M05:'.lip-page',M06:'.m06-page',M07:'.m07-page',M08:'.m08-page',M09:'.m09-page',M10:'.vocal-page',M11:'.mfa-page',M12:'.annotation-page',M13:'.mandarin-ipa-page',M14:'.phonology-page',M15:'.perception-page',M16:'.recording-page',M17:'.ipa-plus-page',M18:'.paper-reading',M19:'.language-atlas'};
  const moduleChecks=[];let vocalFrame;
  for(const code of ['zh-CN','en','ja']){
   await locale(a,code,true);
   for(const [id,selector]of Object.entries(moduleSelectors)){
    await a.locator('.nav-item[data-module-id='+id+']').click();await a.locator(selector).waitFor({state:'visible'});
    assert.equal(await a.locator('#tab-'+id).getAttribute('aria-selected'),'true');
    if(id==='M10'){
     const frame=a.frames().find(f=>f.url().includes('vocal-tract/index.html'));assert(frame);
     await frame.waitForFunction(()=>document.body.dataset.engineState==='ready'&&document.body.dataset.posePending==='false');
     await frame.waitForFunction(c=>document.documentElement.lang===c,code);
     assert((await frame.locator('body').innerText()).trim().length>0,'M10 actual iframe UI is populated');
     if(vocalFrame)assert.equal(frame,vocalFrame,'language changes preserve the existing live vocal frame');else vocalFrame=frame;
     const notice=a.getByRole('dialog',{name:catalogs[code]['modules.vocal-tract.physiological_parameter_synthesis_web'],exact:true});
     if(await notice.isVisible()){await notice.getByRole('button',{name:catalogs[code]['modules.vocal-tract.got_it'],exact:true}).click();await notice.waitFor({state:'detached'});checks.push('M10 first server fallback notice acknowledged through its real control in '+code);}
    }else assert((await a.locator(selector).innerText()).trim().length>0,id+' actual module UI is populated');
    moduleChecks.push({locale:code,module:id});
   }
  }
  await fs.writeFile(path.join(out,'module-navigation.json'),JSON.stringify({scope:'Real HTTP-backed workbench and actual mounted module controls; unavailable server capabilities retain guidance; not every scientific operation',checks:moduleChecks},null,2));
  checks.push('all 19 modules mount in each language through the actual authenticated Web shell');
  await locale(a,'ja',true);const b=await browser('edge-language-b',true);await login(b);assert.equal(await lang(b),'ja');await lpc(b);assert.equal(await b.locator('.lpc-analysis-settings input[type=number]').first().inputValue(),'72');
  assert.equal(await b.locator('.lpc-files select').first().locator('option').count(),1);checks.push('independent Edge profile restores account Japanese and order 72 without transferring the Chrome local WAV');
  await b.reload();await b.locator('#tool-sidebar').waitFor();assert.equal(await lang(b),'ja');checks.push('Japanese account preference survives full browser reload');
  await logout(a);assert.equal(await lang(a),'en');checks.push('logout restores anonymous device English');await login(a,'language-other');assert.equal(await lang(a),'zh-CN');await lpc(a);assert.equal(await a.locator('.lpc-analysis-settings input[type=number]').first().inputValue(),'50');assert.equal(await a.locator('.lpc-files select').first().locator('option').count(),1);await logout(a);checks.push('second account on reused profile has Chinese, default parameters and no previous files');
  const saved=await preferences(b);assert.equal(saved.find(x=>x.module==='APP').values.uiLocale,'ja');assert.equal(String(saved.find(x=>x.module==='M04').values.draft.order),'72');await logout(b);assert.deepEqual(errors,[]);
  await fs.writeFile(path.join(out,'report.json'),JSON.stringify({stage:process.env.PTB_VERIFY_FRONTEND_DIST?'frozen-Web-candidate':'authenticated-Web-integration',scope:'Actual Chrome and Edge engines; isolated profiles, new PostgreSQL and loopback HTTP; frontend identity recorded by surrounding fixture',explicitCandidate:process.env.PTB_VERIFY_FRONTEND_DIST??null,checks,errors,requests},null,2));console.log(JSON.stringify({out,checks,errors}));
 }catch(error){for(let i=0;i<contexts.length;i++){const page=contexts[i].pages().at(-1);if(page){await page.screenshot({path:path.join(out,'failed-'+i+'.png')});await fs.writeFile(path.join(out,'failed-'+i+'.txt'),await page.locator('body').innerText());}}await fs.writeFile(path.join(out,'failed.json'),JSON.stringify({error:String(error),checks,errors,requests},null,2));throw error;}
 finally{for(const context of contexts.reverse())await context.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
