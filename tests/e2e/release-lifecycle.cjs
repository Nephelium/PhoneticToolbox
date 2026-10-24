// Real compiled ServerPage/AppShell UI. API identity and worker output are owned
// fixtures; IndexedDB, draft writes, buttons and downloaded bytes are real.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const {pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const config=JSON.parse(await fs.readFile(process.argv[2],'utf8')),out=config.out;
 const fixture=JSON.parse(await fs.readFile(config.fixture,'utf8'));
 const {preview}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 let server,context,page;const report={success:false,checks:[],errors:[],requests:[],unexpected:[],identity:config.identity};
 try{
  // Same preview fixture as m13-custom-tables.cjs, serving the supplied immutable
  // production dist. No product backend or database is started.
  server=await preview({root:path.join(root,'frontend'),build:{outDir:config.dist},preview:{host:'127.0.0.1',port:0}});
  context=await chromium.launchPersistentContext(path.join(config.scratch,'chrome-profile'),{executablePath:config.chrome||'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--mute-audio'],viewport:{width:1440,height:1000},acceptDownloads:true,downloadsPath:path.join(config.scratch,'downloads')});
  await context.addInitScript(({fixture})=>{
   const qa=window.__lifecycleQa={draftBlocked:false,resultBlocked:false,computes:0,blockedWrites:0,computedPayloads:[],delayPreview:false,previewHeld:false,previewPlays:0};
   const originalDecode=AudioContext.prototype.decodeAudioData;AudioContext.prototype.decodeAudioData=async function(...args){const buffer=await originalDecode.apply(this,args);if(qa.delayPreview){qa.previewHeld=true;await new Promise(resolve=>qa.releasePreview=resolve);qa.previewHeld=false;}return buffer;};
   document.addEventListener('play',event=>{if(event.target.closest?.('.perception-page'))qa.previewPlays++;},true);
   const originalSet=Storage.prototype.setItem;Storage.prototype.setItem=function(key,value){if(qa.draftBlocked&&key.startsWith('ptb.v3.mandarin-ipa.v1.'))throw new DOMException('Controlled owned draft storage failure','QuotaExceededError');return originalSet.call(this,key,value);};
   const originalPut=IDBObjectStore.prototype.put;IDBObjectStore.prototype.put=function(value,...args){if(qa.resultBlocked&&this.name==='files'&&/-(synthesis\.wav|m06\.ptb\.json|parameters\.csv)$/.test(value?.name??'')){qa.blockedWrites++;throw new DOMException('Controlled owned result storage failure','QuotaExceededError');}return originalPut.call(this,value,...args);};
   // The final production worker bridge is exercised with one native-generated
   // result. This checks retention and export, not scientific compute parity.
   const OriginalWorker=window.Worker;window.Worker=class extends EventTarget{
    constructor(url,options){super();if(!/science\.worker/.test(String(url)))return new OriginalWorker(url,options);this.originalUrl=String(url);this.options=options;}
    postMessage(message){if(message.operation!=='speech_synthesis'){this.onerror?.(new Event('error'));return;}qa.computes++;qa.computedPayloads.push(message.args);const output=fixture.outputs[message.args.action];if(!output){this.onmessage?.({data:{id:message.id,error:'Unexpected owned fixture action'}});return;}queueMicrotask(()=>this.onmessage?.({data:{id:message.id,result:structuredClone(output)}}));}
    terminate(){}
   };
  },{fixture});
  const owner='11111111-1111-4111-8111-111111111111',project='22222222-2222-4222-8222-222222222222',preferences=new Map();
  await context.route('**/api/v1/**',async route=>{
   const request=route.request(),url=new URL(request.url()),api=url.pathname.replace('/api/v1/','');report.requests.push({path:api,method:request.method()});let value;
   if(api==='auth/options')value={registration:false,password_recovery:false,ephemeral_workspaces:false};
   else if(api==='auth/me')value={user:{id:owner,username:'owned-lifecycle-fixture'},csrf_token:'owned-fixture-only',expires_at:4102444800};
   else if(api==='preferences')value={modules:[...preferences.values()],allowed_fields:fixture.allowed_fields};
   else if(api.startsWith('preferences/')&&request.method()==='PATCH'){const module=api.split('/')[1],body=request.postDataJSON(),previous=preferences.get(module);value={module,revision:(previous?.revision??0)+1,values:{...previous?.values,...body.changes},updated_at:new Date().toISOString()};preferences.set(module,value);}
   else if(api==='projects')value={projects:[{id:project,name:'独立验证工程',created_at:1}]};
   else if(api==='assets')value={assets:[]};
   else if(api==='jobs')value={jobs:[]};
   else if(api==='web-feedback')value={accepted:true};
   else if(api==='capabilities')value={task_operations:[]};
   else if(api==='storage/usage')value={policy_version:2,ready:true,frozen:false,used_bytes:0,reserved_bytes:0,quota_bytes:1000000000};
   else if(api.startsWith('feedback/'))value={};
   else{report.unexpected.push({api,method:request.method()});await route.fulfill({status:503,contentType:'application/json',body:JSON.stringify({detail:'owned_fixture_unknown_request'})});return;}
   await route.fulfill({status:200,contentType:'application/json',body:JSON.stringify(value)});
  });
  page=await context.newPage();page.on('pageerror',e=>report.errors.push(e.message));page.setDefaultTimeout(30000);
  // Keep the actual /app/ location while resolving the dist's relative assets.
  await context.route('**/app/assets/**',route=>route.continue({url:route.request().url().replace('/app/assets/','/assets/')}));
  await page.goto(server.resolvedUrls.local[0]+'app/');await page.locator('#tool-sidebar').waitFor();
  await page.locator('.nav-item[title="汉字转国际音标"]').click();const m13=page.locator('.mandarin-ipa-page');await m13.waitFor();
  const text='银行与中性保存验证';await m13.getByLabel('待转换汉字文本',{exact:true}).fill(text);
  await page.evaluate(()=>window.__lifecycleQa.draftBlocked=true);
  await page.getByRole('button',{name:'文件与空间',exact:true}).click();
  await m13.getByRole('alert').filter({hasText:'草稿保存失败'}).waitFor();assert.equal(await page.locator('.web-files-dialog').count(),0);assert.equal(await m13.getByLabel('待转换汉字文本',{exact:true}).inputValue(),text);
  await page.screenshot({path:path.join(out,'m13-save-failed.png'),fullPage:true});
  await page.evaluate(()=>window.__lifecycleQa.draftBlocked=false);await page.getByRole('button',{name:'文件与空间',exact:true}).click();
  const manager=page.getByRole('dialog',{name:'文件与空间',exact:true});await manager.waitFor();
  const stored=await page.evaluate(()=>Object.keys(localStorage).filter(k=>k.startsWith('ptb.v3.mandarin-ipa.v1.')).map(k=>JSON.parse(localStorage.getItem(k))));assert(stored.some(s=>s.text===text));
  await manager.getByRole('button',{name:'关闭对话框',exact:true}).click();
  report.checks.push('compiled actual ServerPage/AppShell: M13 failed draft prevents file-manager leave; current text survives; retry saves exact text then opens manager');

  await page.locator('.nav-item[title="声学参数合成"]').click();const m06=page.locator('.m06-page');await m06.waitFor();
  await m06.locator('input[accept=".csv,.json"]').setInputFiles({name:'owned-parameters.csv',mimeType:'text/csv',buffer:Buffer.from(fixture.parameters)});await page.getByRole('button',{name:'应用并覆盖',exact:true}).click();
  await m06.getByRole('button',{name:'保存参数草稿',exact:true}).click();await m06.getByText('完整参数草稿已保存',{exact:true}).waitFor();
  await page.evaluate(()=>window.__lifecycleQa.resultBlocked=true);await m06.getByRole('button',{name:'合成音频',exact:true}).click();
  await m06.getByRole('button',{name:'重试保存结果',exact:true}).waitFor();await page.waitForFunction(()=>document.querySelector('.m06-page')?.getAttribute('aria-busy')==='false');
  assert.equal(await page.evaluate(()=>window.__lifecycleQa.computes),1);assert(await m06.getByRole('button',{name:'导出音频',exact:true}).isDisabled());
  await m06.getByRole('button',{name:'生成元音',exact:true}).click();await m06.getByRole('alert').filter({hasText:'还有待保存的合成结果'}).waitFor();assert.equal(await page.evaluate(()=>window.__lifecycleQa.computes),1);
  await page.getByRole('button',{name:'关闭 声学参数合成',exact:true}).click();const closing=page.getByRole('dialog',{name:'保存参数草稿？',exact:true});await closing.waitFor();
  await closing.getByRole('button',{name:'保存结果与草稿并关闭',exact:true}).click();await closing.getByRole('alert').filter({hasText:'合成结果未保存'}).waitFor();assert(await m06.count());assert.equal(await page.evaluate(()=>window.__lifecycleQa.computes),1);
  await closing.getByRole('button',{name:'取消关闭',exact:true}).click();await page.screenshot({path:path.join(out,'m06-retained-result.png'),fullPage:true});
  await page.evaluate(()=>window.__lifecycleQa.resultBlocked=false);await m06.getByRole('button',{name:'重试保存结果',exact:true}).click();await m06.getByText(/^合成完成，/).waitFor();assert.equal(await page.evaluate(()=>window.__lifecycleQa.computes),1);
  const pending=page.waitForEvent('download');await m06.getByRole('button',{name:'导出音频',exact:true}).click();const download=await pending,bytes=await fs.readFile(await download.path());
  const digest=crypto.createHash('sha256').update(bytes).digest('hex');assert.equal(digest,fixture.wav_sha256);report.export={bytes:bytes.length,sha256:digest};
  await page.screenshot({path:path.join(out,'m06-retry-exported.png'),fullPage:true});
  report.checks.push('compiled actual AppShell M06: IndexedDB quota failure retains computed result; new task refused; save-close failure keeps tab; retry saves without recompute and downloads exact native-generated WAV SHA');

  await page.locator('.nav-item[title="感知实验"]').click();const m15=page.locator('.perception-page');await m15.waitFor();
  await m15.locator('[aria-label="导入刺激 X"]').setInputFiles({name:'owned-neutral.wav',mimeType:'audio/wav',buffer:Buffer.from(fixture.outputs.synthesize.files.find(f=>f.name==='synthesis.wav').data,'base64')});
  await m15.locator('.m15-asset').filter({hasText:'owned-neutral.wav'}).waitFor();await page.evaluate(()=>window.__lifecycleQa.delayPreview=true);
  await m15.locator('.m15-asset').filter({hasText:'owned-neutral.wav'}).getByRole('button',{name:'预览 owned-neutral.wav',exact:true}).click();await page.waitForFunction(()=>window.__lifecycleQa.previewHeld);
  await page.locator('#tab-home').click();await page.evaluate(()=>window.__lifecycleQa.releasePreview());await page.waitForFunction(()=>document.querySelector('.perception-page')?.getAttribute('aria-busy')==='false');
  assert.equal(await m15.locator('.m15-preview audio').count(),0);assert.equal(await page.evaluate(()=>window.__lifecycleQa.previewPlays),0);
  await page.locator('#tab-M15').click();assert.equal(await m15.locator('.m15-preview audio').count(),0);
  await page.evaluate(()=>window.__lifecycleQa.delayPreview=false);await m15.locator('.m15-asset').filter({hasText:'owned-neutral.wav'}).getByRole('button',{name:'预览 owned-neutral.wav',exact:true}).click();await m15.locator('.m15-preview audio').waitFor();
  await m15.getByRole('button',{name:'停止预览',exact:true}).click();assert.equal(await m15.locator('.m15-preview audio').count(),0);
  await page.screenshot({path:path.join(out,'m15-late-preview-discarded.png'),fullPage:true});
  report.checks.push('compiled actual AppShell M15: real browser decode held at completion, tab hidden before release; no late audio or play event, return stays stopped; new active preview and Stop still work (owned Chrome muted)');
  assert.deepEqual(report.errors,[]);assert.deepEqual(report.unexpected,[]);report.success=true;
 }catch(error){report.failure=String(error);if(page){report.body=await page.locator('body').innerText();await page.screenshot({path:path.join(out,'failure.png'),fullPage:true});}throw error;}
 finally{await fs.writeFile(path.join(out,'browser-report.json'),JSON.stringify(report,null,2)+'\n');if(context)await context.close();if(server)await new Promise(resolve=>server.httpServer.close(resolve));}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
