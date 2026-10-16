// Actual browser -> actual isolated feedback backend. No public messages or mail.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {spawn,spawnSync}=require('node:child_process'),{createHash}=require('node:crypto');
const root=path.resolve(__dirname,'..');
async function main(){
 const out=path.join(root,'output/validation/website','feedback-web-'+Date.now()),scratch=path.join(out,'scratch');await fs.mkdir(scratch,{recursive:true});
 const write=(name,value)=>fs.writeFile(path.join(out,name),JSON.stringify(value,null,2)+'\n');
 const marker={schema:'ptb-test-run/1',owner_pid:process.pid,state:'active',started_at:new Date().toISOString(),recipe:'Real feedback API + loopback browser, mock mailbox, no external mail or production writes'};
 await write('run.json',marker);let fixture,browser,page;const report={success:false,checks:[],errors:[]};
 try{
  fixture=spawn(path.join(root,'.venv/m14/Scripts/python.exe'),['-X','utf8',path.join(root,'scripts/website_feedback_fixture.py'),'--scratch',scratch],{cwd:root,windowsHide:true,stdio:['ignore','pipe','pipe']});
  let stderr='';fixture.stderr.on('data',b=>stderr+=b.toString());
  const port=await new Promise((resolve,reject)=>{let data='';const timer=setTimeout(()=>reject(Error('Fixture startup timed out: '+stderr)),20000);fixture.stdout.on('data',b=>{data+=b.toString();if(data.includes('\n')){clearTimeout(timer);try{resolve(JSON.parse(data.split('\n')[0]).port);}catch(e){reject(e);}}});fixture.on('error',reject);fixture.on('exit',code=>{clearTimeout(timer);reject(Error('Fixture exited '+code+' '+stderr));});});
  const origin='http://127.0.0.1:'+port;
  const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
  browser=await chromium.launchPersistentContext(path.join(scratch,'profile'),{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,viewport:{width:1440,height:1100}});
  page=await browser.newPage();page.on('pageerror',e=>report.errors.push(e.message));
  await page.goto(origin+'/#feedback');await page.emulateMedia({reducedMotion:'reduce'});await page.evaluate(()=>document.fonts.ready);
  await page.locator('#feedback-email').fill('website-fixture@example.org');await page.locator('#feedback-code-send').click();await page.waitForFunction(()=>!document.querySelector('#feedback-code-row').hidden);
  const {code}=await (await page.request.get(origin+'/__fixture/code')).json();
  await page.locator('#feedback-code').fill(code==='000000'?'111111':'000000');await page.locator('#feedback-code-verify').click();await page.waitForFunction(()=>document.querySelector('#feedback-status').textContent.includes('不正确'));
  await page.locator('#feedback-code').fill(code);await page.locator('#feedback-code-verify').click();await page.waitForFunction(()=>!document.querySelector('#feedback-submit').disabled);
  assert.match(await page.locator('#feedback-email-label').textContent(),/website-fixture/);report.checks.push('Email verification uses real challenge, wrong code rejected, correct code opens conversation');
  await page.locator('#feedback-module').fill('网页验证');await page.locator('#feedback-text').fill('隔离测试便签：<script>alert(1)</script>');
  await page.locator('#feedback-images').setInputFiles({name:'pixel.png',mimeType:'image/png',buffer:Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jN6kAAAAASUVORK5CYII=','base64')});
  await page.waitForFunction(()=>document.querySelectorAll('#feedback-previews img').length===1&&!document.querySelector('#feedback-submit').disabled);
  let attempts=0;
  await page.route('**/api/v1/feedback/messages',async route=>{attempts++;if(attempts===1){const response=await route.fetch();assert.equal(response.status(),200);await route.fulfill({status:503,json:{detail:'simulated_lost_receipt'}});}else await route.continue();});
  await page.locator('#feedback-submit').click();await page.waitForFunction(()=>document.querySelector('#feedback-status').classList.contains('is-error'));
  assert.match(await page.locator('#feedback-text').inputValue(),/隔离测试/);assert.equal(await page.locator('#feedback-previews img').count(),1);
  await page.locator('#feedback-submit').click();await page.waitForFunction(()=>document.querySelector('#feedback-status').textContent.includes('便签已寄到'));
  let state=await (await page.request.get(origin+'/__fixture/state')).json();assert.equal(state.userMessages,1);assert.equal(state.attachments,1);
  assert.equal(await page.locator('#feedback-text').inputValue(),'');assert.equal(await page.locator('#feedback-previews img').count(),0);assert.match(await page.locator('#feedback-messages').innerText(),/<script>/);assert.equal(await page.locator('#feedback-messages script').count(),0);
  report.checks.push('Real message + validated attachment stored; lost receipt retains draft; retry produces one message; content rendered as text');
  await page.locator('#feedback-mail').uncheck();await page.waitForFunction(()=>document.querySelector('#feedback-status').textContent.includes('已关闭'));
  state=await (await page.request.get(origin+'/__fixture/state')).json();assert.equal(state.mailNotifications,false);
  await page.request.post(origin+'/__fixture/reply');await page.locator('#feedback-refresh').click();await page.waitForFunction(()=>document.querySelector('.from-author'));
  assert.match(await page.locator('.from-author').innerText(),/收到啦/);assert.equal(await page.locator('.from-author img').count(),0);
  await page.reload();await page.waitForFunction(()=>!document.querySelector('#feedback-submit').disabled);assert.equal(await page.locator('#feedback-mail').isChecked(),false);
  await page.locator('#feedback-history').evaluate(e=>e.open=true);assert.equal(await page.locator('.feedback-message').count(),2);
  report.checks.push('Reply visible without email notifications; preference and conversation survive tab reload');
  await page.locator('#feedback').screenshot({path:path.join(out,'desktop-conversation.png'),animations:'disabled'});
  const isolated=await browser.browser().newContext();const blank=await isolated.newPage();await blank.goto(origin+'/#feedback');assert(await blank.locator('#feedback-submit').isDisabled());assert.equal(await blank.locator('.feedback-message').count(),0);await isolated.close();
  await page.locator('#feedback-switch').click();assert(await page.locator('#feedback-submit').isDisabled());assert.equal(await page.locator('.feedback-message').count(),0);
  report.checks.push('Fresh context and switched email cannot read or send under previous identity');
  // A second snapshot exists only in this intercepted test document, never on the site.
  await page.route('**/?version-fixture',async route=>{
   const response=await route.fetch();let html=await response.text();
   html=html.replace('</div></div><article class="release-paper reveal" role="tabpanel"','<button type="button" role="tab" data-release-tab="fixture-release" aria-controls="fixture-release" aria-selected="false" tabindex="-1">旧版测试</button></div></div><article class="release-paper reveal" role="tabpanel"');
   html=html.replace('</article></section>','</article><article id="fixture-release" role="tabpanel" hidden>隔离旧版内容</article></section>');
   await route.fulfill({response,body:html});
  });
  await page.goto(origin+'/?version-fixture');await page.locator('[data-release-tab="fixture-release"]').click();
  assert(await page.locator('#fixture-release').isVisible());assert(await page.locator('#release-panel-0').isHidden());
  await page.keyboard.press('ArrowLeft');assert(await page.locator('#release-panel-0').isVisible());assert(await page.locator('#fixture-release').isHidden());
  report.checks.push('Version tabs switch one body at a time by mouse and keyboard, using a test-only second snapshot');
  await page.goto(origin+'/#feedback');
  for(const width of [1440,390,320]){
   await page.setViewportSize({width,height:1000});await page.locator('#feedback').scrollIntoViewIfNeeded();
   assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
   for(const id of ['feedback','contact','lab']){await page.locator('#'+id).scrollIntoViewIfNeeded();await page.locator('#'+id).screenshot({path:path.join(out,id+'-'+width+'.png'),animations:'disabled'});}
  }
  assert.deepEqual(report.errors,[]);report.success=true;
 }catch(error){report.failure=String(error);if(page)await page.screenshot({path:path.join(out,'failure.png')}).catch(()=>{});throw error;}
 finally{
  await browser?.close();if(fixture&&fixture.exitCode===null){fixture.kill();await new Promise(resolve=>fixture.once('exit',resolve));}
  await write('report.json',report);
  const files=[];async function inventory(dir){for(const entry of await fs.readdir(dir,{withFileTypes:true})){assert(!entry.isSymbolicLink());const file=path.join(dir,entry.name);if(entry.isDirectory())await inventory(file);else{const b=await fs.readFile(file);files.push({path:path.relative(scratch,file),bytes:b.length,sha256:createHash('sha256').update(b).digest('hex')});}}}await inventory(scratch);
  await write('scratch-manifest.json',{schema:'ptb-test-scratch-files/1',files});await write('run.json',{...marker,state:'ready',test_outcome:report.success?'completed':'failed'});
  const cleanup=spawnSync('powershell.exe',['-NoProfile','-ExecutionPolicy','Bypass','-File',path.join(root,'scripts/cleanup_test_scratch.ps1'),'-RunDirectory',out],{windowsHide:true,encoding:'utf8'});assert.equal(cleanup.status,0,cleanup.stdout+cleanup.stderr);console.log(JSON.stringify({out,...report}));
 }
}
main().catch(error=>{console.error(error);process.exitCode=1});
