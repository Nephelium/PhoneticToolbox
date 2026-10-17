const fs=require('node:fs/promises'),path=require('node:path'),os=require('node:os'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'..'),out=path.join(root,'output/validation/web-launch/recording'),origin='http://127.0.0.1:18765',build=path.join(root,'output/validation/web-launch/recording-shared-final');
(async()=>{
 await fs.mkdir(out,{recursive:true});const account=JSON.parse(await fs.readFile(path.join(root,'output/validation/web-launch/isolated/client/module-qa.json'),'utf8'));
 const {chromium}=require(path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const context=await chromium.launchPersistentContext(path.join(out,'browser'),{headless:true,executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',viewport:{width:1440,height:1000},args:['--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream']});
 const page=await context.newPage(),errors=[],checks=[];let session;page.setDefaultTimeout(30000);page.on('pageerror',e=>errors.push(e.message));
 if(!process.env.PTB_TEST_DEPLOYED)await page.route(origin+'/app/**',async route=>{const relative=decodeURIComponent(new URL(route.request().url()).pathname.slice(5))||'index.html',file=path.resolve(build,relative);assert(file.startsWith(build+path.sep));await route.fulfill({path:file});});
 try{
  await page.goto(origin+'/app/',{timeout:120000});await page.getByLabel('登录名').fill(account.username);await page.getByLabel('密码',{exact:true}).fill(account.password);await page.getByRole('button',{name:'登录',exact:true}).click();await page.locator('#tool-sidebar').waitFor();session=await(await context.request.get(origin+'/api/v1/auth/me')).json();
  const api=async route=>(await context.request.get(origin+'/api/v1/'+route,{headers:{'X-PTB-Account':session.user.id}})).json(),project=(await api('projects')).projects[0].id;
  const open=async name=>page.locator('#tool-sidebar').getByRole('button',{name,exact:true}).click();
  await open('录音');await page.getByRole('button',{name:'设备与录前检测',exact:true}).click();await page.getByLabel('同声卡输入通道数').fill('2');
  await page.evaluate(()=>{window.qaMedia=navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);navigator.mediaDevices.getUserMedia=async()=>{throw new DOMException('测试拒绝麦克风','NotAllowedError');};});
  await page.getByRole('button',{name:'● 开始录音',exact:true}).click();await page.getByText('测试拒绝麦克风',{exact:true}).waitFor();await page.evaluate(()=>{navigator.mediaDevices.getUserMedia=window.qaMedia;delete window.qaMedia;});checks.push('permission rejection recovers');
  await page.getByRole('button',{name:'● 开始录音',exact:true}).click();await page.getByText('● 正在录音',{exact:true}).waitFor();await page.waitForTimeout(2200);
  let denied=false;await page.route('**/api/v1/uploads',async route=>{if(!denied){denied=true;await route.fulfill({status:503,contentType:'application/json',body:'{"detail":"synthetic unavailable"}'});}else await route.continue();});
  await page.getByRole('button',{name:'■ 停止录音',exact:true}).click();await page.getByText('录音保留在本页，可重试上传或下载 WAV。',{exact:true}).waitFor();
  const downloadPromise=page.waitForEvent('download');await page.getByRole('button',{name:'下载 WAV',exact:true}).click();const download=await downloadPromise;const bytes=await fs.readFile(await download.path());assert.equal(bytes.readUInt16LE(20),3);assert.equal(bytes.readUInt16LE(22),2);assert.equal(bytes.readUInt32LE(24),48000);assert(bytes.length>384044);assert(bytes.length<100000000);let peak=0;for(let n=44;n<bytes.length;n+=4)peak=Math.max(peak,Math.abs(bytes.readFloatLE(n)));assert(peak>0);await download.delete();checks.push('shared EXE page / AudioWorklet synthetic input / float WAV / download');
  await page.getByRole('button',{name:'重试上传',exact:true}).click();await page.getByText('本页录音已上传到当前项目。',{exact:true}).waitFor();const asset=(await api('assets?project_id='+project)).assets.find(a=>a.name.startsWith('录音-'));assert(asset&&asset.state==='ready');const serverBytes=await(await context.request.get(origin+'/api/v1/assets/'+asset.id+'/content',{headers:{'X-PTB-Account':session.user.id}})).body();assert.deepEqual(serverBytes,bytes);checks.push('upload failure retains WAV / retry Linux owned asset byte equality');
  assert(await page.locator('.recording-page .wave-line').count()>0);await page.getByRole('button',{name:'▶ 播放',exact:true}).click();await page.getByRole('button',{name:'■ 停止播放',exact:true}).waitFor();await page.getByRole('button',{name:'■ 停止播放',exact:true}).click();checks.push('shared waveform and playback');
  assert(await page.locator('.recording-page .module-seal-content[inert]').count()===2);await page.screenshot({path:path.join(out,'recording-shared.png')});checks.push('original task/device/edit controls retained; unavailable operations inert');
  for(const name of ['MFA 自动标注','语音学论文精读','生理参数合成']){
   await open(name);await page.locator('.module-seal-cover:visible').waitFor();
   if(name==='生理参数合成'){await page.frameLocator('iframe[title="生理参数合成"]').locator('body[data-engine-state="presentation"]').waitFor();assert.equal(await page.frameLocator('iframe[title="生理参数合成"]').locator('#p-TTX').getAttribute('max'),'7.5');}
   const underlay=page.locator('.module-seal:visible > .module-seal-content').first();assert.equal(await underlay.getAttribute('inert'),'');
   for(const theme of ['light','dark']){await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);await page.screenshot({path:path.join(out,name+'-'+theme+'.png')});}
   checks.push(name+' real layout / themed seal / inert');
  }
  await page.setViewportSize({width:390,height:844});assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);checks.push('seal mobile no document overflow');
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(out,process.env.PTB_TEST_DEPLOYED?'deployed.json':'local.json'),JSON.stringify({success:true,checks,errors},null,2));console.log(checks);
 }catch(e){await page.screenshot({path:path.join(out,'failure.png')});await fs.writeFile(path.join(out,'failure.txt'),await page.locator('body').innerText());throw e;}
 finally{if(session)await context.request.post(origin+'/api/v1/auth/logout',{headers:{Origin:origin,'X-CSRF-Token':session.csrf_token}}).catch(()=>{});await context.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
