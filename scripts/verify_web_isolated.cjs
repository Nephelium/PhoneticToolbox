// Real SSH-forwarded Linux API and the source Qt manager. No mocked transport.
const fs=require('node:fs/promises'),path=require('node:path'),os=require('node:os'),assert=require('node:assert/strict');
const {execFileSync}=require('node:child_process');
const root=path.resolve(__dirname,'..'),out=path.join(root,'output/validation/web-launch/isolated');
const browser=process.argv[2]||'chrome';assert(['chrome','edge'].includes(browser));
const pause=ms=>new Promise(r=>setTimeout(r,ms));
(async()=>{
 const config=JSON.parse(await fs.readFile(path.join(out,'client/client.json'),'utf8'));
 // Rebuild the small synthetic input; no research recordings are used.
 try{await fs.access(path.join(out,'synthetic.wav'));}catch{
  execFileSync(path.join(root,'.venv/m14/Scripts/python.exe'),['-B','-c',
   'import numpy as np; from scipy.io import wavfile; import sys; d=np.load(sys.argv[1]); wavfile.write(sys.argv[2],44100,np.column_stack([d["load.egg_signal_raw"],d["load.audio_signal"]]))',
   path.join(root,'tests/fixtures/m03/EGG-SYN-PCM16.npz'),path.join(out,'synthetic.wav')],{windowsHide:true});
 }
 const {chromium}=require(path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const context=await chromium.launchPersistentContext(path.join(out,'qa-'+browser),{headless:false,executablePath:browser==='edge'?'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe':'C:/Program Files/Google/Chrome/Application/chrome.exe',viewport:{width:1440,height:1000}});
 const page=await context.newPage(),errors=[],checks=[];page.on('pageerror',e=>errors.push(e.message));
 page.setDefaultTimeout(30000);
 page.on('pageerror',e=>console.log('PAGE_ERROR',e.message));
 page.on('requestfailed',r=>console.log('REQUEST_FAILED',r.url(),r.failure()?.errorText));
 const bridge=(op,args={})=>JSON.parse(execFileSync(path.join(root,'.venv/m14/Scripts/python.exe'),['-B',path.join(root,'scripts/verify_web_isolated_bridge.py')],{input:JSON.stringify({op,args}),encoding:'utf8',windowsHide:true}));
 try{
  const until=async(fn,timeout=60000)=>{const end=Date.now()+timeout;while(Date.now()<end){if(await fn())return;await pause(250);}throw Error('Timed out');};
  const prior=await context.request.get(config.web_origin+'/api/v1/auth/me');
  if(prior.ok()){const s=await prior.json();await context.request.post(config.web_origin+'/api/v1/auth/logout',{headers:{Origin:config.web_origin,'X-CSRF-Token':s.csrf_token}});}
  for(const old of bridge('webRequests').requests.filter(r=>r.username===config.test_account.username))bridge('webDecision',{request_id:old.id,granted_bytes:null,duration:'7d',note:'重跑隔离验证，结束上次未完成的合成申请。'});
  const loadStart=Date.now();await page.goto(config.web_origin+'/app/',{timeout:120000});
  console.log('APP_LOADED',browser,Date.now()-loadStart);
  await page.getByLabel('登录名').fill(config.test_account.username);await page.getByLabel('密码',{exact:true}).fill(config.test_account.password);
  await page.getByRole('button',{name:'登录',exact:true}).click();await page.locator('#tool-sidebar').waitFor();
  checks.push('app-entry-login-default-workbench');
  const session=await(await context.request.get(config.web_origin+'/api/v1/auth/me')).json();
  const api=async route=>{const r=await context.request.get(config.web_origin+'/api/v1/'+route,{headers:{'X-PTB-Account':session.user.id}});assert.equal(r.status(),200);return r.json();};
  await page.getByRole('button',{name:'文件与空间',exact:true}).click();await page.getByText('补充上传文件',{exact:true}).click();
  await page.getByLabel('选择上传文件').setInputFiles(path.join(out,'synthetic.wav'));
  await page.getByRole('button',{name:'上传文件',exact:true}).click();await page.locator('.storage-files strong').filter({hasText:'synthetic.wav'}).waitFor();
  await page.getByRole('dialog',{name:'文件与空间',exact:true}).getByRole('button',{name:'关闭对话框',exact:true}).click();
  await page.getByRole('button',{name:'容量与扩容',exact:true}).click();await page.getByLabel('申请总额度（MB）').fill('2000');await page.getByLabel('申请期限').selectOption('30d');
  await page.getByLabel('扩容用途').fill('隔离内测：合成音频验证，请批准7天。');await page.getByRole('button',{name:'提交扩容申请',exact:true}).click();
  await page.getByText('申请已提交，等待管理员审核。',{exact:true}).waitFor();
  const live=bridge('webStatus');assert.equal(typeof live.load.cpu_percent,'number');assert.equal(typeof live.load.memory_percent,'number');assert.equal(live.load.task_slots,1);
  const request=bridge('webRequests').requests.find(r=>r.username===config.test_account.username);assert(request);
  bridge('webDecision',{request_id:request.id,granted_bytes:1500000000,duration:'7d',note:'隔离内测已批准7天，请及时下载。'});
  await until(async()=>(await api('storage/usage')).quota_bytes===1500000000);
  await page.getByText(/管理员回复：隔离内测已批准7天/).waitFor({timeout:45000});
  assert.equal((await api('workspaces/capacity-request')).granted_duration,'7d');
  await page.screenshot({path:path.join(out,'web-approval.png')});
  await fs.writeFile(path.join(out,'live-status.json'),JSON.stringify(live,null,2));
  checks.push('browser-request-native-manager-real-linux-approval-7-days-reply-quota','live-cpu-memory-disk-online-queue');
  await page.getByRole('dialog').getByRole('button',{name:'关闭对话框',exact:true}).click();
  await page.locator('#tool-sidebar').getByRole('button',{name:/LPC/}).click();await page.getByLabel('LPC 音频文件').selectOption({label:'synthetic.wav'});
  await page.locator('.lpc-page .selection-controls input').nth(1).fill('0.05');await page.locator('.lpc-page .selection-controls input').nth(1).press('Tab');
  await page.getByRole('button',{name:'开始分析',exact:true}).click();
  const project=(await api('projects')).projects[0].id;let job;
  await until(async()=>{job=(await api('jobs?project_id='+project)).jobs.find(j=>j.operation==='lpc_analysis');if(job&&['failed','cancelled','interrupted'].includes(job.state))throw Error('LPC '+job.error_code);return job?.state==='succeeded';},90000);
  const [download]=await Promise.all([page.waitForEvent('download'),page.getByRole('button',{name:/下载.*选区 WAV/}).click()]);await download.saveAs(path.join(out,'result.wav'));assert.equal(await download.failure(),null);
  await page.screenshot({path:path.join(out,'web-lpc.png')});checks.push('browser-upload-real-linux-lpc-download');
  await context.setOffline(true);await page.evaluate(()=>window.dispatchEvent(new Event('online')));await page.getByText('连接暂时中断',{exact:false}).waitFor();
  await context.setOffline(false);await page.evaluate(()=>window.dispatchEvent(new Event('online')));await page.reload();await page.locator('#tool-sidebar').waitFor();assert.equal((await api('projects')).projects[0].id,project);checks.push('disconnect-reconnect-preserves-project');
  await page.getByRole('button',{name:'文件与空间',exact:true}).click();await page.getByRole('button',{name:'退出登录',exact:true}).click();await page.getByRole('button',{name:'登录',exact:true}).waitFor();
  assert.equal((await context.request.get(config.web_origin+'/api/v1/assets/'+job.result_manifest.files[0].id+'/content')).status(),401);
  bridge('webQuota',{owner_id:session.user.id,quota_bytes:1000000000});checks.push('logout-revokes-download-and-test-quota-restored');
  assert.deepEqual(errors,[]);
  await fs.writeFile(path.join(out,'manual-flow-'+browser+'.json'),JSON.stringify({success:true,browser,checks,errors,scope:'Windows browser and native FeedbackClient via TLS/SSH to isolated Linux PG/science/feedback; Qt UI interaction not automated; no public deployment'},null,2));
  console.log(JSON.stringify({success:true,checks}));
 }catch(e){await page.screenshot({path:path.join(out,'manual-failure.png')});throw e;}
 finally{await context.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
