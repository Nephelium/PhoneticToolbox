// Actual built UI; synthetic transport, no SMTP, credentials or production data.
const path=require('node:path'),fs=require('node:fs/promises'),os=require('node:os'),assert=require('node:assert/strict');
const preferences=require('./preferences-fixture.cjs').createPreferencesFixture();
(async()=>{
 const root=path.resolve(__dirname,'../..'),out=path.resolve(process.argv[2]),scratch=path.resolve(process.argv[3]),browser=process.argv[4]||'chrome';
 assert.ok(['chrome','edge'].includes(browser));
 assert.equal(path.dirname(scratch),out);assert.ok(out.startsWith(path.join(root,'output','validation','web-launch')+path.sep));
 const {preview}=await import('../../frontend/node_modules/vite/dist/node/index.js');
 const server=await preview({root:path.join(root,'frontend'),build:{outDir:path.join(root,'output/validation/web-launch/frontend')},logLevel:'error',preview:{host:'127.0.0.1',port:0}});
 const origin=`http://127.0.0.1:${server.httpServer.address().port}`;
 const {chromium}=require(path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 let context;const errors=[],requests=[],checks=[];
 try{
  context=await chromium.launchPersistentContext(path.join(scratch,browser+'-profile'),{headless:false,executablePath:browser==='chrome'?'C:/Program Files/Google/Chrome/Application/chrome.exe':'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',viewport:{width:1440,height:1000},downloadsPath:path.join(scratch,'downloads')});
  const owner='00000000-0000-4000-8000-000000000001',id='00000000-0000-4000-8000-000000000002';
  let signed=false,expired=false,workspaceId=id,logoutCount=0;
  let projects=[{id:'00000000-0000-4000-8000-000000000003',name:'合成研究 A',created_at:'2026-10-16T00:00:00Z'},
   {id:'00000000-0000-4000-8000-000000000004',name:'合成研究 B',created_at:'2026-10-16T00:00:00Z'}];
  await context.route('**/server/assets/**',route=>route.continue({url:route.request().url().replace('/server/assets/','/assets/')}));
  await context.route('**/api/v1/**',async route=>{
   const p=new URL(route.request().url()).pathname,method=route.request().method();requests.push({path:p,method});let body={},status=200;
   const session={user:{id:owner,username:'synthetic'},csrf_token:'synthetic-only',expires_at:'2099-01-01T00:00:00Z'};
   const prefs=preferences.reply(route.request(),owner);
   if(prefs){status=prefs.status;body=prefs.body;}
   else if(p.endsWith('/auth/options'))body={registration:true,password_recovery:true,ephemeral_workspaces:true};
   else if(p.endsWith('/auth/challenge'))body={csrf_token:'synthetic-form'};
   else if(p.endsWith('/auth/me')){body=signed?session:{detail:'authentication_required'};status=signed?200:401;}
   else if(p.endsWith('/auth/login')){signed=true;body=session;}
   else if(p.endsWith('/auth/logout')){signed=false;logoutCount++;status=204;}
   else if(p.endsWith('/auth/register')||p.endsWith('/auth/forgot-password')){body={accepted:true};status=202;}
   else if(p.endsWith('/auth/verify-email')||p.endsWith('/auth/reset-password'))status=204;
   else if(p.endsWith('/workspaces/presence'))body={online_accounts:1,window_seconds:90,measured_at:Date.now()/1000,server:{running_tasks:1,waiting_tasks:2,task_slots:1,cpu_percent:65.2,memory_percent:38.4,measured_at:Date.now()/1000}};
   else if(p.endsWith('/workspaces/default-project')){if(!projects.length)projects=[{id:'00000000-0000-4000-8000-000000000010',name:'默认项目',created_at:'2026-10-16T00:00:00Z'}];body=projects[0];}
   else if(p.endsWith('/workspaces/capacity-request'))body=null;
   else if(p.includes('/workspaces/')){
    if(p.endsWith('/open')&&route.request().postDataJSON()?.fresh){expired=false;workspaceId='00000000-0000-4000-8000-000000000009';projects=[];}
    if(expired){status=410;body={detail:'workspace_expired'};}
    else {const now=Date.now()/1000;body={id:workspaceId,generation:1,last_seen:now,expires_at:now+1800,grace_seconds:1800};}
   }
   else if(p.endsWith('/projects'))body={projects};
   else if(p.endsWith('/assets'))body={assets:[]};
   else if(p.endsWith('/jobs'))body={jobs:[]};
   else if(p.endsWith('/storage/usage'))body={quota_bytes:1000000000,used_bytes:0,reserved_bytes:0,available_bytes:1000000000,ready:true,frozen:false,policy_version:2,workspace_policy_version:1,retention_seconds:1800,over_quota:false};
   else if(p.endsWith('/capabilities'))body={task_operations:[],storage_operations:['upload','download'],algorithms:[]};
   await route.fulfill({status,contentType:'application/json',body:status===204?'':JSON.stringify(body)});
  });
  const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
  await page.goto(origin+'/server/');await page.getByRole('button',{name:'注册账号',exact:true}).waitFor();
  async function layout(name){
   await page.screenshot({path:path.join(out,name+'.png'),animations:'disabled',fullPage:true});
   assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),name+' horizontal overflow');
  }
  for(const width of [1440,390]){await page.setViewportSize({width,height:1000});for(const theme of ['light','dark']){await page.getByLabel('配色主题').selectOption(theme);await page.waitForFunction(t=>document.documentElement.dataset.theme===t,theme);await layout('login-'+theme+'-'+width);}}
  await page.setViewportSize({width:1440,height:1000});await page.getByLabel('配色主题').selectOption('light');
  await page.getByRole('button',{name:'注册账号',exact:true}).click();
  await page.getByLabel('登录名').fill('synthetic');await page.getByLabel('邮箱',{exact:true}).fill('synthetic@example.test');await page.locator('input[autocomplete="new-password"]').fill('synthetic-password-only');
  await page.getByRole('button',{name:'发送邮件',exact:true}).click();await page.getByText('请检查邮箱中的验证链接。',{exact:false}).waitFor();
  await page.screenshot({path:path.join(out,'registration-light.png')});checks.push('registration-form');
  await page.goto(origin+'/server/#verify='+'a'.repeat(43));await page.getByRole('button',{name:'确认验证',exact:true}).waitFor();assert.equal(new URL(page.url()).hash,'');
  await page.getByRole('button',{name:'确认验证',exact:true}).click();await page.getByText('邮箱验证成功，请登录。',{exact:true}).waitFor();checks.push('verify-fragment-cleared');
  await page.getByRole('button',{name:'忘记密码',exact:true}).click();await page.getByLabel('邮箱',{exact:true}).fill('synthetic@example.test');await page.getByRole('button',{name:'发送邮件',exact:true}).click();await page.getByText('若该邮箱已关联账号',{exact:false}).waitFor();
  await page.goto(origin+'/server/#reset='+'b'.repeat(43));await page.locator('input[autocomplete="new-password"]').fill('new-synthetic-password');await page.getByRole('button',{name:'保存新密码'}).click();await page.getByText('密码已更新，请使用新密码登录。').waitFor();checks.push('password-recovery-form');
  await page.getByLabel('登录名').fill('synthetic');await page.getByLabel('密码',{exact:true}).fill('synthetic-password-only');await page.getByRole('button',{name:'登录',exact:true}).click();
  await page.locator('#tool-sidebar').waitFor();checks.push('login-directly-enters-default-workbench');
  await page.getByRole('button',{name:'文件与空间',exact:true}).click();
  const manager=page.getByRole('dialog',{name:'文件与空间',exact:true});await manager.waitFor();
  for(const width of [1440,390]){await page.setViewportSize({width,height:800});for(const theme of ['light','dark']){
   await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);
   const box=await manager.boundingBox();assert.ok(box.x>=0 && box.y>=0 && box.y+box.height<=801 && box.x+box.width<=width+1);
   assert.ok(await manager.evaluate(e=>e.scrollWidth<=e.clientWidth+1));await manager.screenshot({path:path.join(out,'files-'+theme+'-'+width+'.png'),animations:'disabled'});
  }}
  await manager.getByText('高级管理 · 分组整理不同研究',{exact:true}).click();await page.getByLabel('查看项目').selectOption(projects[1].id);await page.getByLabel('查看项目').selectOption(projects[0].id);
  await manager.getByRole('button',{name:'关闭对话框',exact:true}).click();
  await page.setViewportSize({width:1440,height:900});await page.evaluate(()=>document.documentElement.dataset.theme='light');
  await page.getByRole('button',{name:'容量与扩容',exact:true}).click();await page.getByRole('dialog',{name:'个人容量',exact:true}).waitFor();await layout('capacity-dialog');await page.getByRole('button',{name:'关闭对话框',exact:true}).click();
  await page.getByRole('button',{name:'计算 1 / 1 · 等待 2',exact:true}).click();await page.getByRole('dialog',{name:'服务器状态',exact:true}).waitFor();await page.getByText('65.2%',{exact:true}).waitFor();await layout('server-status');await page.getByRole('button',{name:'关闭对话框',exact:true}).click();
  checks.push('compact-files-light-dark-wide-narrow-and-public-load');
  await page.locator('#tool-sidebar').getByRole('button',{name:/MFA/}).click();
  await page.getByRole('link',{name:'下载 Windows 桌面版'}).waitFor();await page.screenshot({path:path.join(out,'mfa-web.png')});checks.push('shared-shell-mfa-guidance');
  await page.locator('#tool-sidebar').getByRole('button',{name:/国际音标表/}).click();await page.getByLabel('国际音标文本').fill('synthetic private IPA ɑ');
  await page.getByRole('button',{name:/^关闭 国际音标表/}).click();
  const confirm=page.getByRole('dialog',{name:'保存音标草稿？',exact:true});await confirm.waitFor();
  for(const width of [1440,390]){await page.setViewportSize({width,height:1000});for(const theme of ['light','dark']){
   await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);
   const box=await confirm.boundingBox();assert.ok(box.width<=440 && box.height>=200 && box.x>=0 && box.x+box.width<=width+1);
   assert.ok(await confirm.evaluate(e=>e.scrollWidth<=e.clientWidth+1));
   await confirm.screenshot({path:path.join(out,'save-confirm-'+theme+'-'+width+'.png'),animations:'disabled'});
  }}
  await page.getByRole('button',{name:'取消关闭',exact:true}).click();assert.equal(await page.getByLabel('国际音标文本').inputValue(),'synthetic private IPA ɑ');
  await page.setViewportSize({width:1440,height:1000});await page.evaluate(()=>document.documentElement.dataset.theme='light');checks.push('compact-save-dialog-cancel-preserves-draft');
  await page.locator('#tool-sidebar').getByRole('button',{name:'感知实验',exact:true}).click();await page.getByRole('tab',{name:'资源与恢复',exact:true}).waitFor();
  await page.waitForFunction(async()=> (await indexedDB.databases()).filter(d=>d.name?.startsWith('ptb-m15-local-v1:web:')).length===1);
  // Navigation simulates refresh/close; neither is an explicit logout.
  await page.goto(origin+'/server/');assert.equal(logoutCount,0);await page.getByRole('button',{name:'文件与空间',exact:true}).click();await page.getByText('高级管理 · 分组整理不同研究',{exact:true}).click();await page.getByLabel('查看项目').selectOption(projects[1].id);await page.getByRole('button',{name:'切换到此项目工作台',exact:true}).click();
  await page.locator('#tool-sidebar').getByRole('button',{name:/国际音标表/}).click();await page.getByLabel('国际音标文本').waitFor();assert.equal(await page.getByLabel('国际音标文本').inputValue(),'');checks.push('project-ipa-isolation-refresh-preserved');
  await page.locator('#tool-sidebar').getByRole('button',{name:'感知实验',exact:true}).click();await page.getByRole('tab',{name:'资源与恢复',exact:true}).waitFor();
  await page.waitForFunction(async()=> (await indexedDB.databases()).filter(d=>d.name?.startsWith('ptb-m15-local-v1:web:')).length===2);checks.push('perception-project-databases-isolated');
  expired=true;await page.evaluate(()=>window.dispatchEvent(new Event('online')));await page.getByRole('button',{name:'开始新的工作区'}).waitFor();
  await page.screenshot({path:path.join(out,'expired-workspace.png')});assert.equal(logoutCount,0);await page.getByRole('button',{name:'开始新的工作区'}).click();await page.locator('#tool-sidebar').waitFor();checks.push('expiry-explicit-fresh-start');
  await page.getByRole('button',{name:'文件与空间',exact:true}).click();await page.getByRole('button',{name:'退出登录'}).click();await page.getByRole('button',{name:'登录',exact:true}).waitFor();assert.equal(logoutCount,1);checks.push('explicit-logout');
  const count=requests.filter(r=>r.path.includes('/workspaces/')).length;
  await page.goto(origin+'/');await page.locator('#tool-sidebar').getByRole('button',{name:/MFA/}).click();assert.equal(await page.getByRole('link',{name:'下载 Windows 桌面版'}).count(),0);assert.equal(requests.filter(r=>r.path.includes('/workspaces/')).length,count);checks.push('local-route-keeps-original-module');
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(out,'checks.json'),JSON.stringify({scope:'Built UI with synthetic HTTP; visible independent '+browser,checks,errors},null,2));console.log(JSON.stringify({out,browser,checks,errors}));
 }catch(error){await fs.writeFile(path.join(out,'failure.json'),JSON.stringify({message:String(error),errors,requests},null,2));throw error;}
 finally{if(context)await context.close();await new Promise(resolve=>server.httpServer.close(resolve));}
})().catch(e=>{console.error(e);process.exitCode=1;});
