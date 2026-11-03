
const path=require('node:path'),fs=require('node:fs/promises'),os=require('node:os'),assert=require('node:assert/strict');
const preferences=require(path.join(process.cwd(),'tests/e2e/preferences-fixture.cjs')).createPreferencesFixture();
(async()=>{
 const root=process.cwd(),out=process.argv[2],scratch=process.argv[3],captureMode=true;const config=JSON.parse(await fs.readFile(path.join(out,'config.json'),'utf8')),L=config.labels,locale=config.locale;
 const {preview}=await import(require('node:url').pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')).href);
 const server=await preview({root:path.join(root,'frontend'),build:{outDir:path.join(scratch,'frontend')},logLevel:'error',preview:{host:'127.0.0.1',port:0}});
 const origin=`http://127.0.0.1:${server.httpServer.address().port}`;
 const {chromium}=require(path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 let context,signed=false;const errors=[],checks=[],captures=[],requests=[];
 try{
  context=await chromium.launchPersistentContext(path.join(scratch,'chrome-profile'),{headless:false,executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',viewport:{width:1707,height:1067},deviceScaleFactor:1.5,colorScheme:'light',args:['--start-maximized']});
  await context.addInitScript(()=>{localStorage.setItem('ptb.v3.account-theme','light');localStorage.setItem('ptb.v3.theme',JSON.stringify('light'));localStorage.setItem('ptb.v3.palette',JSON.stringify('absolutely'));localStorage.setItem('ptb.v3.pageScale','100');});
  await context.route('**/server/assets/**',route=>route.continue({url:route.request().url().replace('/server/assets/','/assets/')}));
  await context.route('**/server/manual/**',route=>route.continue({url:route.request().url().replace('/server/manual/','/manual/')}));
  const owner='00000000-0000-4000-8000-000000000001',project='00000000-0000-4000-8000-000000000010';
  preferences.seed(owner,'APP',{theme:'light',palette:'absolutely',pageScale:100,uiLocale:locale});
  await context.route('**/api/v1/**',async route=>{
   const p=new URL(route.request().url()).pathname,now=Date.now()/1000;let body={},status=200;
   const session={user:{id:owner,username:'synthetic',role:'user',email_verified:true},csrf_token:'synthetic-only'};
   const prefs=preferences.reply(route.request(),owner);
   if(prefs){status=prefs.status;body=prefs.body;}
   else if(p.endsWith('/auth/options'))body={registration:true,password_recovery:true,ephemeral_workspaces:true};
   else if(p.endsWith('/auth/me')){body=signed?session:{detail:'authentication_required'};status=signed?200:401;}
   else if(p.endsWith('/auth/challenge'))body={csrf_token:'synthetic-only'};
   else if(p.endsWith('/auth/login')){signed=true;body=session;}
   else if(p.endsWith('/workspaces/default-project'))body={id:project,name:'默认项目',created_at:'2026-10-16T00:00:00Z'};
   else if(p.endsWith('/workspaces/presence'))body={online_accounts:1,window_seconds:90,measured_at:now};
   else if(p.endsWith('/workspaces/capacity-request'))body=null;
   else if(p.endsWith('/workspaces/server-load'))body={running_tasks:0,task_slots:1,waiting_tasks:0,cpu_percent:null,memory_percent:null,measured_at:now};
   else if(p.includes('/workspaces/'))body={id:project,generation:1,last_seen:now,expires_at:now+1800,grace_seconds:1800};
   else if(p.endsWith('/projects'))body={projects:[{id:project,name:'默认项目',created_at:'2026-10-16T00:00:00Z'}]};
   else if(p.endsWith('/storage/usage'))body={quota_bytes:1000000000,used_bytes:0,reserved_bytes:0,available_bytes:1000000000,ready:true,frozen:false,policy_version:2,workspace_policy_version:1,retention_seconds:1800,over_quota:false};
   else if(p.endsWith('/assets'))body={assets:[]};
   else if(p.endsWith('/jobs'))body={jobs:[]};
   else if(p.endsWith('/capabilities'))body={task_operations:['lpc_spectrum'],storage_operations:['upload','download'],algorithms:[]};
   await route.fulfill({status,contentType:'application/json',body:JSON.stringify(body)});
  });

  await context.route('https://**',route=>route.abort('blockedbyclient'));
  const page=await context.newPage();page.setDefaultTimeout(20000);page.on('pageerror',e=>errors.push(e.message));
  const cdp=await context.newCDPSession(page),window=await cdp.send('Browser.getWindowForTarget');
  await cdp.send('Browser.setWindowBounds',{windowId:window.windowId,bounds:{windowState:'maximized'}});
  await page.goto(origin+'/server/');
  await page.locator('.language-switch select').first().selectOption(locale);
  await page.waitForFunction(code=>document.documentElement.lang===code&&document.documentElement.dataset.fontLocale===code&&!document.querySelector('.language-switch select:disabled')&&document.fonts.status==='loaded',locale);
  async function capture(name){
   await page.evaluate(async()=>{await document.fonts.ready;await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));});
   const bounds=await cdp.send('Browser.getWindowBounds',{windowId:window.windowId});assert.equal(bounds.bounds.windowState,'maximized');
   const metrics=await page.evaluate(()=>({locale:document.documentElement.lang,fontLocale:document.documentElement.dataset.fontLocale,dpr:devicePixelRatio,theme:document.documentElement.dataset.theme,pageZoom:Number(getComputedStyle(document.documentElement).zoom)||1,fonts:document.fonts.status,fontFamily:getComputedStyle(document.querySelector('.language-atlas')||document.body).fontFamily,viewport:[innerWidth,innerHeight]}));
   assert.equal(metrics.locale,locale);assert.equal(metrics.fontLocale,locale);assert.equal(metrics.dpr,1.5);assert.equal(metrics.theme,'light');assert.equal(metrics.pageZoom,1);assert.equal(metrics.fonts,'loaded');
   await fs.writeFile(path.join(out,name+'.txt'),await page.locator('body').innerText());
   await page.screenshot({path:path.join(out,name+'-raw.png')});
   captures.push({name,file:name+'.png',maximized:bounds.bounds.windowState==='maximized',browserWindow:bounds.bounds,...metrics});
  }
  await page.getByRole('button',{name:L['注册账号'],exact:true}).waitFor();await capture('login');
  await page.getByRole('button',{name:L['注册账号'],exact:true}).click();await page.getByRole('heading',{name:L['创建研究者账号'],exact:true}).waitFor();await capture('register');
  await page.getByRole('button',{name:L['返回登录'],exact:true}).click();
  await page.getByLabel(L['登录名'],{exact:true}).fill('manual-demo');await page.getByLabel(L['密码'],{exact:true}).fill('manual-demo-only');await page.getByRole('button',{name:L['登录'],exact:true}).click();
  await page.locator('.announcements-panel').waitFor();await page.getByRole('button',{name:L['知道了'],exact:true}).click();await page.locator('.announcements-panel').waitFor({state:'detached'});
  await page.getByRole('button',{name:L['文件与空间'],exact:true}).click();await page.getByRole('heading',{name:L['项目文件'],exact:true}).waitFor();await page.getByRole('progressbar',{name:L['账号文件空间']}).waitFor();await capture('files');
  await page.getByRole('button',{name:L['任务记录'],exact:true}).click();await page.getByText(L['暂无任务'],{exact:true}).waitFor();await capture('tasks');
  await page.getByRole('button',{name:L['关闭对话框'],exact:true}).click();await page.getByRole('button',{name:L['容量与扩容'],exact:true}).click();await page.getByLabel(L['申请总额度（MB）'],{exact:true}).waitFor();await capture('capacity');
  await page.getByRole('button',{name:L['关闭对话框'],exact:true}).click();await page.getByRole('button',{name:L['设置'],exact:true}).click();await page.getByRole('button',{name:L['浅色'],exact:true}).click();await capture('settings');
  assert.deepEqual(errors,[]);checks.push('six-actual-Vue-scenes-with-real-language-control','synthetic-local-account-login-no-external-service','empty-project-files-and-tasks','actual-Chrome-CDP-maximized-window');
  await fs.writeFile(path.join(out,'browser.json'),JSON.stringify({scope:'Actual built Vue and visible isolated Chrome; local synthetic account HTTP only; no external account, email or server acceptance',locale,browser:context.browser().version(),checks,errors,captures},null,2));
 }catch(error){const page=context?.pages().at(-1);if(page){await page.screenshot({path:path.join(out,'failure.png')});await fs.writeFile(path.join(out,'failure.txt'),await page.locator('body').innerText());}throw error;}
 finally{await context?.close();await new Promise(resolve=>server.httpServer.close(resolve));}
})().catch(error=>{console.error(error);process.exitCode=1;});
