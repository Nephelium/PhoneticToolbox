// Built Vue pages; synthetic account/workspace transport, owned browser profile.
const path=require('node:path'),fs=require('node:fs/promises'),os=require('node:os'),assert=require('node:assert/strict');
const preferences=require('./preferences-fixture.cjs').createPreferencesFixture();
(async()=>{
 const root=path.resolve(__dirname,'../..'),out=path.resolve(process.argv[2]),scratch=path.resolve(process.argv[3]);
 assert.equal(path.dirname(scratch),out);assert.ok(out.startsWith(path.join(root,'output/validation/web-settings')+path.sep));
 const {preview}=await import('../../frontend/node_modules/vite/dist/node/index.js');
 const server=await preview({root:path.join(root,'frontend'),build:{outDir:path.join(scratch,'frontend')},logLevel:'error',preview:{host:'127.0.0.1',port:0}});
 const origin=`http://127.0.0.1:${server.httpServer.address().port}`;
 const {chromium}=require(path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 let context,owner='00000000-0000-4000-8000-000000000001',project='00000000-0000-4000-8000-000000000010',signed=true;
 const errors=[],checks=[];
 try{
  context=await chromium.launchPersistentContext(path.join(scratch,'chrome-profile'),{headless:false,executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',viewport:{width:1440,height:960}});
  await context.route('**/server/assets/**',route=>route.continue({url:route.request().url().replace('/server/assets/','/assets/')}));
  await context.route('**/api/v1/**',async route=>{
   const p=new URL(route.request().url()).pathname,now=Date.now()/1000;let status=200,body={};
   const session={user:{id:owner,username:'synthetic',role:'user',email_verified:true},csrf_token:'synthetic-only'};
   const prefs=preferences.reply(route.request(),owner);
   if(prefs){status=prefs.status;body=prefs.body;}
   else if(p.endsWith('/auth/options'))body={registration:false,password_recovery:false,ephemeral_workspaces:true};
   else if(p.endsWith('/auth/me')){status=signed?200:401;body=signed?session:{detail:'authentication_required'};}
   else if(p.endsWith('/auth/login')){signed=true;body=session;}
   else if(p.endsWith('/auth/logout')){signed=false;status=204;}
   else if(p.endsWith('/auth/challenge'))body={csrf_token:'synthetic-only'};
   else if(p.endsWith('/workspaces/default-project'))body={id:project,name:'默认项目',created_at:'2026-10-16T00:00:00Z'};
   else if(p.endsWith('/workspaces/presence'))body={online_accounts:1,window_seconds:90,measured_at:now,server:{running_tasks:0,waiting_tasks:0,task_slots:1,measured_at:now}};
   else if(p.endsWith('/workspaces/capacity-request'))body=null;
   else if(p.includes('/workspaces/'))body={id:project,generation:1,last_seen:now,expires_at:now+1800,grace_seconds:1800};
   else if(p.endsWith('/projects'))body={projects:[{id:project,name:'默认项目',created_at:'2026-10-16T00:00:00Z'}]};
   else if(p.endsWith('/assets'))body={assets:[]};
   else if(p.endsWith('/jobs'))body={jobs:[]};
   else if(p.endsWith('/storage/usage'))body={quota_bytes:1000000000,used_bytes:0,reserved_bytes:0,available_bytes:1000000000,ready:true,frozen:false,policy_version:2,workspace_policy_version:1,retention_seconds:1800,over_quota:false};
   else if(p.endsWith('/capabilities'))body={task_operations:[],storage_operations:['upload','download'],algorithms:[]};
   await route.fulfill({status,contentType:'application/json',body:status===204?'':JSON.stringify(body)});
  });
  const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));page.on('dialog',d=>void d.accept());
  async function open(module){await page.locator('#tool-sidebar').getByRole('button',{name:module,exact:true}).click();}
  async function order(expected){await open('LPC 谱图');await page.getByLabel('LPC 阶数',{exact:true}).waitFor();assert.equal(await page.getByLabel('LPC 阶数',{exact:true}).inputValue(),expected);await page.getByText('参数尚未保存',{exact:true}).waitFor({state:'hidden'});}
  await page.goto(origin+'/server/');await order('50');
  await page.getByLabel('LPC 阶数',{exact:true}).fill('72');await page.waitForTimeout(300);
  await open('汉字转国际音标');await page.getByLabel('转换标准',{exact:true}).selectOption('汉语拼音');await page.waitForTimeout(300);
  await page.reload();await order('72');await open('汉字转国际音标');assert.equal(await page.getByLabel('转换标准',{exact:true}).inputValue(),'汉语拼音');checks.push('refresh-restores-analysis-and-conversion-settings');
  // Unmount flush must preserve the final edit even inside the debounce window.
  await open('LPC 谱图');await page.getByLabel('LPC 阶数',{exact:true}).fill('73');
  await page.evaluate(()=>{window.dispatchEvent(new PageTransitionEvent('pagehide'));});
  project='00000000-0000-4000-8000-000000000011';await page.reload();await order('73');checks.push('new-workspace-keeps-settings-and-empty-file-list');
  for(const theme of ['light','dark']){await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);await page.screenshot({path:path.join(out,'restored-'+theme+'.png')});}
  await page.getByRole('button',{name:'文件与空间',exact:true}).click();await page.getByRole('button',{name:'退出登录',exact:true}).click();await page.getByRole('button',{name:'登录',exact:true}).waitFor();
  project='00000000-0000-4000-8000-000000000012';await page.getByLabel('登录名',{exact:true}).fill('synthetic');await page.getByLabel('密码',{exact:true}).fill('synthetic-only');await page.getByRole('button',{name:'登录',exact:true}).click();await order('73');checks.push('explicit-logout-login-retains-settings');
  owner='00000000-0000-4000-8000-000000000002';project='00000000-0000-4000-8000-000000000020';await page.reload();await order('50');checks.push('another-account-keeps-independent-defaults');
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(out,'result.json'),JSON.stringify({scope:'Windows independent visible Chrome; built UI with synthetic HTTP; no production or physical-device evidence',checks,errors},null,2));
  console.log(JSON.stringify({checks,errors}));
 }catch(error){if(context){const page=context.pages().at(-1);if(page){await page.screenshot({path:path.join(out,'failure.png')});await fs.writeFile(path.join(out,'failure.txt'),await page.locator('body').innerText());}}throw error;}
 finally{await context?.close();await new Promise(resolve=>server.httpServer.close(resolve));}
})().catch(e=>{console.error(e);process.exitCode=1;});
