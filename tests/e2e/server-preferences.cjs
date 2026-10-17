// Real loopback HTTP/PostgreSQL. Every browser uses an owned, empty profile.
const path=require('node:path'),fs=require('node:fs/promises'),os=require('node:os'),assert=require('node:assert/strict');
(async()=>{
 const [origin,out,scratch]=process.argv.slice(2);assert.equal(path.dirname(scratch),out);
 const {chromium}=require(path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const contexts=[],errors=[],checks=[],requests=[];
 async function browser(name,engine='chrome'){
  const context=await chromium.launchPersistentContext(path.join(scratch,name),{headless:false,executablePath:engine==='edge'?'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe':'C:/Program Files/Google/Chrome/Application/chrome.exe',viewport:{width:1440,height:960}});contexts.push(context);
  const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));page.on('dialog',d=>void d.accept());page.on('response',r=>{if(r.url().includes('/api/v1/'))requests.push({path:new URL(r.url()).pathname,status:r.status()});});page.on('requestfailed',r=>requests.push({path:new URL(r.url()).pathname,failure:r.failure()?.errorText}));await page.goto(origin+'/server/');return page;
 }
 async function login(page,name='synthetic'){await page.getByLabel('登录名',{exact:true}).fill(name);await page.getByLabel('密码',{exact:true}).fill('synthetic-password-only');await page.getByRole('button',{name:'登录',exact:true}).click();await page.locator('#tool-sidebar').waitFor();}
 async function open(page,title){await page.bringToFront();await page.locator('#tool-sidebar').getByRole('button',{name:title,exact:true}).click();}
 async function lpc(page,value){await open(page,'LPC 谱图');await page.getByLabel('LPC 阶数',{exact:true}).waitFor();assert.equal(await page.getByLabel('LPC 阶数',{exact:true}).inputValue(),value);}
 async function saved(page,edit){await Promise.all([page.waitForResponse(r=>r.url().includes('/preferences/')&&r.request().method()==='PATCH'&&r.status()===200),edit()]);}
 async function logout(page){await page.getByRole('button',{name:'文件与空间',exact:true}).click();await page.getByRole('button',{name:'退出登录',exact:true}).click();await page.getByRole('button',{name:'登录',exact:true}).waitFor();}
 try{
  const a=await browser('chrome-a');await login(a);await lpc(a,'72');
  await saved(a,()=>a.getByLabel('LPC 阶数',{exact:true}).fill('84'));
  await open(a,'汉字转国际音标');await saved(a,()=>a.getByLabel('转换标准',{exact:true}).selectOption('吴宗济'));
  const b=await browser('edge-b','edge');await login(b);await lpc(b,'84');await open(b,'汉字转国际音标');assert.equal(await b.getByLabel('转换标准',{exact:true}).inputValue(),'吴宗济');checks.push('fresh-Edge-profile-restores-Chrome-account-settings-over-real-HTTP-PG');
  await lpc(b,'84');await saved(b,()=>b.getByLabel('LPC 阶数',{exact:true}).fill('85'));
  await open(a,'LPC 谱图');await saved(a,()=>a.getByLabel('LPC 幅度上限',{exact:true}).fill('45'));
  const merged=await a.evaluate(async()=>{const r=await fetch('/api/v1/preferences');return (await r.json()).modules.find(v=>v.module==='M04').values;});
  assert.equal(String(merged.draft.order),'85');assert.equal(String(merged.draft.amp_max_db),'45');
  await saved(a,()=>a.getByLabel('LPC 幅度上限',{exact:true}).fill('46'));
  const repeated=await a.evaluate(async()=>{const r=await fetch('/api/v1/preferences');return (await r.json()).modules.find(v=>v.module==='M04').values;});
  assert.equal(String(repeated.draft.order),'85');assert.equal(String(repeated.draft.amp_max_db),'46');checks.push('stale-device-repeated-saves-merge-only-edited-nested-fields');
  let blocked=true;await a.route('**/api/v1/preferences/M04',route=>blocked?route.abort('failed'):route.continue());
  await a.getByLabel('LPC 阶数',{exact:true}).fill('86');await a.getByText('设置尚未保存到服务器',{exact:false}).waitFor();
  blocked=false;await a.getByRole('button',{name:'重试保存',exact:true}).click();await a.getByText('设置尚未保存到服务器',{exact:false}).waitFor({state:'hidden'});checks.push('network-failure-is-visible-and-explicit-retry-commits');
  await a.evaluate(()=>localStorage.clear());await logout(a);checks.push('clearing-local-data-and-logout-do-not-delete-server-settings');
  await logout(b);await login(b,'settings-other');await lpc(b,'50');await logout(b);checks.push('other-account-in-reused-browser-receives-own-defaults');
  const c=await browser('chrome-c');await login(c);await lpc(c,'86');await open(c,'汉字转国际音标');assert.equal(await c.getByLabel('转换标准',{exact:true}).inputValue(),'吴宗济');
  const all=await c.evaluate(async()=>{const r=await fetch('/api/v1/preferences');return (await r.json()).modules;});assert.equal(all.length,19);checks.push('third-empty-profile-restores-all-18-module-records-and-global-preferences');
  await open(c,'LPC 谱图');for(const theme of ['light','dark']){await c.evaluate(t=>document.documentElement.dataset.theme=t,theme);await c.screenshot({path:path.join(out,'server-settings-'+theme+'.png')});}
  await logout(c);assert.deepEqual(errors,[]);
  await fs.writeFile(path.join(out,'result.json'),JSON.stringify({scope:'Windows independent visible Chrome and Edge profiles; real isolated PostgreSQL and HTTP; no production or physical second-device claim',checks,errors},null,2));console.log(JSON.stringify({checks,errors}));
 }catch(error){await fs.writeFile(path.join(out,'requests.json'),JSON.stringify(requests,null,2));const page=contexts.at(-1)?.pages().at(-1);if(page){await page.screenshot({path:path.join(out,'failure.png')});await fs.writeFile(path.join(out,'failure.txt'),await page.locator('body').innerText());}throw error;}
 finally{for(const context of contexts.reverse())await context.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
