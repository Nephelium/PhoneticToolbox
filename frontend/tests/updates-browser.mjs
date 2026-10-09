// Actual maximized Chrome checks. Fixture uses a native-channel stand-in; no updates execute.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {createServer} from 'vite';
import vue from '@vitejs/plugin-vue';
const frontend=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),repo=path.dirname(frontend),folder=path.join(repo,'output','updates-ui',new Date().toISOString().replace(/[:.]/g,'-'));
await fs.mkdir(folder,{recursive:true});
const relative=path.relative(repo,folder).split(path.sep).join('/');
await fs.writeFile(path.join(folder,'index.html'),`<!doctype html><html lang="zh-CN"><meta charset="UTF-8"><title>更新界面验证</title><div id="app"></div><script type="module" src="/@fs/${path.join(folder,'main.ts').split(path.sep).join('/')}"></script></html>`);
await fs.writeFile(path.join(folder,'main.ts'),`
import {createApp,ref} from 'vue';
import Panel from '/src/components/UpdatePanel.vue';
import Notice from '/src/components/UpdateNotice.vue';
import {installUpdatesChannel} from '/src/platform/updates.ts';
import '/src/design/tokens.css';
const requests=[],cancelled=[],prefs={source:'auto',channel:'preview',autoCheck:true,currentVersion:'3.0.0-preview.1',packageKind:'portable',checkIntervalHours:6,applyAvailable:true};
const callbacks=()=>{const values=new Set();return{connect:fn=>values.add(fn),disconnect:fn=>values.delete(fn),emit:(id,value)=>values.forEach(fn=>fn(id,JSON.stringify(value)))}};
const ready=callbacks(),progress=callbacks();
const fixture={mode:'available',requests,cancelled,preferences:prefs,offline:()=>installUpdatesChannel(undefined)};
window.fixture=fixture;
const release={id:'owned-release',version:'3.0.0-preview.2',source:'server',notes:'修复说明书媒体阅读与更新提示。\\n用户设置和语料会保留。',publishedAt:'2026-10-05',packages:[{kind:'portable',name:'PhoneticToolbox-windows-x64-portable.zip',size:104857600,sha256:'a'.repeat(64),source:'server'}]};
installUpdatesChannel({ready,progress,cancel:id=>cancelled.push(id),request(id,payload){const {operation,args}=JSON.parse(payload);requests.push({id,operation,args});
  const reply=value=>{if(!cancelled.includes(id))ready.emit(id,{ok:true,value})};
  if(operation==='preferences')reply({...prefs});
  if(operation==='configure'){Object.assign(prefs,args);reply({...prefs});}
  if(operation==='acknowledge')reply({acknowledged:true});
  if(operation==='check')setTimeout(()=>reply({status:fixture.mode==='incomplete'?'incomplete':'available',currentVersion:prefs.currentVersion,channel:prefs.channel,checkedAt:1000000,region:{source:prefs.source==='auto'?'system':'manual',country:prefs.source==='auto'?'CN':null,label:prefs.source==='auto'?'系统地区线索':'手动选择',preferredSource:prefs.source==='github'?'github':'server'},preferredSource:prefs.source==='github'?'github':'server',sources:{server:fixture.mode==='incomplete'?{status:'error',message:'连接失败或超时，请稍后重试。'}:{status:'checked',version:release.version,count:1},github:{status:'error',message:'发布来源返回 HTTP 404。'}},candidate:fixture.mode==='incomplete'?null:release,shouldPrompt:!args.manual,packageKind:'portable'}),fixture.mode==='slow'?250:15);
  if(operation==='download'){let received=0;const timer=setInterval(()=>{received+=26214400;if(cancelled.includes(id)){clearInterval(timer);return;}progress.emit(id,{phase:'downloading',source:'server',received,total:104857600});if(received===104857600){clearInterval(timer);progress.emit(id,{phase:'verified',source:'server',received,total:104857600});reply({downloadId:'owned-download',name:release.packages[0].name,size:104857600,sha256:'a'.repeat(64),source:'server',kind:'portable',verified:true,applyAvailable:true});}},30);}
  if(operation==='apply')reply({started:true});
}});
const result=ref();createApp({components:{Panel,Notice},setup(){return{result};},template:'<Panel :initial-result="result" @checked="result=$event"/><Notice @checked="result=$event"/>'}).mount('#app');
`);
const server=await createServer({configFile:false,root:frontend,cacheDir:path.join(folder,'vite-cache'),optimizeDeps:{noDiscovery:true,include:['vue']},plugins:[vue()],resolve:{alias:{vue:path.join(frontend,'node_modules/vue/dist/vue.esm-bundler.js')}},server:{host:'127.0.0.1',port:0,fs:{allow:[repo]}},logLevel:'error'});
await server.listen();
const runtime=process.env.PTB_PLAYWRIGHT??path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs');
const {chromium}=await import(pathToFileURL(runtime)),browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:false,args:['--start-maximized','--no-first-run']}),context=await browser.newContext({viewport:null}),page=await context.newPage(),checks=[],errors=[];
page.on('pageerror',error=>errors.push(error.message));
try{
  const url=`http://127.0.0.1:${server.httpServer.address().port}/@fs/${path.join(folder,'index.html').split(path.sep).join('/')}`;
  await page.goto(url);const cdp=await context.newCDPSession(page),windowInfo=await cdp.send('Browser.getWindowForTarget');await cdp.send('Browser.setWindowBounds',{windowId:windowInfo.windowId,bounds:{windowState:'maximized'}});const bounds=await cdp.send('Browser.getWindowBounds',{windowId:windowInfo.windowId});assert.equal(bounds.bounds.windowState,'maximized');checks.push({name:'actual maximized Chrome',bounds:bounds.bounds,viewport:await page.evaluate(()=>({width:innerWidth,height:innerHeight}))});
  await page.getByLabel('新版本提示').waitFor();assert.equal(await page.evaluate(()=>window.fixture.requests.filter(r=>r.operation==='check').length),1);assert.equal(await page.evaluate(()=>window.fixture.requests.filter(r=>r.operation==='acknowledge').length),1);await page.getByRole('button',{name:'查看更新',exact:true}).click();assert.equal(await page.getByLabel('新版本提示').count(),0);checks.push({name:'startup notice checks once and acknowledges displayed version'});
  await page.getByRole('heading',{name:'发现新版本 3.0.0-preview.2'}).waitFor();assert.ok((await page.locator('.sources').innerText()).includes('HTTP 404'));assert.ok((await page.locator('.update-result').innerText()).includes('系统地区线索'));checks.push({name:'server release remains usable with GitHub failure and explicit OS region fallback'});
  await page.screenshot({path:path.join(folder,'updates-light-maximized.png')});await page.evaluate(()=>document.documentElement.dataset.theme='dark');await page.screenshot({path:path.join(folder,'updates-dark-maximized.png')});checks.push({name:'light and dark actual maximized screenshots'});
  await page.getByRole('button',{name:'下载更新',exact:true}).click();await page.getByRole('dialog').waitFor();assert.equal(await page.evaluate(()=>window.fixture.requests.filter(r=>r.operation==='download').length),0);await page.getByRole('button',{name:'暂不下载',exact:true}).click();assert.equal(await page.evaluate(()=>window.fixture.requests.filter(r=>r.operation==='download').length),0);await page.getByRole('button',{name:'下载更新',exact:true}).click();await page.getByRole('button',{name:'确认下载',exact:true}).click();await page.getByRole('heading',{name:'更新包已下载并通过校验'}).waitFor();assert.deepEqual(await page.evaluate(()=>window.fixture.requests.find(r=>r.operation==='download').args),{releaseId:'owned-release',packageKind:'portable',confirmed:true});checks.push({name:'cancelled confirmation does not download; confirmed native identifier download and progress'});
  await page.getByRole('button',{name:'退出并更新',exact:true}).click();assert.equal(await page.evaluate(()=>window.fixture.requests.filter(r=>r.operation==='apply').length),0);await page.getByRole('button',{name:'确认退出并更新',exact:true}).click();assert.deepEqual(await page.evaluate(()=>window.fixture.requests.find(r=>r.operation==='apply').args),{downloadId:'owned-download',confirmed:true});checks.push({name:'apply needs separate explicit confirmation and opaque verified download'});
  await page.getByLabel('下载来源').selectOption('github');await page.waitForFunction(()=>window.fixture.preferences.source==='github');await page.getByLabel('启动时检查更新').uncheck();await page.waitForFunction(()=>window.fixture.preferences.autoCheck===false);checks.push({name:'manual source and startup preference saved through native configure'});
  await page.evaluate(()=>window.fixture.mode='incomplete');await page.getByRole('button',{name:'检查更新',exact:true}).click();await page.getByRole('heading',{name:'检查未完整，请查看各来源状态'}).waitFor();assert.equal(await page.getByRole('heading',{name:'当前已是最新版本'}).count(),0);assert.equal(await page.locator('.source-error').count(),2);checks.push({name:'two-source failure remains unknown with both errors'});
  await page.evaluate(()=>window.fixture.mode='slow');await page.getByRole('button',{name:'检查更新',exact:true}).click();await page.getByRole('button',{name:'取消',exact:true}).click();await page.getByRole('alert').waitFor();assert.ok((await page.getByRole('alert').innerText()).includes('取消'));await page.waitForTimeout(300);assert.equal(await page.getByRole('heading',{name:'当前已是最新版本'}).count(),0);checks.push({name:'cancel check ignores late reply'});
  await page.evaluate(()=>window.fixture.offline());await page.getByText('在线更新在桌面版提供。网页工具可继续离线使用。',{exact:true}).waitFor();checks.push({name:'browser capability absence has no network fallback'});
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(folder,'report.json'),JSON.stringify({checks,errors,fixture:'native protocol stand-in; no release downloads or installers executed'},null,2));console.log(JSON.stringify({passed:checks.length,report:path.join(folder,'report.json')}));
}catch(error){await page.screenshot({path:path.join(folder,'failure-maximized.png')});await fs.writeFile(path.join(folder,'failure.json'),JSON.stringify({error:error.stack,checks,errors},null,2));throw error;}finally{await browser.close();await server.close();}
